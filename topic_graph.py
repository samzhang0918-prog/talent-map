"""
按论文方向检索并构建人才合作网络的核心逻辑。

命令行工具（talent_map_by_topic.py）和 Web 应用（app.py）都用这个模块，
区别只在于结果怎么呈现：命令行写成静态 HTML 文件，Web 应用推成 JSON 给前端实时渲染。

数据源：dblp 官方论文检索 API（https://dblp.org/search/publ/api）。
用它而不是抓 dblp 的 /pid/*.xml 页面，是因为 dblp 的 robots.txt 写了 `Disallow: /*.xml`，
而 search API 是官方文档里公开提供给程序调用的接口。

实测到的 API 脾气（都已处理）：
1. 会毫无规律地返回 HTTP 500，跟请求本身无关，同一个 URL 重试就好。所以必须区分
   "重试后仍失败"和"真的没有更多数据"——把 500 直接当结束条件会静默漏数据。
2. 分页最多翻到 offset 10000，再往后返回空列表。
3. 多个关键词之间是 AND（全部命中才算），词堆得越多结果越少：实测 "BEV" 有 4979 篇，
   "BEV perception" 剩 135 篇，"BEV perception autonomous driving" 只剩 19 篇。
"""

import colorsys
import hashlib
import json
import os
import re
import threading
import time
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from urllib.parse import urlparse

import networkx as nx
import requests
from networkx.algorithms.community import louvain_communities

DBLP_API = "https://dblp.org/search/publ/api"
PAGE_SIZE = 100          # dblp 单页上限
MAX_OFFSET = 10000       # dblp 分页硬上限，超过返回空
CRAWL_DELAY = 4.5        # dblp robots.txt 曾要求 Crawl-delay: 4，留点余量
# 滚雪球阶段每个种子一次请求，比翻页更容易触发限流：实测 4.5 秒时 20 个种子有一半拿到
# 500 失败，放宽到 6 秒后 8/8 全部成功，所以这里单独用更保守的间隔。
SNOWBALL_DELAY = 6.0

# 一篇论文作者太多时（竞赛报告、大型综述），作者之间的"合作"信号很弱，
# 而且两两连边会让边数爆炸（20 个作者就是 190 条边），所以跳过这类论文。
MAX_AUTHORS_PER_PAPER = 15

# 预置的智驾 / CV 常用方向，只是快捷入口，任意关键词都能搜。
PRESET_TOPICS = [
    "BEV perception", "occupancy prediction", "end-to-end driving",
    "trajectory prediction", "motion planning", "3D object detection",
    "LiDAR point cloud", "visual SLAM", "multimodal fusion",
    "vision language model", "world model", "driving simulation",
    "pedestrian detection", "lane detection", "depth estimation",
    "semantic segmentation", "reinforcement learning driving", "corner case",
]

SESSION = requests.Session()
_UA_CONTACT = os.environ.get("CONTACT_URL", "").strip()
_UA = (
    f"talent-map/1.0 (+{_UA_CONTACT}) "
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
) if _UA_CONTACT else (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SESSION.headers.update({
    # 挂到公网长期跑之后，请求量不再是"自用脚本"那种偶发流量，向 dblp/OpenAlex
    # 表明身份和联系方式是对公开 API 的基本礼貌，出问题时对方也能找到人而不是直接封 IP。
    # 没配置 CONTACT_URL 时退化成普通浏览器 UA，本地自用不受影响。
    "User-Agent": _UA,
    # 明确要 JSON；Anubis 挑战页仍可能以 text/html 返回，下面会识别并完成 PoW。
    "Accept": "application/json, text/javascript;q=0.9, */*;q=0.1",
    "Accept-Language": "en-US,en;q=0.9",
})

# Web 应用给每个搜索请求单独开一个线程（见 app.py 的 worker），如果限速只在各自
# 线程里 sleep，N 个人同时搜就等于把 dblp 的请求频率放大 N 倍——集体违反了
# robots.txt 的 Crawl-delay: 4，而且被限的是出口 IP，会连累所有使用者。
# 所以把节流做成进程级的：全局同一时刻只允许一个 dblp 请求在飞，且与上一个请求
# 至少间隔 min_gap 秒。单人使用时行为跟原来完全一致（等待量为 0）。
_DBLP_GATE = threading.Lock()
_DBLP_LAST = 0.0

# 成功 JSON 页缓存：同样的 q/offset 在短时间内重复搜索时不必再打 dblp。
_DBLP_RESP_CACHE = {}
_DBLP_RESP_CACHE_TTL = 20 * 60
_DBLP_RESP_CACHE_LOCK = threading.Lock()
_DBLP_RESP_CACHE_MAX = 64

# Anubis（Techaro）保护：dblp 对疑似自动化流量返回 PoW 挑战页，浏览器解完后
# 拿 auth cookie。这是站点正常的 soft challenge，用同样的 HTTP 客户端完成即可。
_ANUBIS_CHALLENGE_RE = re.compile(
    r'<script id="anubis_challenge" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_ANUBIS_AUTHED = False


def _dblp_cache_get(url):
    now = time.time()
    with _DBLP_RESP_CACHE_LOCK:
        hit = _DBLP_RESP_CACHE.get(url)
        if not hit:
            return None
        ts, payload = hit
        if now - ts > _DBLP_RESP_CACHE_TTL:
            del _DBLP_RESP_CACHE[url]
            return None
        return payload


def _dblp_cache_put(url, payload):
    with _DBLP_RESP_CACHE_LOCK:
        if len(_DBLP_RESP_CACHE) >= _DBLP_RESP_CACHE_MAX:
            # 丢掉最旧的一条
            oldest = min(_DBLP_RESP_CACHE.items(), key=lambda kv: kv[1][0])[0]
            del _DBLP_RESP_CACHE[oldest]
        _DBLP_RESP_CACHE[url] = (time.time(), payload)


def _anubis_meets(digest: bytes, difficulty: int) -> bool:
    """与 Anubis sha256-purejs worker 相同的难度判定。"""
    prefix = difficulty // 2
    odd = difficulty % 2 != 0
    for i in range(prefix):
        if digest[i] != 0:
            return False
    if odd and (digest[prefix] >> 4) != 0:
        return False
    return True


def _solve_anubis_pow(random_data: str, difficulty: int, timeout_s: float = 60.0):
    """暴力找 nonce，使 sha256(randomData + nonce) 满足 difficulty。返回 (hash_hex, nonce)。"""
    deadline = time.monotonic() + timeout_s
    nonce = 0
    data_prefix = random_data  # str concat, matching JS `data + nonce`
    while time.monotonic() < deadline:
        digest = hashlib.sha256(f"{data_prefix}{nonce}".encode("utf-8")).digest()
        if _anubis_meets(digest, difficulty):
            return digest.hex(), nonce
        nonce += 1
    raise TimeoutError(f"Anubis PoW timed out after {timeout_s}s (difficulty={difficulty})")


def _is_anubis_challenge(resp) -> bool:
    if resp is None:
        return False
    text_head = resp.text[:4000] if resp.text else ""
    if "anubis_challenge" in text_head or "Making sure you're not a bot" in text_head:
        return True
    ct = (resp.headers.get("content-type") or "").lower()
    # 要 JSON 却拿到 HTML，多半是挑战页或其他拦截
    if "html" in ct and "json" not in ct:
        return "anubis" in text_head.lower() or "within.website" in text_head
    return False


def _pass_anubis_challenge(resp, original_url) -> bool:
    """解析挑战、算 PoW、打 pass-challenge，把 auth cookie 写入 SESSION。"""
    global _ANUBIS_AUTHED
    m = _ANUBIS_CHALLENGE_RE.search(resp.text or "")
    if not m:
        print("[dblp] Anubis HTML without challenge JSON", flush=True)
        return False
    try:
        payload = json.loads(m.group(1))
        challenge = payload["challenge"]
        rules = payload.get("rules") or {}
        random_data = challenge["randomData"]
        difficulty = int(rules.get("difficulty") or challenge.get("difficulty") or 5)
        cid = challenge["id"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[dblp] Anubis challenge parse failed: {exc}", flush=True)
        return False

    t0 = time.monotonic()
    try:
        hash_hex, nonce = _solve_anubis_pow(random_data, difficulty)
    except TimeoutError as exc:
        print(f"[dblp] {exc}", flush=True)
        return False
    elapsed_ms = max(1, int((time.monotonic() - t0) * 1000))
    print(f"[dblp] Anubis PoW solved difficulty={difficulty} nonce={nonce} "
          f"in {elapsed_ms}ms", flush=True)

    parsed = urlparse(original_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    pass_url = f"{origin}/.within.website/x/cmd/anubis/api/pass-challenge"
    try:
        # 不跟随重定向：只要 Set-Cookie；跟着走有时会踩到 dblp 间歇 500。
        pass_resp = SESSION.get(
            pass_url,
            params={
                "id": cid,
                "response": hash_hex,
                "nonce": str(nonce),
                "redir": "/",
                "elapsedTime": str(elapsed_ms),
            },
            timeout=30,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        print(f"[dblp] Anubis pass-challenge failed: {exc}", flush=True)
        return False

    authed = any(k.startswith("dblp_org-auth") for k in SESSION.cookies.keys())
    if not authed:
        # 少数情况下 cookie 名不同，再看响应是否带 Location
        print(f"[dblp] Anubis pass status={pass_resp.status_code} "
              f"cookies={list(SESSION.cookies.keys())}", flush=True)
    _ANUBIS_AUTHED = authed or pass_resp.status_code in (200, 302, 303, 307, 308)
    return _ANUBIS_AUTHED


def dblp_get(url, min_gap, timeout=30):
    """按全局节奏发一个 dblp 请求；遇 Anubis 挑战则完成 PoW 后重试一次。"""
    global _DBLP_LAST
    with _DBLP_GATE:
        wait = min_gap - (time.monotonic() - _DBLP_LAST)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = SESSION.get(url, timeout=timeout)
            if _is_anubis_challenge(resp):
                print("[dblp] Anubis challenge detected, solving PoW…", flush=True)
                if _pass_anubis_challenge(resp, url):
                    # 挑战刚通过后 dblp 偶发对紧接的业务请求回 500，稍等再取一次。
                    time.sleep(0.6)
                    resp = SESSION.get(url, timeout=timeout)
                    if resp.status_code >= 500:
                        time.sleep(1.2)
                        resp = SESSION.get(url, timeout=timeout)
                    if _is_anubis_challenge(resp):
                        print("[dblp] Still challenged after PoW; cookie may be rejected",
                              flush=True)
            return resp
        finally:
            # 以请求"结束"时刻计时，宁可比 Crawl-delay 更保守
            _DBLP_LAST = time.monotonic()


def fetch_page(query, offset, max_retries=5):
    """
    取一页搜索结果，返回 (hits, total)；彻底失败返回 (None, None)。

    dblp 的 500 是间歇性的（实测同一 offset 第一次 500、重试就 200），
    所以这里退避重试，并把"失败"和"没数据"区分开交给调用方判断。
    另外：2025 起 dblp 前置了 Anubis PoW，HTTP 200 + HTML 挑战页不能当成功。
    """
    url = f"{DBLP_API}?q={urllib.parse.quote_plus(query)}&format=json&h={PAGE_SIZE}&f={offset}"
    cached = _dblp_cache_get(url)
    if cached is not None:
        return cached

    for attempt in range(max_retries):
        try:
            resp = dblp_get(url, CRAWL_DELAY)
            if _is_anubis_challenge(resp):
                print(f"[dblp] 第 {attempt+1}/{max_retries} 次仍是 Anubis HTML", flush=True)
                time.sleep(CRAWL_DELAY * (attempt + 1))
                continue
            ct = (resp.headers.get("content-type") or "").lower()
            if resp.status_code == 200 and "json" in ct:
                hits = resp.json()["result"]["hits"]
                payload = (hits.get("hit", []) or [], int(hits.get("@total", 0)))
                _dblp_cache_put(url, payload)
                return payload
            # 临时诊断日志：定位公开部署后 dblp 持续失败到底是什么原因
            # （403 封禁 / 429 限流 / 500 抽风，处理方式完全不同）。
            print(f"[dblp] 第 {attempt+1}/{max_retries} 次失败 status={resp.status_code} "
                  f"ct={ct!r} body={resp.text[:200]!r}", flush=True)
            time.sleep(CRAWL_DELAY * (attempt + 1))
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[dblp] 第 {attempt+1}/{max_retries} 次异常 "
                  f"{type(exc).__name__}: {exc}", flush=True)
            time.sleep(CRAWL_DELAY * (attempt + 1))
    return None, None


def collect_papers(query, max_papers, on_progress=None):
    """
    按方向关键词分页抓论文。

    on_progress(fetched, total, message) 会在每页之后调用，用于向上层汇报进度
    （Web 应用靠它把进度实时推给浏览器，命令行则用来打印日志）。
    """
    papers = []
    offset = 0
    total = None
    consecutive_failures = 0

    while offset < min(max_papers, MAX_OFFSET):
        hits, page_total = fetch_page(query, offset)

        if hits is None:
            consecutive_failures += 1
            if on_progress:
                on_progress(len(papers), total or 0, f"第 {offset // PAGE_SIZE + 1} 页重试失败，跳过")
            if consecutive_failures >= 3:
                if on_progress:
                    on_progress(len(papers), total or 0, "连续多页失败，用已抓到的数据继续")
                break
            offset += PAGE_SIZE
            continue

        consecutive_failures = 0
        if total is None:
            total = page_total
            if total == 0:
                break

        papers.extend(hits)
        if on_progress:
            planned = min(max_papers, MAX_OFFSET, total)
            on_progress(len(papers), planned, f"已获取 {len(papers)} / {planned} 篇论文")

        if len(hits) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        # 翻页间隔由 dblp_get 的全局节流统一负责，这里不再单独 sleep

    return papers, (total or 0)


OPENALEX_API = "https://api.openalex.org/works"
# OpenAlex 的 polite pool 标识：带上邮箱能拿到更稳定的配额和更快的响应。
# 不是密钥，不带也能用，只是会被归到匿名池。用环境变量而不是写死在代码里，
# 免得把私人邮箱一起提交进仓库。
# 2026 起匿名池按出口 IP 共享日预算，生产用法应配置免费 API key：
#   OPENALEX_API_KEY / Authorization: Bearer …（见 https://help.openalex.org/api/authentication/）
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "").strip()
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()
OPENALEX_BATCH = 50      # 一次用 filter=doi:a|b|c 查这么多篇

CROSSREF_API = "https://api.crossref.org/works"
CROSSREF_MAILTO = os.environ.get(
    "CROSSREF_MAILTO",
    OPENALEX_MAILTO or "talent-map@users.noreply.github.com",
).strip()
CROSSREF_PAGE = 100


def _openalex_params(extra=None):
    params = dict(extra or {})
    if OPENALEX_MAILTO:
        params.setdefault("mailto", OPENALEX_MAILTO)
    if OPENALEX_API_KEY:
        params.setdefault("api_key", OPENALEX_API_KEY)
    return params


def _openalex_headers():
    if OPENALEX_API_KEY:
        return {"Authorization": f"Bearer {OPENALEX_API_KEY}"}
    return {}


def _author_pid_from_name(name, prefix="xref"):
    """无权威 ID 时，用归一化姓名生成稳定伪 pid（仅用于本图内合作边）。"""
    tokens = sorted(normalize_person(name))
    slug = "-".join(tokens) if tokens else "unknown"
    return f"{prefix}:{slug}"


def _openalex_author_pid(authorship):
    author = authorship.get("author") or {}
    oid = (author.get("id") or "").rstrip("/").split("/")[-1]
    name = author.get("display_name") or ""
    if oid:
        return f"oa:{oid}", name
    if name:
        return _author_pid_from_name(name, prefix="oa"), name
    return None, None


def openalex_work_to_paper(work):
    """把 OpenAlex work 转成 parse_authors / build_network 能吃的伪 dblp 记录。"""
    title = work.get("display_name") or work.get("title") or ""
    year = work.get("publication_year") or ""
    doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
    authors = []
    inline = {}
    for a in work.get("authorships") or []:
        pid, name = _openalex_author_pid(a)
        if not pid or not name:
            continue
        authors.append({"@pid": pid, "text": name})
        inline[pid] = {
            "institutions": [inst.get("display_name") for inst in (a.get("institutions") or [])
                             if inst.get("display_name")],
        }
    if not authors:
        return None
    topics = [t.get("display_name") for t in (work.get("topics") or [])[:3] if t.get("display_name")]
    return {
        "info": {
            "title": title,
            "year": str(year) if year else "",
            "doi": doi,
            "authors": {"author": authors},
        },
        "_source": "openalex",
        "_citations": int(work.get("cited_by_count") or 0),
        "_topics": topics,
        "_author_meta": inline,
    }


def crossref_work_to_paper(work):
    """把 Crossref work 转成伪 dblp 记录。"""
    title = (work.get("title") or [""])[0] if isinstance(work.get("title"), list) else (work.get("title") or "")
    if not title:
        return None
    date_parts = (
        (work.get("published-print") or {}).get("date-parts")
        or (work.get("published-online") or {}).get("date-parts")
        or (work.get("published") or {}).get("date-parts")
        or (work.get("created") or {}).get("date-parts")
        or [[]]
    )
    year = date_parts[0][0] if date_parts and date_parts[0] else ""
    doi = (work.get("DOI") or "").lower()
    authors = []
    for a in work.get("author") or []:
        given, family = a.get("given") or "", a.get("family") or ""
        name = f"{given} {family}".strip() or a.get("name") or ""
        if not name:
            continue
        orcid = (a.get("ORCID") or "").rstrip("/").split("/")[-1]
        pid = f"orcid:{orcid}" if orcid else _author_pid_from_name(name, prefix="xref")
        authors.append({"@pid": pid, "text": name})
    if not authors:
        return None
    return {
        "info": {
            "title": title,
            "year": str(year) if year else "",
            "doi": doi,
            "authors": {"author": authors},
        },
        "_source": "crossref",
        "_citations": int(work.get("is-referenced-by-count") or 0),
        "_topics": [],
        "_author_meta": {},
    }


def apply_inline_paper_meta(G, papers):
    """把 fallback 源自带的引用/机构/主题写进图（无需再打 OpenAlex enrich）。"""
    per_pid = defaultdict(lambda: {"citations": 0, "institutions": Counter(), "topics": Counter()})
    for paper in papers:
        cites = int(paper.get("_citations") or 0)
        topics = paper.get("_topics") or []
        meta = paper.get("_author_meta") or {}
        for pid, name in parse_authors(paper):
            rec = per_pid[pid]
            rec["citations"] += cites
            for t in topics:
                rec["topics"][t] += 1
            for inst in (meta.get(pid) or {}).get("institutions") or []:
                rec["institutions"][inst] += 1
    apply_enrichment(G, per_pid)
    return {
        "enrich_status": "inline",
        "enrich_message": "",
        "degraded": False,
        "papers_with_doi": sum(1 for p in papers if p.get("info", {}).get("doi")),
        "papers_matched": len(papers),
    }


def collect_papers_openalex(query, max_papers, on_progress=None):
    """关键词检索 OpenAlex；失败返回 ([], 0, reason)。"""
    papers = []
    total = 0
    per_page = min(50, max_papers)
    cursor = "*"
    reason = ""
    while len(papers) < max_papers:
        params = _openalex_params({
            "search": query,
            "per_page": min(per_page, max_papers - len(papers)),
            "cursor": cursor,
            "select": "id,doi,display_name,publication_year,cited_by_count,authorships,topics",
        })
        try:
            resp = SESSION.get(OPENALEX_API, params=params, headers=_openalex_headers(), timeout=40)
        except requests.RequestException as exc:
            reason = f"OpenAlex 网络错误：{type(exc).__name__}"
            break
        if resp.status_code == 429:
            reason = "OpenAlex 配额用尽(429)，可配置 OPENALEX_API_KEY"
            break
        if resp.status_code != 200:
            reason = f"OpenAlex HTTP {resp.status_code}"
            break
        try:
            data = resp.json()
        except ValueError:
            reason = "OpenAlex 返回非 JSON"
            break
        results = data.get("results") or []
        if total == 0:
            total = int((data.get("meta") or {}).get("count") or 0)
        if not results:
            break
        for work in results:
            paper = openalex_work_to_paper(work)
            if paper:
                papers.append(paper)
                if len(papers) >= max_papers:
                    break
        if on_progress:
            on_progress(len(papers), min(max_papers, total or max_papers),
                        f"OpenAlex 备用源已获取 {len(papers)} 篇")
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.35)
    return papers, total or len(papers), reason


def collect_papers_crossref(query, max_papers, on_progress=None):
    """关键词检索 Crossref（礼貌池）；失败返回 ([], 0, reason)。"""
    papers = []
    total = 0
    reason = ""
    offset = 0
    while len(papers) < max_papers:
        rows = min(CROSSREF_PAGE, max_papers - len(papers))
        params = {
            "query": query,
            "rows": rows,
            "offset": offset,
            "mailto": CROSSREF_MAILTO,
        }
        try:
            resp = SESSION.get(CROSSREF_API, params=params, timeout=40)
        except requests.RequestException as exc:
            reason = f"Crossref 网络错误：{type(exc).__name__}"
            break
        if resp.status_code == 429:
            reason = "Crossref 限流(429)"
            time.sleep(2)
            # 再试一次
            try:
                resp = SESSION.get(CROSSREF_API, params=params, timeout=40)
            except requests.RequestException as exc:
                reason = f"Crossref 网络错误：{type(exc).__name__}"
                break
        if resp.status_code != 200:
            reason = f"Crossref HTTP {resp.status_code}"
            break
        try:
            data = resp.json()
        except ValueError:
            reason = "Crossref 返回非 JSON"
            break
        msg = data.get("message") or {}
        if total == 0:
            total = int(msg.get("total-results") or 0)
        items = msg.get("items") or []
        if not items:
            break
        for work in items:
            paper = crossref_work_to_paper(work)
            if paper:
                papers.append(paper)
                if len(papers) >= max_papers:
                    break
        if on_progress:
            on_progress(len(papers), min(max_papers, total or max_papers),
                        f"Crossref 备用源已获取 {len(papers)} 篇")
        if len(items) < rows:
            break
        offset += rows
        time.sleep(0.2)
    return papers, total or len(papers), reason


def collect_papers_with_fallback(query, max_papers, on_progress=None):
    """
    先 dblp；若整页失败或零结果且像被拦，再试 OpenAlex，再试 Crossref。
    返回 (papers, total, source, status_message)
    source: "dblp" | "openalex" | "crossref"
    """
    papers, total = collect_papers(query, max_papers, on_progress)
    if papers:
        return papers, total, "dblp", ""

    dblp_msg = "dblp 未返回论文（可能被 Anubis/网络拦截，或关键词无命中）"
    if on_progress:
        on_progress(0, 0, "dblp 无结果，尝试 OpenAlex 备用源…")
    oa_papers, oa_total, oa_reason = collect_papers_openalex(query, max_papers, on_progress)
    if oa_papers:
        msg = f"主源 dblp 失败，已改用 OpenAlex（{len(oa_papers)} 篇）"
        return oa_papers, oa_total, "openalex", msg

    if on_progress:
        on_progress(0, 0, "OpenAlex 不可用，尝试 Crossref 备用源…")
    cr_papers, cr_total, cr_reason = collect_papers_crossref(query, max_papers, on_progress)
    if cr_papers:
        detail = oa_reason or dblp_msg
        msg = f"主源 dblp 失败（{detail}），已改用 Crossref（{len(cr_papers)} 篇）"
        return cr_papers, cr_total, "crossref", msg

    parts = [dblp_msg]
    if oa_reason:
        parts.append(oa_reason)
    if cr_reason:
        parts.append(cr_reason)
    return [], 0, "none", "；".join(parts)


def normalize_person(name):
    """把人名归一成词集合，用于同一篇论文内跨库匹配作者。"""
    # dblp 会给同名学者加消歧编号（"Zhe Wang 0006"），OpenAlex 那边没有，先去掉
    name = re.sub(r"\s+\d{4}$", "", name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return frozenset(re.findall(r"[a-z]+", name.lower()))


def enrich_from_openalex(papers, on_progress=None):
    """
    用论文 DOI 批量向 OpenAlex 换取引用数、署名机构、研究主题。

    为什么按 DOI 而不是按人名查：这是绕开同名问题的关键。之前 fetch_affiliations.py
    按人名查 OpenAlex，"Jianping Shi" 有 5 个同名候选、根本判不出是哪一个，443 条结果里
    只有 38 条可用。改成按 DOI 精确定位到论文，再在这一篇论文内部把作者对上号，
    候选集就只有这篇论文的几位作者，实测 94% 的 dblp 署名能正确匹配。

    成本上也差得远：按人名查用的是全文 search，一次 10 credits；按 DOI 用 filter
    精确匹配，一次请求带 50 个 DOI 只花 1 credit。每天 1000 credits 够查 5 万篇论文。

    返回 {dblp_pid: {"citations": int, "institutions": Counter, "topics": Counter}}
    """
    by_doi = {}
    for p in papers:
        doi = (p.get("info", {}).get("doi") or "").lower()
        if doi:
            by_doi.setdefault(doi, []).append(p)

    if not by_doi:
        return {}, {
            "papers_with_doi": 0,
            "papers_matched": 0,
            "enrich_status": "skipped",
            "enrich_message": "这批论文缺少 DOI，无法向 OpenAlex 补充引用/机构",
            "degraded": False,
        }

    dois = list(by_doi)
    enriched = defaultdict(lambda: {"citations": 0, "institutions": Counter(), "topics": Counter()})
    matched_papers = 0
    batches_ok = 0
    batches_fail = 0
    fail_reasons = []  # e.g. HTTP 429 / network error — surfaced to UI as degraded

    for i in range(0, len(dois), OPENALEX_BATCH):
        chunk = dois[i:i + OPENALEX_BATCH]
        params = {
            "filter": "doi:" + "|".join(chunk),
            "per_page": OPENALEX_BATCH * 2,
            "select": "doi,cited_by_count,authorships,topics",
        }
        params = _openalex_params(params)
        try:
            resp = SESSION.get(OPENALEX_API, params=params, headers=_openalex_headers(), timeout=40)
            if resp.status_code != 200:
                batches_fail += 1
                # Keep a short readable reason; quota exhaustion is the common public-trial case
                reason = f"HTTP {resp.status_code}"
                if resp.status_code == 429:
                    reason = "配额用尽(429)"
                elif resp.status_code in (401, 403):
                    reason = f"接口拒绝({resp.status_code})"
                if reason not in fail_reasons:
                    fail_reasons.append(reason)
                continue
            results = resp.json().get("results", [])
            batches_ok += 1
        except (requests.RequestException, ValueError, KeyError) as exc:
            batches_fail += 1
            reason = type(exc).__name__
            if reason not in fail_reasons:
                fail_reasons.append(reason)
            continue

        for work in results:
            doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
            dblp_papers = by_doi.get(doi)
            if not dblp_papers:
                continue
            matched_papers += 1

            citations = work.get("cited_by_count") or 0
            topics = [t["display_name"] for t in (work.get("topics") or [])[:3]]

            # 建同篇论文内的 OpenAlex 作者索引，再和 dblp 的署名对上号
            oa_authors = {}
            for a in work.get("authorships", []):
                key = normalize_person(a.get("author", {}).get("display_name", ""))
                oa_authors[key] = [inst["display_name"] for inst in a.get("institutions", [])]

            for dblp_paper in dblp_papers:
                for pid, name in parse_authors(dblp_paper):
                    rec = enriched[pid]
                    rec["citations"] += citations
                    for t in topics:
                        rec["topics"][t] += 1
                    insts = oa_authors.get(normalize_person(name))
                    if insts:
                        for inst in insts:
                            rec["institutions"][inst] += 1

        if on_progress:
            on_progress(min(i + OPENALEX_BATCH, len(dois)), len(dois),
                        f"正在补充引用量/机构/主题（{min(i + OPENALEX_BATCH, len(dois))}/{len(dois)} 篇）")
        time.sleep(0.3)   # OpenAlex 没有硬性 crawl-delay，但别把请求打太密

    # Status for the UI: main graph stays usable even when enrich degrades.
    if batches_ok == 0 and batches_fail > 0:
        enrich_status = "degraded"
        enrich_message = "引用/机构补充已降级：配额用尽或接口失败（" + "、".join(fail_reasons[:3]) + "）"
        degraded = True
    elif batches_fail > 0:
        enrich_status = "partial"
        enrich_message = "引用/机构补充部分失败（" + "、".join(fail_reasons[:3]) + "），已展示拿到的部分"
        degraded = True
    else:
        enrich_status = "ok"
        enrich_message = ""
        degraded = False

    coverage = {
        "papers_total": len(papers),
        "papers_with_doi": len(dois),
        "papers_matched": matched_papers,
        "enrich_status": enrich_status,
        "enrich_message": enrich_message,
        "degraded": degraded,
        "enrich_batches_ok": batches_ok,
        "enrich_batches_fail": batches_fail,
    }
    return dict(enriched), coverage


def parse_authors(paper):
    """从一条 dblp 记录里取出 [(pid, name), ...]。"""
    authors = paper.get("info", {}).get("authors", {}).get("author", [])
    if isinstance(authors, dict):
        authors = [authors]
    out = []
    for a in authors:
        pid, name = a.get("@pid"), a.get("text")
        if pid and name:
            out.append((pid, name))
    return out


def build_network(papers, min_papers):
    """
    从论文列表构建合作网络。

    节点 = 作者，papers 属性 = 他在这批论文里的发文数（衡量在该方向的活跃度）；
    边 = 两人共同署名过同一篇论文，weight = 共同论文数。
    """
    paper_count = Counter()
    author_names = {}
    coauthor_count = Counter()
    author_papers = defaultdict(list)
    skipped_big = 0

    for paper in papers:
        authors = parse_authors(paper)
        if not authors:
            continue
        if len(authors) > MAX_AUTHORS_PER_PAPER:
            skipped_big += 1
            continue

        info = paper.get("info", {})
        title, year = info.get("title", ""), info.get("year", "")

        for pid, name in authors:
            paper_count[pid] += 1
            author_names[pid] = name
            if len(author_papers[pid]) < 5:
                author_papers[pid].append({"year": year, "title": title})

        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a, b = sorted([authors[i][0], authors[j][0]])
                coauthor_count[(a, b)] += 1

    # 只保留在该方向发过 >= min_papers 篇的作者：发 1 篇的人占绝大多数，
    # 留着会把图糊成一片，也说明不了这人是该方向的核心研究者。
    kept = {pid for pid, c in paper_count.items() if c >= min_papers}

    G = nx.Graph()
    for pid in kept:
        G.add_node(pid, label=author_names[pid],
                   papers=paper_count[pid], samples=author_papers[pid],
                   citations=0, institution="", topics=[])
    for (a, b), w in coauthor_count.items():
        if a in kept and b in kept:
            G.add_edge(a, b, weight=w)

    return G, len(paper_count), skipped_big


def apply_enrichment(G, enriched):
    """把 OpenAlex 补充到的引用量 / 机构 / 主题写进图的节点属性。"""
    for pid, rec in enriched.items():
        if pid not in G:
            continue
        G.nodes[pid]["citations"] = rec["citations"]
        if rec["institutions"]:
            # 一个人可能在不同论文署不同单位（换过工作、双聘），取出现次数最多的那个
            G.nodes[pid]["institution"] = rec["institutions"].most_common(1)[0][0]
        G.nodes[pid]["topics"] = [t for t, _ in rec["topics"].most_common(3)]


def generate_palette(n):
    """按黄金角在色轮上取色，保证相邻编号的颜色也能一眼区分。"""
    out = []
    for i in range(max(n, 1)):
        hue = (i * 137.508) % 360
        r, g, b = colorsys.hls_to_rgb(hue / 360, 0.60, 0.55)
        out.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return out


def detect_communities(G):
    """按合作紧密度聚类研究团体，返回 (节点->团体号, 节点->颜色, 图例数据)。"""
    if G.number_of_nodes() == 0:
        return {}, {}, []
    if G.number_of_edges() == 0:
        palette = generate_palette(1)
        return ({n: 0 for n in G}, {n: palette[0] for n in G},
                [{"id": 0, "color": palette[0], "label": "独立学者", "size": G.number_of_nodes()}])

    communities = sorted(louvain_communities(G, weight="weight", seed=42), key=len, reverse=True)
    palette = generate_palette(len(communities))

    node_community, node_color, legend = {}, {}, []
    for cid, members in enumerate(communities):
        color = palette[cid]
        # 用团体里发文最多的人命名，比"社区 7"这种编号有信息量。
        # 深度模式下滚雪球带进来的合作者 papers 都是 0，用度数兜底，
        # 保证即使某个团体里没有种子也能选出有代表性的人来命名。
        hub = max(members, key=lambda n: (G.nodes[n]["papers"], G.degree(n)))
        for n in members:
            node_community[n] = cid
            node_color[n] = color
        legend.append({"id": cid, "color": color,
                       "label": f"{G.nodes[hub]['label']} 团队", "size": len(members)})
    return node_community, node_color, legend


def graph_to_payload(G, node_community, node_color, legend):
    """把网络转成前端 vis-network 能直接吃的 JSON 结构。"""
    max_papers = max((d["papers"] for _, d in G.nodes(data=True)), default=1) or 1

    nodes = []
    for node, attrs in G.nodes(data=True):
        is_seed = attrs.get("is_seed", True)
        if is_seed:
            # 节点大小随该方向发文数增长，一眼看出谁是主力
            size = 12 + 28 * (attrs["papers"] / max_papers)
        else:
            # 滚雪球带进来的合作者没有该方向发文数，统一给个小尺寸，
            # 视觉上和"该方向核心学者"区分开
            size = 10
        nodes.append({
            "id": node,
            "label": attrs["label"],
            "papers": attrs["papers"],
            "samples": attrs["samples"],
            "is_seed": is_seed,
            "citations": attrs.get("citations", 0),
            "institution": attrs.get("institution", ""),
            "topics": attrs.get("topics", []),
            "community": node_community[node],
            "color": node_color[node],
            "size": round(size, 1),
        })

    edges = [{"from": s, "to": t, "value": d["weight"]} for s, t, d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges, "legend": legend}


def fetch_author_papers(name, max_retries=3):
    """
    按学者姓名取他的论文列表（用于滚雪球扩展）。

    这里故意用纯姓名做全文检索，而不是 dblp 的 `author:Xxx_Yyy:` 精确语法：
    实测 author: 前缀对带中间名缩写的姓名会直接返回 0 条（"Ian R. Lane" 查不到，
    去掉缩写的 "Ian Lane" 才有 127 条），而纯姓名检索对各种姓名形态都稳定。
    混进来的同名他人论文，靠调用方用 pid 精确过滤即可剔除，不影响准确性。
    """
    url = f"{DBLP_API}?q={urllib.parse.quote_plus(name)}&format=json&h={PAGE_SIZE}"
    for attempt in range(max_retries):
        try:
            resp = dblp_get(url, SNOWBALL_DELAY)
            if resp.status_code == 200:
                return resp.json()["result"]["hits"].get("hit", [])
            time.sleep(SNOWBALL_DELAY * (attempt + 1))
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(SNOWBALL_DELAY * (attempt + 1))
    return None


def snowball_expand(G, seed_count=15, per_seed=15, on_progress=None):
    """
    以该方向发文最多的学者为种子做滚雪球扩展，把他们的合作者也纳入图中。

    为什么需要这一步：直接按方向检索建出来的图是高度碎片化的（实测 "lane detection"
    123 人散成 30 个互不相连的小组，最大的组才 16 人）。原因不是采集方式有问题，
    而是同一方向的研究组之间本来就各写各的、没有共同作者，连不起来是客观事实。
    实测过"只在这些核心学者之间补边"的做法，查了 top 20 人的全部论文只补出 1 条边，
    基本无效 —— 他们是真的互不相识。

    滚雪球之所以有效，是因为每个新节点都是顺着一条边被引入的，连通性是构造出来的：
    同样的方向实测从 30 个组降到 4 个组，最大组占比从 13% 升到 51%。

    代价是图里会混入不做这个方向的人（种子的其他领域合作者），所以节点上用 is_seed
    区分：种子是该方向的核心学者，其余是被带进来的协作圈。
    """
    seeds = [pid for pid, _ in sorted(
        G.nodes(data=True), key=lambda x: -x[1]["papers"])[:seed_count]]

    S = nx.Graph()
    for pid in seeds:
        d = G.nodes[pid]
        # 把补充来的引用量/机构/主题一并带过去，否则深度模式会把这些数据丢掉
        S.add_node(pid, label=d["label"], papers=d["papers"],
                   samples=d["samples"], is_seed=True,
                   citations=d.get("citations", 0),
                   institution=d.get("institution", ""),
                   topics=d.get("topics", []))

    done = 0
    for pid in seeds:
        name = G.nodes[pid]["label"]
        hits = fetch_author_papers(name)
        done += 1
        if on_progress:
            on_progress(done, len(seeds),
                        f"正在扩展 {name} 的合作网络（{done}/{len(seeds)}）")
        if hits is None:
            continue

        collab = Counter()
        names = {}
        for paper in hits:
            authors = parse_authors(paper)
            pids = [p for p, _ in authors]
            # 用 pid 精确确认这篇论文确实是本人的（纯姓名检索会混入同名他人）
            if pid not in pids or len(authors) > MAX_AUTHORS_PER_PAPER:
                continue
            for p, nm in authors:
                if p != pid:
                    collab[p] += 1
                    names[p] = nm

        for p, w in collab.most_common(per_seed):
            if p not in S:
                S.add_node(p, label=names[p], papers=0, samples=[], is_seed=False,
                           citations=0, institution="", topics=[])
            S.add_edge(pid, p, weight=w)

        # 种子之间的间隔同样交给 dblp_get 的全局节流

    # 种子之间在原方向图里已有的合作关系也要保留，否则会丢掉真实的边
    for a, b, d in G.edges(data=True):
        if a in S and b in S and not S.has_edge(a, b):
            S.add_edge(a, b, weight=d["weight"])

    return S


def search_topic(query, max_papers=300, min_papers=2, on_progress=None, enrich=True):
    """
    完整流程：联网检索 -> 建图 -> 补充属性 -> 聚类 -> 输出前端可用的数据。
    返回 dict，其中 error 非空表示这次检索没有可用结果。

    主源是 dblp（含 Anubis PoW 自动通过）；若出口 IP 仍被硬拦或日配额耗尽，
    自动降级到 OpenAlex / Crossref，并在 stats 里写明 data_source / source_message。
    """
    papers, total_available, source, source_message = collect_papers_with_fallback(
        query, max_papers, on_progress)

    if not papers:
        detail = source_message or "所有数据源均无结果"
        return {"error": f"没有检索到「{query}」相关论文。{detail}。"
                         f"dblp 多个关键词之间是 AND 关系，建议只用 1~2 个核心词；"
                         f"若长期失败可配置 OPENALEX_API_KEY 启用 OpenAlex 备用源。"}

    if on_progress:
        on_progress(len(papers), len(papers), "正在构建合作网络...")
    G, total_authors, skipped = build_network(papers, min_papers)

    if G.number_of_nodes() == 0:
        return {"error": f"这批论文涉及 {total_authors} 位作者，但没有人在该方向发过 "
                         f"{min_papers} 篇以上。可以调低「最少发文数」，或调高检索论文数。"}

    coverage = {
        "enrich_status": "skipped",
        "enrich_message": "",
        "degraded": False,
    }
    if source in ("openalex", "crossref"):
        # fallback 记录已带引用/机构，直接写入；再打 enrich 只会浪费配额
        coverage = apply_inline_paper_meta(G, papers)
        if source_message:
            coverage["degraded"] = True
            coverage["enrich_message"] = source_message
            coverage["enrich_status"] = "fallback"
    elif enrich:
        enriched, coverage = enrich_from_openalex(papers, on_progress)
        apply_enrichment(G, enriched)
        if source_message:
            coverage["degraded"] = True
            coverage["enrich_message"] = (
                (coverage.get("enrich_message") + "；" if coverage.get("enrich_message") else "")
                + source_message
            )

    if on_progress:
        on_progress(len(papers), len(papers), "正在聚类研究团体...")
    node_community, node_color, legend = detect_communities(G)

    payload = graph_to_payload(G, node_community, node_color, legend)
    with_inst = sum(1 for _, d in G.nodes(data=True) if d.get("institution"))
    with_cite = sum(1 for _, d in G.nodes(data=True) if d.get("citations"))
    payload["stats"] = {
        "query": query,
        "papers_fetched": len(papers),
        "papers_available": total_available,
        "scholars": G.number_of_nodes(),
        "relations": G.number_of_edges(),
        "communities": len(legend),
        "total_authors": total_authors,
        "skipped_big_papers": skipped,
        "min_papers": min_papers,
        "mode": "fast",
        "components": nx.number_connected_components(G),
        # 如实记录补充数据的覆盖率：OpenAlex 并非每篇论文都有机构元数据，
        # 界面上要让人知道有多少人是真拿到了数据，而不是默认全都有
        "enriched": bool(enrich) or source in ("openalex", "crossref"),
        "with_institution": with_inst,
        "with_citations": with_cite,
        "data_source": source,
        "source_message": source_message,
        **coverage,
    }
    payload["error"] = None
    payload["_graph"] = G   # 供深度模式接着扩展，序列化给前端之前会被去掉
    # 非 dblp 源没有可靠的作者主页滚雪球接口，深度扩展会空转；交给 deep_expand 自行判断
    payload["_data_source"] = source
    return payload


def deep_expand(fast_payload, seed_count=15, per_seed=15, on_progress=None):
    """
    在快速结果之上做滚雪球扩展，产出连通性好得多的人才网络。
    传入 search_topic 的返回值，返回一份新的 payload。
    """
    G = fast_payload.get("_graph")
    if G is None or G.number_of_nodes() == 0:
        return {"error": "没有可用于扩展的检索结果"}

    source = fast_payload.get("_data_source") or (fast_payload.get("stats") or {}).get("data_source") or "dblp"
    if source != "dblp":
        return {"error": f"当前结果来自备用源「{source}」，深度扩展依赖 dblp 作者检索，已跳过。"
                         f"请待 dblp 恢复后重试深度模式。"}

    S = snowball_expand(G, seed_count, per_seed, on_progress)

    if on_progress:
        on_progress(seed_count, seed_count, "正在聚类研究团体...")
    node_community, node_color, legend = detect_communities(S)

    payload = graph_to_payload(S, node_community, node_color, legend)
    base = fast_payload["stats"]
    payload["stats"] = {
        **base,
        "scholars": S.number_of_nodes(),
        "relations": S.number_of_edges(),
        "communities": len(legend),
        "mode": "deep",
        "components": nx.number_connected_components(S),
        "seed_count": sum(1 for _, d in S.nodes(data=True) if d.get("is_seed")),
        # 覆盖率要按扩展后的图重算：滚雪球带进来的合作者没有这些补充数据，
        # 沿用快速阶段的数字会虚报
        "with_institution": sum(1 for _, d in S.nodes(data=True) if d.get("institution")),
        "with_citations": sum(1 for _, d in S.nodes(data=True) if d.get("citations")),
    }
    payload["error"] = None
    return payload
