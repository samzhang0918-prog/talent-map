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
import os
import re
import threading
import time
import unicodedata
import urllib.parse
from collections import Counter, defaultdict

import networkx as nx
import requests
from networkx.algorithms.community import louvain_communities

DBLP_API = "https://dblp.org/search/publ/api"
PAGE_SIZE = 100          # dblp 单页上限
MAX_OFFSET = 10000       # dblp 分页硬上限，超过返回空
CRAWL_DELAY = 4.5        # dblp robots.txt 要求 Crawl-delay: 4，留点余量
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
SESSION.headers.update({
    # 挂到公网长期跑之后，请求量不再是"自用脚本"那种偶发流量，向 dblp/OpenAlex
    # 表明身份和联系方式是对公开 API 的基本礼貌，出问题时对方也能找到人而不是直接封 IP。
    # 没配置 CONTACT_URL 时退化成普通浏览器 UA，本地自用不受影响。
    "User-Agent": (
        f"talent-map/1.0 (+{_UA_CONTACT}) "
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ) if _UA_CONTACT else (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
})

# Web 应用给每个搜索请求单独开一个线程（见 app.py 的 worker），如果限速只在各自
# 线程里 sleep，N 个人同时搜就等于把 dblp 的请求频率放大 N 倍——集体违反了
# robots.txt 的 Crawl-delay: 4，而且被限的是出口 IP，会连累所有使用者。
# 所以把节流做成进程级的：全局同一时刻只允许一个 dblp 请求在飞，且与上一个请求
# 至少间隔 min_gap 秒。单人使用时行为跟原来完全一致（等待量为 0）。
_DBLP_GATE = threading.Lock()
_DBLP_LAST = 0.0


def dblp_get(url, min_gap, timeout=30):
    """按全局节奏发一个 dblp 请求。并发调用会自动排队，不会叠加请求频率。"""
    global _DBLP_LAST
    with _DBLP_GATE:
        wait = min_gap - (time.monotonic() - _DBLP_LAST)
        if wait > 0:
            time.sleep(wait)
        try:
            return SESSION.get(url, timeout=timeout)
        finally:
            # 以请求"结束"时刻计时，宁可比 Crawl-delay 更保守
            _DBLP_LAST = time.monotonic()


def fetch_page(query, offset, max_retries=5):
    """
    取一页搜索结果，返回 (hits, total)；彻底失败返回 (None, None)。

    dblp 的 500 是间歇性的（实测同一 offset 第一次 500、重试就 200），
    所以这里退避重试，并把"失败"和"没数据"区分开交给调用方判断。
    """
    url = f"{DBLP_API}?q={urllib.parse.quote_plus(query)}&format=json&h={PAGE_SIZE}&f={offset}"
    for attempt in range(max_retries):
        try:
            resp = dblp_get(url, CRAWL_DELAY)
            if resp.status_code == 200:
                hits = resp.json()["result"]["hits"]
                return hits.get("hit", []), int(hits.get("@total", 0))
            time.sleep(CRAWL_DELAY * (attempt + 1))
        except (requests.RequestException, ValueError, KeyError):
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
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "").strip()
OPENALEX_BATCH = 50      # 一次用 filter=doi:a|b|c 查这么多篇


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
        return {}, {"papers_with_doi": 0, "papers_matched": 0}

    dois = list(by_doi)
    enriched = defaultdict(lambda: {"citations": 0, "institutions": Counter(), "topics": Counter()})
    matched_papers = 0

    for i in range(0, len(dois), OPENALEX_BATCH):
        chunk = dois[i:i + OPENALEX_BATCH]
        params = {
            "filter": "doi:" + "|".join(chunk),
            "per_page": OPENALEX_BATCH * 2,
            "select": "doi,cited_by_count,authorships,topics",
        }
        if OPENALEX_MAILTO:
            params["mailto"] = OPENALEX_MAILTO
        try:
            resp = SESSION.get(OPENALEX_API, params=params, timeout=40)
            if resp.status_code != 200:
                continue
            results = resp.json().get("results", [])
        except (requests.RequestException, ValueError, KeyError):
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

    coverage = {
        "papers_total": len(papers),
        "papers_with_doi": len(dois),
        "papers_matched": matched_papers,
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
    """
    papers, total_available = collect_papers(query, max_papers, on_progress)

    if not papers:
        return {"error": f"没有检索到「{query}」相关论文。dblp 的多个关键词之间是 AND 关系，"
                         f"词越多结果越少，建议只用 1~2 个核心词。"}

    if on_progress:
        on_progress(len(papers), len(papers), "正在构建合作网络...")
    G, total_authors, skipped = build_network(papers, min_papers)

    if G.number_of_nodes() == 0:
        return {"error": f"这批论文涉及 {total_authors} 位作者，但没有人在该方向发过 "
                         f"{min_papers} 篇以上。可以调低「最少发文数」，或调高检索论文数。"}

    coverage = {}
    if enrich:
        enriched, coverage = enrich_from_openalex(papers, on_progress)
        apply_enrichment(G, enriched)

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
        "enriched": bool(enrich),
        "with_institution": with_inst,
        "with_citations": with_cite,
        **coverage,
    }
    payload["error"] = None
    payload["_graph"] = G   # 供深度模式接着扩展，序列化给前端之前会被去掉
    return payload


def deep_expand(fast_payload, seed_count=15, per_seed=15, on_progress=None):
    """
    在快速结果之上做滚雪球扩展，产出连通性好得多的人才网络。
    传入 search_topic 的返回值，返回一份新的 payload。
    """
    G = fast_payload.get("_graph")
    if G is None or G.number_of_nodes() == 0:
        return {"error": "没有可用于扩展的检索结果"}

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
