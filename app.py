"""
人才地图 Web 应用：在网页里输入论文方向关键词，实时联网检索并渲染人才关系图。

启动：
    /usr/bin/python3 app.py
然后浏览器打开 http://127.0.0.1:8000

为什么要做进度推送（SSE）而不是普通的一问一答接口：
dblp 的 robots.txt 要求 Crawl-delay 4 秒，抓 300 篇论文（3 页）就要十几秒，
抓 1000 篇要 40 多秒。这么长的等待如果只给一个转圈动画，用户完全不知道发生了什么、
还要等多久，很容易以为卡死了。所以检索接口用 Server-Sent Events 把
"已获取 200/300 篇"这样的进度实时推给浏览器。

缓存：同一个关键词 + 参数在 30 分钟内重复搜索会直接命中内存缓存，
既让用户秒出结果，也避免对 dblp 反复发起相同的抓取。
时间范围（全部 / 近 5 年 / 近 3 年）只影响建图，不影响抓取：抓到的原始论文和
OpenAlex 补充数据按「关键词 + 论文数」单独缓存，切换范围时直接在这份数据上过滤重算。
"""

import asyncio
import json
import threading

import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import topic_graph

app = FastAPI(title="论文方向人才地图")

# vis-network 从本地 lib/ 提供，不走 CDN：CDN 一旦被网络环境挡住，
# 页面会静默变成一片空白（图例、统计都不出来，因为 render 第一行就用到了 vis）。
# 项目里本来就有 pyvis 带的这份 9.1.2 副本，直接复用，顺带离线也能跑。
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if os.path.isdir(_LIB_DIR):
    app.mount("/lib", StaticFiles(directory=_LIB_DIR), name="lib")

# 进程内缓存，重启即失效，够用了。按用途分三份，各自有容量上限（LRU 淘汰）+ 30 分钟 TTL：
#   result：快速结果（关键词 + 论文数 + 最少发文 + 时间范围）
#   raw   ：抓到的原始论文 + OpenAlex 补充（关键词 + 论文数），切换时间范围时复用；单条最大
#           （papers=2000 时几 MB），所以上限最小
#   deep  ：深度模式结果
# 上限可用环境变量覆盖：TALENT_MAP_RESULT_CACHE_MAX（1–1024）/ TALENT_MAP_RAW_CACHE_MAX（1–64）/
# TALENT_MAP_DEEP_CACHE_MAX（1–512）；dblp 响应缓存见 topic_graph（TALENT_MAP_DBLP_CACHE_MAX，1–1024）。
_CACHE_TTL = 30 * 60
# 默认值与允许范围见 topic_graph.CACHE_LIMITS（超界封顶，非法值回落默认）
RESULT_CACHE_MAX = topic_graph.cache_limit("TALENT_MAP_RESULT_CACHE_MAX")
RAW_CACHE_MAX = topic_graph.cache_limit("TALENT_MAP_RAW_CACHE_MAX")
DEEP_CACHE_MAX = topic_graph.cache_limit("TALENT_MAP_DEEP_CACHE_MAX")
_RESULT_CACHE = topic_graph.TTLLRUCache(RESULT_CACHE_MAX, _CACHE_TTL, name="result")
_RAW_CACHE = topic_graph.TTLLRUCache(RAW_CACHE_MAX, _CACHE_TTL, name="raw")
_DEEP_CACHE = topic_graph.TTLLRUCache(DEEP_CACHE_MAX, _CACHE_TTL, name="deep")


class _CacheGroup:
    """只读汇总视图（keys / len），方便排障时一次看全部 key；读写直接用各自的缓存。"""

    def __init__(self, *caches):
        self.caches = caches

    def keys(self):
        return [k for c in self.caches for k in c.keys()]

    def __len__(self):
        return sum(len(c) for c in self.caches)


_CACHE = _CacheGroup(_RESULT_CACHE, _RAW_CACHE, _DEEP_CACHE)

# 公开部署时的参数上限，比 topic_graph 里 dblp 自身的技术上限（10000 篇 / 40 秒子）
# 收紧很多——见 /api/search 里的注释。
PUBLIC_MAX_PAPERS = 2000
PUBLIC_MAX_SEEDS = 15

# 正在检索中的访客 IP，用来拒绝同一个人开多个并发检索（见 /api/search）。
_INFLIGHT_IPS = set()
_INFLIGHT_LOCK = threading.Lock()


def get_client_ip(request):
    """
    取访客的真实 IP，而不是反向代理自己的 IP。

    部署在 Render / Hugging Face Spaces（或者本地用 cloudflared 隧道分享）时，
    uvicorn 接到的 TCP 连接来自平台的反向代理，request.client.host 拿到的是
    那个代理的地址——所有访客在这个值上看起来都是同一个人，会让"同一访客不能
    并发检索"这条限制误伤成"全站同一时刻只能有一个人在搜"。
    反向代理转发时通常会把真实来源写进 X-Forwarded-For（可能有多级，取第一个，
    即离真实访客最近的那个）；没有这个头时说明是直连（比如本地不经隧道直接
    访问），退回 request.client.host。
    这个值是访客自己可控的请求头，伪造后最多是绕开这条限速优化，不涉及权限
    或数据安全，可以接受。
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def fast_cache_key(q, papers, min_papers, time_range):
    # 时间范围写进 key；缓存跨自然年时 resolve_time_range 的起止年会变，
    # 所以连同具体年份一起放进去，避免跨年后命中旧口径
    range_sig = f"{time_range['key']}:{time_range['start']}-{time_range['end']}"
    return f"{q}|{papers}|{min_papers}|{range_sig}"


def range_alternatives(raw, q, papers, min_papers, time_range):
    """
    当前时间范围没有结果时，用已缓存的原始数据（不再请求 dblp / OpenAlex）试算更宽的范围，
    返回有结果的那些档位，供前端给出「切到近 5 年 / 切到全部」。
    算出来的结果顺手放进结果缓存，用户点击后直接命中。
    """
    out = []
    for key in topic_graph.wider_time_range_keys(time_range["key"]):
        tr = topic_graph.resolve_time_range(key)
        alt_key = fast_cache_key(q, papers, min_papers, tr)
        res = _RESULT_CACHE.get(alt_key)
        if res is None:
            res = topic_graph.build_topic_payload(raw, min_papers, tr)
            if res.get("error"):
                continue
            _RESULT_CACHE.put(alt_key, res)
        out.append({
            "key": tr["key"],
            "value": str(tr["years"]) if tr["years"] else "all",   # 对应前端下拉框的 value
            "label": tr["label"],
            "scholars": res["stats"]["scholars"],
        })
    return out


def strip_internal(payload):
    """去掉内部字段（networkx 图对象无法 JSON 序列化）。"""
    return {k: v for k, v in payload.items() if not k.startswith("_")}


@app.get("/api/search")
async def search(request: Request, q: str, papers: int = 300, min_papers: int = 2,
                 deep: bool = False, seeds: int = 10, years: str = "all"):
    """
    SSE 流式接口：边抓边推进度，最后推完整结果。

    事件类型：
      progress      -> {fetched, total, message}
      done          -> 快速结果的图数据（nodes/edges/legend/stats）
      deep_progress -> {done, total, message}   深度模式的滚雪球进度
      deep_done     -> 滚雪球扩展后的图数据，前端用它替换快速结果
      error         -> {message}

    深度模式为什么分两段推：快速结果十几秒就能出来，滚雪球还要每个种子一次请求
    （6 秒间隔，15 个种子约 90 秒）。先把能看的图给用户，再在后台把连通性补上，
    比让人对着转圈等两分钟好得多。

    years: all / 5 / 3（也接受 5y / 3y）。按论文发表年份过滤、含当年，
    口径见 topic_graph.resolve_time_range；结果 stats.time_range 里带着范围描述。
    """
    q = (q or "").strip()
    # 公开部署时把上限收紧一些：papers=10000 (dblp 硬上限) 单次要抓 7 分钟以上，
    # seeds=40 的深度模式要跑将近 4 分钟，一个访客就能把全局节流闸占满，
    # 让其他人排很久的队。命令行版（talent_map_by_topic.py）不走这个接口，不受影响。
    papers = max(100, min(papers, PUBLIC_MAX_PAPERS))
    min_papers = max(1, min(min_papers, 20))
    seeds = max(3, min(seeds, PUBLIC_MAX_SEEDS))
    time_range = topic_graph.resolve_time_range(years)
    client_ip = get_client_ip(request)

    async def event_stream():
        if not q:
            yield sse("error", {"message": "请输入方向关键词"})
            return

        # 同一访客不能同时开好几个检索：dblp 请求是全局串行的（见 topic_graph.dblp_get），
        # 一个人开几个标签页会把队列占满，让其他访客等更久。
        with _INFLIGHT_LOCK:
            if client_ip in _INFLIGHT_IPS:
                # error_kind=busy：前端据此留在上次结果页，把提示放进提示区（不是「没有结果」）
                yield sse("error", {"message": "你有一个检索还在进行中，请等它结束后再发起新的搜索",
                                    "error_kind": "busy"})
                return
            _INFLIGHT_IPS.add(client_ip)

        try:
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def make_reporter(event_name):
                def report(done_n, total_n, message):
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        (event_name, {"fetched": done_n, "done": done_n,
                                      "total": total_n, "message": message}),
                    )
                return report

            # 抓取是同步阻塞的（requests + time.sleep），必须放到线程里跑，
            # 否则会卡住整个事件循环，连已经产生的进度都推不出去。
            def worker():
                try:
                    fast_key = fast_cache_key(q, papers, min_papers, time_range)
                    fast = _RESULT_CACHE.get(fast_key)
                    if fast is None:
                        # 原始论文 + OpenAlex 补充与时间范围无关，单独缓存：
                        # 同一关键词切换范围时不再打 dblp / OpenAlex，只在本地重算
                        raw_key = f"raw|{q}|{papers}"
                        raw = _RAW_CACHE.get(raw_key)
                        if raw is None:
                            raw = topic_graph.fetch_topic_papers(
                                q, papers, make_reporter("progress"))
                            if raw.get("papers"):
                                _RAW_CACHE.put(raw_key, raw)
                        else:
                            make_reporter("progress")(0, 0, "已抓取过该关键词，按所选时间范围重新统计…")
                        fast = topic_graph.build_topic_payload(
                            raw, min_papers, time_range, make_reporter("progress"))
                        if not fast.get("error"):
                            _RESULT_CACHE.put(fast_key, fast)
                        elif fast.get("error_kind") == "empty_range" and raw.get("papers"):
                            fast["range_alternatives"] = range_alternatives(
                                raw, q, papers, min_papers, time_range)
                    else:
                        make_reporter("progress")(0, 0, "命中缓存，直接返回")

                    loop.call_soon_threadsafe(queue.put_nowait, ("fast_result", fast))
                    if fast.get("error") or not deep:
                        loop.call_soon_threadsafe(queue.put_nowait, ("finish", None))
                        return

                    deep_key = f"{fast_key}|deep{seeds}"
                    deep_res = _DEEP_CACHE.get(deep_key)
                    if deep_res is None:
                        deep_res = topic_graph.deep_expand(
                            fast, seeds, on_progress=make_reporter("deep_progress"))
                        if not deep_res.get("error"):
                            _DEEP_CACHE.put(deep_key, deep_res)
                    loop.call_soon_threadsafe(queue.put_nowait, ("deep_result", deep_res))
                    loop.call_soon_threadsafe(queue.put_nowait, ("finish", None))
                except Exception as exc:
                    # 意外异常始终打出来，别静默吞掉——跟下面 dblp 请求失败的
                    # 预期内重试不是一回事，这里是代码本身出了没预料到的问题。
                    import traceback
                    traceback.print_exc()
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("fast_result", {"error": f"检索出错：{exc}"}))
                    loop.call_soon_threadsafe(queue.put_nowait, ("finish", None))

            threading.Thread(target=worker, daemon=True).start()

            while True:
                kind, data = await queue.get()
                if kind in ("progress", "deep_progress"):
                    yield sse(kind, data)
                elif kind == "fast_result":
                    if data.get("error"):
                        err = {"message": data["error"]}
                        if data.get("range_alternatives"):
                            err["range_alternatives"] = data["range_alternatives"]
                        yield sse("error", err)
                    else:
                        yield sse("done", strip_internal(data))
                elif kind == "deep_result":
                    if data.get("error"):
                        yield sse("deep_error", {"message": data["error"]})
                    else:
                        yield sse("deep_done", strip_internal(data))
                else:
                    return
        finally:
            # 无论正常结束、报错还是访客中途关掉页面（StreamingResponse 会把
            # GeneratorExit 抛进这个生成器），都要把这个 IP 从"占用中"里摘掉，
            # 否则一次异常断开就会把这个人永久卡在"检索进行中"的错误提示里。
            with _INFLIGHT_LOCK:
                _INFLIGHT_IPS.discard(client_ip)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # no-transform：明确告诉中间代理/CDN 不要重新压缩或缓冲这个响应体。
        # Render 部署时前面套了 Cloudflare，如果它对 text/event-stream 做了
        # 压缩缓冲，进度事件会被攒起来一次性吐出，SSE 就失去了实时推送的意义
        # （极端情况下甚至会在客户端超时之前完全收不到任何字节）。
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/presets")
async def presets():
    return {
        "topics": topic_graph.PRESET_TOPICS,
        # 时间范围选项的文案（含具体起止年份）由后端给出，与结果里的 stats.time_range 同源
        "time_ranges": [topic_graph.resolve_time_range(k) for k in ("all", "5y", "3y")],
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>论文方向人才地图</title>
<script src="/lib/vis-9.1.2/vis-network.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #1e1e2e; color: #cdd6f4;
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif;
    height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  }
  header { background: #181825; padding: 14px 22px; box-shadow: 0 2px 12px rgba(0,0,0,.45); z-index: 10; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  h1 { margin: 0 16px 0 0; font-size: 17px; font-weight: 600; white-space: nowrap; }
  h1 span { color: #89b4fa; }
  #topicInput {
    flex: 1; min-width: 240px; padding: 10px 14px; font-size: 15px;
    background: #313244; border: 1px solid #45475a; border-radius: 6px; color: #cdd6f4;
  }
  #topicInput:focus { outline: none; border-color: #89b4fa; }
  button {
    padding: 10px 20px; font-size: 14px; background: #89b4fa; color: #1e1e2e;
    border: none; border-radius: 6px; cursor: pointer; font-weight: 600; white-space: nowrap;
  }
  button:hover:not(:disabled) { background: #a6c8ff; }
  button:disabled { background: #45475a; color: #7f849c; cursor: not-allowed; }
  .btnSecondary {
    padding: 6px 12px; font-size: 12px; background: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; cursor: pointer; font-weight: 500;
  }
  .btnSecondary:hover { background: #45475a; }
  .opt { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #a6adc8; white-space: nowrap; }
  .opt input { width: 74px; padding: 6px 8px; background: #313244; border: 1px solid #45475a;
               border-radius: 4px; color: #cdd6f4; font-size: 12px; }
  .deepOpt { cursor: pointer; user-select: none; }
  .deepOpt input { width: auto; cursor: pointer; }
  .opt select { padding: 6px 8px; background: #313244; border: 1px solid #45475a;
                border-radius: 4px; color: #cdd6f4; font-size: 12px; cursor: pointer; }
  .tabs { display: flex; gap: 6px; margin-bottom: 9px; }
  .tab { flex: 1; padding: 5px 8px; font-size: 12px; text-align: center; cursor: pointer;
         background: #313244; border: 1px solid #45475a; border-radius: 4px; color: #bac2de; }
  .tab.active { background: #45475a; color: #cdd6f4; font-weight: 600; }
  .viewToggle {
    display: none; gap: 0; margin-left: 8px; border: 1px solid #45475a; border-radius: 6px; overflow: hidden;
  }
  .viewToggle button {
    padding: 7px 14px; font-size: 12px; background: #313244; color: #bac2de;
    border: none; border-radius: 0; font-weight: 500;
  }
  .viewToggle button.active { background: #45475a; color: #cdd6f4; font-weight: 600; }
  .instItem { display: flex; align-items: center; padding: 5px 6px; border-radius: 4px;
              cursor: pointer; font-size: 12px; }
  .instItem:hover { background: #313244; }
  .instItem.active { background: #45475a; }
  /* 统一提示区：页面上所有提示条（数据源/补充降级、深度进行中/失败、限流、连接中断）
     都是 #noticeArea 里的一行，放在文档流里（范围条下方、结构图与短名单上方），
     多条按固定顺序往下堆叠，<main> 相应变矮——任何组合都不会浮在工具栏、标签页或查找框上。
     不要给 .notice 加 position:absolute/fixed。 */
  #noticeArea { position: static; }
  #noticeArea .notice {
    display: none; position: static; padding: 6px 22px; font-size: 12px; line-height: 1.5;
    border-bottom: 1px solid; word-break: break-word;
  }
  .notice.warn { background: rgba(249,226,175,.10); border-color: rgba(249,226,175,.45); color: #f9e2af; }
  .notice.info { background: rgba(137,180,250,.10); border-color: rgba(137,180,250,.45); color: #cdd6f4; }
  .notice.error { background: rgba(243,139,168,.10); border-color: rgba(243,139,168,.50); color: #f38ba8; }
  #noticeArea .noticeClose {
    float: right; margin: 0 0 0 12px; padding: 0 6px; background: none; border: none;
    color: inherit; font-size: 15px; line-height: 18px; font-weight: 400; opacity: .75;
  }
  #noticeArea .noticeClose:hover { opacity: 1; }
  #noticeArea .noticeClose::before { content: "×"; }
  #noticeArea .noticeClose[hidden] { display: none; }
  #presets { margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap; }
  .chip {
    padding: 4px 11px; font-size: 12px; background: #313244; border: 1px solid #45475a;
    border-radius: 20px; cursor: pointer; color: #bac2de;
  }
  .chip:hover { background: #45475a; color: #cdd6f4; }
  .chip.suggest { border-color: #89b4fa; color: #89b4fa; }
  #scopeBar {
    display: none; padding: 7px 22px; font-size: 12.5px; line-height: 1.5;
    background: rgba(137,180,250,.10); border-bottom: 1px solid #45475a; color: #bac2de;
  }
  #scopeBar b { color: #89b4fa; font-weight: 600; }
  #scopeBar .scopeNote { color: #7f849c; }
  main { flex: 1; min-height: 0; position: relative; }
  #graph { position: absolute; inset: 0; }
  #listView {
    position: absolute; inset: 0; display: none; flex-direction: column;
    background: #1e1e2e; z-index: 4; padding: 14px 18px 18px;
  }
  #listView.visible { display: flex; }
  .listToolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
  .listToolbar .opt input[type="text"] { width: 140px; }
  .listToolbar .opt input[type="number"] { width: 64px; }
  .listToolbar label.chk { display: flex; align-items: center; gap: 5px; font-size: 12px; color: #a6adc8; cursor: pointer; }
  .listMeta { font-size: 12px; color: #a6adc8; margin-left: auto; }
  .listDisclaimer {
    font-size: 11px; color: #7f849c; margin-bottom: 8px; line-height: 1.5;
  }
  #listTableWrap { flex: 1; min-height: 0; overflow: auto; border: 1px solid #313244; border-radius: 6px; }
  table.shortlist { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.shortlist th, table.shortlist td {
    padding: 8px 10px; text-align: left; border-bottom: 1px solid #313244; vertical-align: top;
  }
  table.shortlist th {
    position: sticky; top: 0; background: #181825; color: #a6adc8; font-weight: 600; z-index: 1;
  }
  table.shortlist tr:hover td { background: #262637; }
  table.shortlist a { color: #89b4fa; text-decoration: none; }
  table.shortlist a:hover { text-decoration: underline; }
  .gapTag {
    display: inline-block; padding: 1px 6px; margin: 1px 3px 1px 0; border-radius: 3px;
    background: #313244; color: #f9e2af; font-size: 11px;
  }
  .roleCore { color: #a6e3a1; }
  td.citeGap { color: #f9e2af; cursor: help; }
  .roleCollab { color: #cba6f7; }
  .panel {
    position: absolute; background: rgba(30,30,46,.96); padding: 14px;
    border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,.4); z-index: 5;
  }
  #legendPanel { top: 16px; left: 16px; width: 250px; display: none; }
  #searchPanel  { top: 16px; right: 16px; width: 260px; display: none; }
  #detailCard {
    position: absolute; top: 16px; right: 16px; width: 340px; max-height: calc(100% - 32px);
    overflow-y: auto; display: none; z-index: 8;
    background: rgba(24,24,37,.98); padding: 16px; border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0,0,0,.5); border: 1px solid #45475a;
  }
  #detailCard h3 { margin: 0 0 4px; font-size: 16px; font-weight: 600; color: #cdd6f4; }
  #detailCard .roleLine { font-size: 12px; color: #a6adc8; margin-bottom: 12px; }
  #detailCard .field { margin-bottom: 10px; font-size: 12.5px; line-height: 1.55; }
  #detailCard .field .k { color: #7f849c; font-size: 11px; margin-bottom: 2px; }
  #detailCard .field .v { color: #cdd6f4; word-break: break-word; }
  #detailCard .actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
  #detailCard .closeX {
    position: absolute; top: 10px; right: 12px; background: transparent; border: none;
    color: #a6adc8; font-size: 18px; cursor: pointer; padding: 2px 8px; font-weight: 400;
  }
  #detailCard .closeX:hover { color: #cdd6f4; }
  .copyBtn {
    display: inline-block; margin-left: 6px; padding: 1px 6px; font-size: 10px;
    background: #313244; border: 1px solid #45475a; border-radius: 3px; cursor: pointer; color: #a6adc8;
  }
  .copyBtn:hover { background: #45475a; color: #cdd6f4; }
  .panel h3 { margin: 0 0 9px; font-size: 13px; font-weight: 600; }
  #resetBtn {
    display: inline-block; margin-bottom: 9px; padding: 4px 11px; font-size: 12px;
    background: #313244; border: 1px solid #45475a; border-radius: 4px; cursor: pointer;
  }
  #resetBtn:hover { background: #45475a; }
  #legendList { max-height: 46vh; overflow-y: auto; }
  .legendItem { display: flex; align-items: center; padding: 5px 6px; border-radius: 4px;
                cursor: pointer; font-size: 12px; }
  .legendItem:hover { background: #313244; }
  .legendItem.active { background: #45475a; }
  .dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
  .lbl { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cnt { color: #a6adc8; margin-left: 6px; flex-shrink: 0; }
  #nodeSearch { width: 100%; padding: 8px 10px; background: #313244; border: 1px solid #45475a;
                border-radius: 5px; color: #cdd6f4; font-size: 13px; }
  #nodeSearch:focus { outline: none; border-color: #89b4fa; }
  #nodeResults { margin-top: 8px; max-height: 40vh; overflow-y: auto; }
  .hit { padding: 7px 10px; border-bottom: 1px solid #313244; cursor: pointer; font-size: 12px; }
  .hit:hover { background: #313244; }
  .hit .n { color: #89b4fa; font-weight: 600; }
  .hit .m { color: #a6adc8; font-size: 11px; margin-top: 2px; }
  .empty { padding: 10px; text-align: center; color: #a6adc8; font-size: 12px; }
  #overlay {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    background: rgba(30,30,46,.93); z-index: 20; text-align: center; padding: 24px;
  }
  #overlay.hidden { display: none; }
  .box { max-width: 560px; }
  .spinner {
    width: 38px; height: 38px; margin: 0 auto 18px; border: 3px solid #45475a;
    border-top-color: #89b4fa; border-radius: 50%; animation: spin 0.9s linear infinite;
  }
  .spinner.hidden { display: none; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #msg { font-size: 15px; margin-bottom: 8px; }
  #sub { font-size: 12.5px; color: #a6adc8; line-height: 1.65; }
  #relaxBox { margin-top: 14px; display: none; text-align: left; }
  #relaxBox h4 { margin: 0 0 8px; font-size: 13px; color: #cdd6f4; font-weight: 600; }
  #relaxChips { display: flex; flex-wrap: wrap; gap: 6px; }
  #rangeAltBox, #restoreBox { margin-top: 14px; display: none; text-align: left; }
  #rangeAltBox h4 { margin: 0 0 8px; font-size: 13px; color: #cdd6f4; font-weight: 600; }
  #rangeAltChips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip.rangeAlt { border-color: #a6e3a1; color: #a6e3a1; }
  .bar { width: 300px; height: 4px; background: #313244; border-radius: 2px; margin: 14px auto 0; overflow: hidden; }
  .bar div { height: 100%; background: #89b4fa; width: 0; transition: width .3s; }
  .bar.hidden { display: none; }
  #stats {
    position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
    background: rgba(24,24,37,.94); padding: 7px 18px; border-radius: 18px;
    font-size: 12px; color: #a6adc8; z-index: 5; display: none; white-space: nowrap;
    max-width: 92vw; overflow: hidden; text-overflow: ellipsis;
  }
</style>
</head>
<body>

<header>
  <div class="row">
    <h1>论文方向<span>人才地图</span></h1>
    <input id="topicInput" placeholder="输入论文方向关键词，如 BEV perception、occupancy prediction" autocomplete="off" />
    <div class="opt">论文数 <input id="papersInput" type="number" value="300" min="100" max="10000" step="100" /></div>
    <div class="opt">最少发文 <input id="minInput" type="number" value="2" min="1" max="20" /></div>
    <div class="opt" title="按论文发表年份过滤（含当年）。结构图、短名单、CSV 的发文数、方向核心判定、合作关系、引用与排序都只统计该范围内的论文。已检索过的关键词切换范围不会重新抓取 dblp。">时间范围
      <select id="rangeSelect">
        <option value="all">全部</option>
        <option value="5">近 5 年</option>
        <option value="3">近 3 年</option>
      </select>
    </div>
    <div class="opt">节点大小
      <select id="sizeMode">
        <option value="papers">按发文量</option>
        <option value="citations">按引用量</option>
      </select>
    </div>
    <label class="opt deepOpt" title="可选：先出快速结果，再用该方向 top 学者做滚雪球扩展。默认关闭。每位种子约 14 秒，10 位约 2~3 分钟。深度模式会混入非该方向的合作者，图上会标注「经关联引入的合作者」">
      <input type="checkbox" id="deepInput" /> 深度模式
    </label>
    <button id="goBtn">检索</button>
    <div class="viewToggle" id="viewToggle">
      <button type="button" id="viewGraphBtn" class="active">结构图</button>
      <button type="button" id="viewListBtn">短名单</button>
    </div>
  </div>
  <div id="presets"></div>
</header>
<div id="scopeBar"></div>
<div id="noticeArea" aria-live="polite">
  <!-- 顺序即堆叠顺序：数据源/补充降级 → 深度进行中/失败 → 限流 → 连接中断 -->
    <div class="notice warn" id="degradeBanner" role="status"><button type="button" class="noticeClose" aria-label="关闭提示" title="关闭这条提示（不影响数据；新的检索会重新显示）" hidden></button><span class="noticeText"></span></div>
    <div class="notice info" id="deepBanner" role="status"><button type="button" class="noticeClose" aria-label="关闭提示" title="关闭这条提示（不影响数据；新的检索会重新显示）" hidden></button><span class="noticeText"></span></div>
    <div class="notice warn" id="rateBanner" role="status"><button type="button" class="noticeClose" aria-label="关闭提示" title="关闭这条提示（不影响数据；新的检索会重新显示）" hidden></button><span class="noticeText"></span></div>
    <div class="notice error" id="connBanner" role="status"><button type="button" class="noticeClose" aria-label="关闭提示" title="关闭这条提示（不影响数据；新的检索会重新显示）" hidden></button><span class="noticeText"></span></div>
</div>

<main>
  <div id="graph"></div>
  <div id="listView">
    <div class="listDisclaimer">
      短名单仅供学术结构探索与试用筛选。论文署名机构反映发表当时的署名单位，不等于现职担保。
      Google Scholar 链接为作者搜索页（非精确主页）。引用/机构可能因 OpenAlex 配额或元数据缺口未补全。
    </div>
    <div class="listToolbar">
      <div class="opt">姓名 <input id="listNameFilter" type="text" placeholder="关键字…" /></div>
      <div class="opt">最少发文 <input id="listMinPapers" type="number" value="0" min="0" max="99" /></div>
      <div class="opt">聚类
        <select id="listClusterFilter"><option value="">全部</option></select>
      </div>
      <label class="chk"><input type="checkbox" id="listCoreOnly" /> 仅方向核心学者</label>
      <label class="chk"><input type="checkbox" id="listHasInst" /> 仅有署名机构</label>
      <button type="button" class="btnSecondary" id="exportCsvBtn">导出 CSV</button>
      <span class="listMeta" id="listMeta"></span>
    </div>
    <div id="listTableWrap">
      <table class="shortlist">
        <thead>
          <tr>
            <th>姓名</th>
            <th>论文署名机构（发表当时）</th>
            <th id="thPapers">方向发文</th>
            <th id="thCites">方向相关引用</th>
            <th>角色</th>
            <th>团体/聚类</th>
            <th>Scholar 搜索</th>
            <th>数据缺口</th>
          </tr>
        </thead>
        <tbody id="listBody"></tbody>
      </table>
    </div>
  </div>
  <div class="panel" id="legendPanel">
    <div class="tabs">
      <div class="tab active" id="tabTeam">研究团体</div>
      <div class="tab" id="tabInst">署名机构（历史）</div>
    </div>
    <div id="resetBtn">重置视图</div>
    <div id="legendList"></div>
    <div id="instList" style="display:none; max-height:46vh; overflow-y:auto;"></div>
  </div>
  <div class="panel" id="searchPanel">
    <h3>在结果中查找学者</h3>
    <input id="nodeSearch" placeholder="输入姓名..." autocomplete="off" />
    <div id="nodeResults"></div>
  </div>
  <div id="detailCard">
    <button type="button" class="closeX" id="detailClose" title="关闭">&times;</button>
    <div id="detailBody"></div>
  </div>
  <div id="stats"></div>
  <div id="overlay">
    <div class="box">
      <div class="spinner hidden" id="spinner"></div>
      <div id="msg">输入一个论文方向，开始检索</div>
      <div id="sub">
        主数据源为 dblp；若被反爬拦截会自动改用 OpenAlex / Crossref，并在界面标明。<br />
        关键词建议只用 1~2 个核心词 —— dblp 多个词之间是 AND 关系，词越多结果越少
        （"BEV" 有 4979 篇，"BEV perception autonomous driving" 只剩 19 篇）。<br />
        dblp 要求每次请求间隔 4 秒，检索 300 篇约需 10 秒。<br />
        默认快速检索；深度模式为可选项（默认关闭），会引入非该方向的合作者。
      </div>
      <div id="rangeAltBox">
        <h4>切换时间范围（用已抓取的数据重新统计，不重新检索）</h4>
        <div id="rangeAltChips"></div>
      </div>
      <div id="restoreBox">
        <button type="button" class="btnSecondary" id="restoreLastBtn">返回上次结果</button>
      </div>
      <div id="relaxBox">
        <h4>可点选放宽关键词后重搜</h4>
        <div id="relaxChips"></div>
      </div>
      <div class="bar hidden" id="bar"><div id="barFill"></div></div>
    </div>
  </div>
</main>

<script>
var network = null, nodesDS = null, edgesDS = null;
var colorById = {}, sizeById = {}, labelById = {}, allNodes = [];
var modified = [], activeCid = null, currentES = null, deepStart = null;
var activeInst = null, lastStats = null, currentView = 'graph', lastQuery = '';
var searchSeq = 0;   // 每次 runSearch 自增；旧检索的延时回调据此失效，不会盖掉新的提示
var DIM = '#3a3a4a';

var $ = function (id) { return document.getElementById(id); };

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function roleLabel(n) {
  return n.is_seed === false ? '经关联引入的合作者' : '方向核心';
}

function scholarSearchUrl(n) {
  var q = '"' + n.label + '"' + (n.institution ? ' "' + n.institution + '"' : '');
  return 'https://scholar.google.com/citations?view_op=search_authors&mauthors='
    + encodeURIComponent(q);
}

// 引用三态：ok（全部论文取到引用数据，0 就是真实的 0）/ partial（部分取到，数值为下限）/
// missing（一篇都没取到：没 DOI、OpenAlex 未命中或补充降级）。界面、详情卡、短名单、复制摘要、
// CSV 都走下面这组函数，保证口径一致。
function citeStatus(n) {
  if (n.citations === null || n.citations === undefined) return 'missing';
  return n.citation_status === 'partial' ? 'partial' : 'ok';
}

function citePapers(n) {
  var cp = n.citation_papers;
  return (cp && cp.length === 2) ? (cp[0] + '/' + cp[1]) : '';
}

// 界面上显示的值：真实 0 显示 0；缺失显示「未补全」；部分缺失显示「≥N」
function citationText(n) {
  var st = citeStatus(n);
  if (st === 'missing') return '未补全';
  if (st === 'partial') return '≥' + n.citations;
  return String(n.citations);
}

function citationNote(n) {
  var st = citeStatus(n);
  if (st === 'missing') return '未取到引用数据（论文缺 DOI、OpenAlex 未命中、补充降级或备用源未提供被引数），不代表 0 次';
  if (st === 'partial') return '部分缺失：仅 ' + citePapers(n) + ' 篇论文取到引用数据，数值为已取到部分之和（下限）';
  return '';
}

// CSV 里的值：缺失留空（原因写在缺口标记列），真实 0 写 0，部分缺失写已取到部分之和（下限）；
// 部分缺失且已取到部分为 0 时也留空——下限 0 没有信息量，写 0 会被误读成真实被引 0 次
function citationCsv(n) {
  var st = citeStatus(n);
  if (st === 'missing') return '';
  if (st === 'partial' && !n.citations) return '';
  return n.citations;
}

// 短名单 / CSV 排序：方向发文降序；同发文时有引用值的在前（按值降序），缺失统一排在后面
function compareShortlist(a, b) {
  if (b.papers !== a.papers) return b.papers - a.papers;
  var am = citeStatus(a) === 'missing', bm = citeStatus(b) === 'missing';
  if (am !== bm) return am ? 1 : -1;
  if (am) return 0;
  return b.citations - a.citations;
}

function gapMarkers(n) {
  var gaps = [];
  var cst = citeStatus(n);
  if (cst === 'missing') gaps.push('缺引用');
  else if (cst === 'partial') gaps.push('引用部分缺失(' + citePapers(n) + '篇有数据)，数值为下限');
  if (!n.institution) gaps.push('缺署名机构');
  if (n.is_seed === false) gaps.push('非方向发文引入');
  return gaps;
}

function currentRange() {
  // 以「当前展示的结果」的口径为准（而不是下拉框的值），保证界面与 CSV 一致
  var tr = (lastStats && lastStats.time_range) || null;
  return tr || { key: 'all', years: null, label: '全部年份', basis: '按论文发表年份，含当年' };
}

function rangeShort() {
  var tr = currentRange();
  return tr.years ? ('近 ' + tr.years + ' 年') : '全部年份';
}

function rangeFullText() {
  var tr = currentRange();
  return tr.years ? (tr.label + '，' + tr.basis) : '全部年份（不按年份过滤）';
}

function renderScopeBar() {
  var s = lastStats || {};
  var tr = currentRange();
  var html = '当前时间范围：<b>' + esc(tr.label) + '</b>';
  if (tr.years) {
    html += ' · ' + esc(tr.basis) + ' · 范围内论文 ' + esc(s.papers_in_range) + ' / 已抓取 '
      + esc(s.papers_fetched) + ' 篇';
  } else {
    html += ' · 已抓取论文 ' + esc(s.papers_fetched) + ' 篇';
    if (s.papers_year_min && s.papers_year_max) {
      html += '（年份 ' + esc(s.papers_year_min) + '–' + esc(s.papers_year_max) + '）';
    }
  }
  html += ' <span class="scopeNote">· 结构图、短名单与 CSV 的发文数、方向核心判定、合作关系、聚类、引用与排序均只统计该范围内论文'
    + (tr.years ? '；引用 = 范围内论文至今的累计被引' : '') + '</span>';
  $('scopeBar').innerHTML = html;
  $('scopeBar').style.display = 'block';
  var suffix = tr.years ? '（' + rangeShort() + '）' : '';
  $('thPapers').textContent = '方向发文' + suffix;
  $('thCites').textContent = '方向相关引用' + suffix;
}


fetch('/api/presets').then(function (r) { return r.json(); }).then(function (d) {
  (d.time_ranges || []).forEach(function (tr) {
    var v = tr.years ? String(tr.years) : 'all';
    var opt = $('rangeSelect').querySelector('option[value="' + v + '"]');
    if (opt) opt.textContent = tr.years ? tr.label : '全部';
  });
  var box = $('presets');
  d.topics.forEach(function (t) {
    var c = document.createElement('span');
    c.className = 'chip'; c.textContent = t;
    c.onclick = function () { $('topicInput').value = t; runSearch(); };
    box.appendChild(c);
  });
});

// ---- 统一提示区 API：所有提示条只通过这里显示/隐藏 ----
// level: info（进行中）/ warn（降级、限流）/ error（失败、中断）；dismissible: 显示关闭按钮。
// 关闭只是隐藏这一条（同样的文字本次结果内不再弹出），不影响数据，新检索会清空重来。
var NOTICE_IDS = ['degradeBanner', 'deepBanner', 'rateBanner', 'connBanner'];
var noticeDismissed = {};
function setNotice(id, text, opts) {
  opts = opts || {};
  var el = $(id);
  if (!el) return;
  if (!text) { clearNotice(id); return; }
  if (opts.dismissible && noticeDismissed[id] === text) return;
  el.className = 'notice ' + (opts.level || 'warn');
  el.querySelector('.noticeText').textContent = text;
  el.querySelector('.noticeClose').hidden = !opts.dismissible;
  el.style.display = 'block';
}
function clearNotice(id) {
  var el = $(id);
  if (el) el.style.display = 'none';
}
function clearAllNotices() {
  NOTICE_IDS.forEach(clearNotice);
  noticeDismissed = {};
}
NOTICE_IDS.forEach(function (id) {
  var el = $(id);
  el.querySelector('.noticeClose').onclick = function () {
    noticeDismissed[id] = el.querySelector('.noticeText').textContent;
    clearNotice(id);
  };
});

function setOverlay(show, opts) {
  opts = opts || {};
  $('overlay').classList.toggle('hidden', !show);
  if (!show || opts.busy) {
    $('rangeAltBox').style.display = 'none';
    $('restoreBox').style.display = 'none';
  }
  if (!show) { $('relaxBox').style.display = 'none'; return; }
  $('msg').textContent = opts.msg || '';
  $('sub').innerHTML = opts.sub || '';
  $('spinner').classList.toggle('hidden', !opts.busy);
  $('bar').classList.toggle('hidden', !opts.progress);
  if (opts.progress) $('barFill').style.width = opts.percent + '%';
  if (opts.relaxQuery) showRelaxSuggestions(opts.relaxQuery, opts.message || opts.sub || '');
  else if (!opts.busy) { /* keep existing relax if any */ }
  else $('relaxBox').style.display = 'none';
}

function suggestQueries(q) {
  var parts = q.trim().split(/\\s+/).filter(Boolean);
  var out = [], seen = {};
  function add(s, note) {
    if (!s || s === q || seen[s]) return;
    seen[s] = true;
    out.push({ q: s, note: note });
  }
  if (parts.length >= 2) {
    add(parts[0], '只保留首词');
    add(parts.slice(0, -1).join(' '), '去掉末词');
    if (parts.length >= 3) add(parts.slice(0, 2).join(' '), '只保留前两词');
  }
  return out;
}

function showRelaxSuggestions(q, msgText) {
  var suggestions = suggestQueries(q);
  var looksStrict = /AND|词越多|没有检索到|没有人在该方向|调低|调高|结果/.test(msgText || '')
    || suggestions.length > 0;
  if (!looksStrict || !suggestions.length) {
    $('relaxBox').style.display = 'none';
    return;
  }
  var box = $('relaxChips');
  box.innerHTML = '';
  suggestions.forEach(function (s) {
    var c = document.createElement('span');
    c.className = 'chip suggest';
    c.textContent = s.q + '（' + s.note + '）';
    c.title = '填入并重新检索';
    c.onclick = function () {
      $('topicInput').value = s.q;
      runSearch();
    };
    box.appendChild(c);
  });
  $('relaxBox').style.display = 'block';
}

// 某个时间范围没有结果时，后端用已缓存的原始数据试算更宽的档位，有结果的才给出来
function showRangeAlternatives(q, alts) {
  var box = $('rangeAltChips');
  box.innerHTML = '';
  if (!alts || !alts.length) { $('rangeAltBox').style.display = 'none'; return; }
  alts.forEach(function (a) {
    var c = document.createElement('span');
    c.className = 'chip rangeAlt';
    c.setAttribute('data-range', a.value);
    c.textContent = (a.key === 'all' ? '切到全部' : '切到' + a.label) + ' · ' + a.scholars + ' 人';
    c.title = '下拉框同步切换，并用已抓取的论文重新统计（不重新请求 dblp）';
    c.onclick = function () {
      $('rangeSelect').value = a.value;
      $('topicInput').value = q;
      runSearch();
    };
    box.appendChild(c);
  });
  $('rangeAltBox').style.display = 'block';
}

// 报错后仍保留上一次结果的入口（结构图 / 短名单 / 导出）
function showRestoreEntry() {
  if (!lastStats || !allNodes.length) { $('restoreBox').style.display = 'none'; return; }
  $('restoreLastBtn').textContent = '返回上次结果（' + (lastStats.query || '') + ' · '
    + currentRange().label + ' · ' + allNodes.length + ' 人）';
  $('restoreBox').style.display = 'block';
  $('viewToggle').style.display = 'flex';
}

function showResultChrome() {
  renderScopeBar();
  renderStatsBar();
  $('viewToggle').style.display = 'flex';
  setView(currentView);
}

function restoreLastResult() {
  if (!lastStats || !allNodes.length) return;
  // 头部控件回到「正在展示的结果」的口径，避免下拉框和图/短名单/CSV 不一致
  var tr = currentRange();
  $('rangeSelect').value = tr.years ? String(tr.years) : 'all';
  if (lastStats.query) { $('topicInput').value = lastStats.query; lastQuery = lastStats.query; }
  setOverlay(false);
  showResultChrome();
}

function runSearch() {
  var q = $('topicInput').value.trim();
  if (!q) { $('topicInput').focus(); return; }
  if (currentES) currentES.close();
  searchSeq += 1;
  lastQuery = q;
  closeDetail();
  setView('graph');

  $('goBtn').disabled = true;
  $('legendPanel').style.display = 'none';
  $('searchPanel').style.display = 'none';
  $('stats').style.display = 'none';
  $('viewToggle').style.display = 'none';
  clearAllNotices();
  $('scopeBar').style.display = 'none';
  $('listView').classList.remove('visible');
  setOverlay(true, { busy: true, msg: '正在检索「' + q + '」...',
                     sub: '正在联网查询论文库（dblp，必要时自动备用源）', progress: true, percent: 0 });

  var deep = $('deepInput').checked;
  var fastShown = false;   // 本次检索的快速结果是否已经展示（决定报错走提示区还是覆盖层）
  var url = '/api/search?q=' + encodeURIComponent(q)
          + '&papers=' + encodeURIComponent($('papersInput').value)
          + '&min_papers=' + encodeURIComponent($('minInput').value)
          + '&deep=' + (deep ? 'true' : 'false')
          + '&years=' + encodeURIComponent($('rangeSelect').value);
  var es = new EventSource(url);
  currentES = es;

  es.addEventListener('progress', function (e) {
    var d = JSON.parse(e.data);
    var pct = d.total ? Math.min(100, Math.round(d.fetched / d.total * 100)) : 0;
    setOverlay(true, { busy: true, msg: '正在检索「' + q + '」...',
                       sub: d.message, progress: true, percent: pct });
  });

  es.addEventListener('done', function (e) {
    render(JSON.parse(e.data));
    fastShown = true;
    if (deep) {
      deepStart = Date.now();
      setNotice('deepBanner', '深度模式：正在扩展合作网络，完成后自动更新（可先浏览当前结果）', { level: 'info' });
    } else {
      es.close(); currentES = null; $('goBtn').disabled = false;
    }
  });

  es.addEventListener('deep_progress', function (e) {
    var d = JSON.parse(e.data);
    var extra = '';
    if (deepStart && d.done > 0 && d.total > d.done) {
      var perSeed = (Date.now() - deepStart) / d.done;
      var leftSec = Math.round(perSeed * (d.total - d.done) / 1000);
      extra = leftSec >= 60
        ? '，预计还需 ' + Math.ceil(leftSec / 60) + ' 分钟'
        : '，预计还需 ' + leftSec + ' 秒';
    }
    setNotice('deepBanner', '深度模式：' + d.message + extra, { level: 'info' });
  });

  es.addEventListener('deep_done', function (e) {
    es.close(); currentES = null; $('goBtn').disabled = false;
    render(JSON.parse(e.data));
    clearNotice('deepBanner');
  });

  es.addEventListener('deep_error', function (e) {
    es.close(); currentES = null; $('goBtn').disabled = false;
    var m = '扩展失败';
    try { m = JSON.parse(e.data).message; } catch (err) {}
    setNotice('deepBanner', '深度扩展失败：' + m + '（快速结果仍可用）', { level: 'error', dismissible: true });
  });

  es.addEventListener('error', function (e) {
    // 只处理服务端主动推送的 error 事件（带 data）；连接断开时浏览器派发的是不带 data 的
    // 普通 error 事件，交给下面的 es.onerror，否则会被误显示成「没有结果」并整页覆盖
    if (!e || typeof e.data !== 'string') return;
    es.close(); currentES = null; $('goBtn').disabled = false;
    var m = '检索失败', alts = [], kind = '';
    try {
      var d = JSON.parse(e.data);
      m = d.message || m;
      alts = d.range_alternatives || [];
      kind = d.error_kind || '';
    } catch (err) {}
    // 限流（同一访客已有检索在进行）不是「没有结果」：有上次结果就留在结果页，提示放进提示区
    if (kind === 'busy' && allNodes.length && lastStats) {
      restoreLastResult();
      setNotice('rateBanner', m + '（「' + q + '」未发起，当前仍显示上次结果）', { level: 'warn', dismissible: true });
      return;
    }
    setOverlay(true, { busy: false, msg: '没有结果', sub: esc(m),
                       relaxQuery: q, message: m });
    showRangeAlternatives(q, alts);
    showRestoreEntry();
  });

  es.onerror = function () {
    if (currentES !== es) return;
    es.close(); currentES = null; $('goBtn').disabled = false;
    // 快速结果已经在页面上（深度扩展阶段断开）：结果仍可用，提示放进提示区，不再整页覆盖
    if (fastShown) {
      clearNotice('deepBanner');
      setNotice('connBanner', '与本地服务的连接中断' + (deep ? '，深度扩展未完成' : '')
        + '（当前结果仍可用；确认 app.py 仍在运行后可重新检索）', { level: 'error', dismissible: true });
      return;
    }
    setOverlay(true, { busy: false, msg: '连接中断',
                       sub: '与本地服务的连接断开了，确认 app.py 仍在运行后重试。' });
    showRestoreEntry();
  };
}

function render(data) {
  if (typeof vis === 'undefined' || !vis.Network) {
    setOverlay(true, { busy: false, msg: '图形库未能加载',
      sub: 'vis-network 没加载成功，图无法绘制。<br>请确认 app.py 与 lib/ 目录在同一个文件夹下，然后刷新页面。' });
    return;
  }
  allNodes = data.nodes;
  lastStats = data.stats || {};
  colorById = {}; sizeById = {}; labelById = {};
  allNodes.forEach(function (n) {
    colorById[n.id] = n.color;
    labelById[n.id] = (n.label || '').toLowerCase();
  });

  computeSizes();
  nodesDS = new vis.DataSet(allNodes.map(function (n) {
    var samples = (n.samples || []).slice(0, 3).map(function (s) {
      return '· ' + s.year + ' ' + (s.title || '').substring(0, 60);
    }).join('<br>');
    var head = n.is_seed === false
      ? '<b>' + n.label + '</b><br><i>经关联引入的合作者（不一定做该方向）</i>'
      : '<b>' + n.label + '</b><br>方向核心 · 该方向发文: ' + n.papers + ' 篇';
    var extra = '';
    extra += '<br>该方向相关引用: ' + esc(citationText(n))
      + (citeStatus(n) === 'missing' ? '' : ' 次')
      + (citationNote(n) ? '<br><i>' + esc(citationNote(n)) + '</i>' : '');
    if (n.institution) extra += '<br>论文署名机构（发表当时）: ' + n.institution;
    else extra += '<br>论文署名机构: 未获取';
    if (n.topics && n.topics.length) extra += '<br>主题: ' + n.topics.join('、');
    return {
      id: n.id, label: n.label, color: n.color, size: sizeById[n.id], community: n.community,
      title: head + extra + '<br>dblp: ' + n.id + (samples ? '<br><br>代表作:<br>' + samples : '')
        + '<br><br><i>点击打开详情卡</i>'
    };
  }));
  edgesDS = new vis.DataSet(data.edges.map(function (e) {
    return { from: e.from, to: e.to, value: e.value, color: '#585b70',
             title: '共同发表 ' + e.value + ' 篇该方向论文' };
  }));

  if (network) network.destroy();
  network = new vis.Network($('graph'), { nodes: nodesDS, edges: edgesDS }, {
    nodes: { shape: 'dot', font: { color: '#cdd6f4', size: 13 } },
    edges: { smooth: false, scaling: { min: 1, max: 6 } },
    physics: {
      stabilization: { iterations: 200 },
      barnesHut: { gravitationalConstant: -3000, centralGravity: 0.75,
                   springLength: 70, springConstant: 0.05, damping: 0.5 }
    },
    interaction: { hover: true, tooltipDelay: 120 }
  });

  // Click opens detail card (Scholar is a button inside the card, not a direct jump)
  network.on('click', function (params) {
    if (!params.nodes.length) { closeDetail(); return; }
    openDetail(params.nodes[0]);
  });

  modified = []; activeCid = null; activeInst = null;
  buildLegend(data.legend);
  buildInstitutions();
  refreshListClusterOptions();
  renderShortlist();
  $('nodeSearch').value = '';
  $('nodeResults').innerHTML = '';
  $('legendPanel').style.display = currentView === 'graph' ? 'block' : 'none';
  $('searchPanel').style.display = currentView === 'graph' ? 'block' : 'none';
  $('viewToggle').style.display = 'flex';

  renderScopeBar();
  renderStatsBar();
  var s = data.stats || {};

  // Few-result hint: still show the graph, but offer relaxation chips on overlay briefly? 
  // Keep graph; if scholars are very few, surface suggestions under stats via overlay only on error.
  // Soft hint when papers_available is tiny relative to query wordiness:
  if (lastQuery && suggestQueries(lastQuery).length && s.papers_available != null && s.papers_available < 30) {
    // non-blocking: user already has a graph; skip overlay
  }

  var mySeq = searchSeq;
  network.once('stabilizationIterationsDone', function () {
    if (mySeq !== searchSeq) return;
    network.fit({ animation: false });
    setOverlay(false);
  });
  setTimeout(function () {
    if (mySeq !== searchSeq) return;   // 之后又发起了检索（可能已经在显示报错提示），别把它盖掉
    if (network) network.fit({ animation: false });
    setOverlay(false);
  }, 6000);
}

function renderStatsBar() {
  var s = lastStats || {};
  var txt;
  var tr0 = currentRange();
  var rangePrefix = tr0.years ? (tr0.label + ' · ') : '全部年份 · ';
  var inRange = tr0.years ? '（范围内 ' + s.papers_in_range + ' 篇）' : '';
  if (s.mode === 'deep') {
    txt = rangePrefix + '深度模式 · 检索 ' + s.papers_fetched + ' / ' + s.papers_available + ' 篇论文' + inRange + ' · '
        + s.seed_count + ' 位方向核心学者 + 合作圈共 ' + s.scholars + ' 人 · 合作关系 '
        + s.relations + ' 条 · 独立网络 ' + s.components + ' 个 · 研究团体 ' + s.communities + ' 个';
  } else {
    txt = rangePrefix + '检索 ' + s.papers_fetched + ' / ' + s.papers_available
        + ' 篇论文' + inRange + ' · 方向核心 ' + s.scholars + ' 位（该方向发文 ≥ ' + s.min_papers + ' 篇）· 合作关系 '
        + s.relations + ' 条 · 独立网络 ' + s.components + ' 个 · 研究团体 ' + s.communities + ' 个';
  }
  if (s.enriched) {
    txt += ' · 引用数 ' + s.with_citations + ' 人'
      + (s.with_citations_partial ? '（其中部分缺失 ' + s.with_citations_partial + ' 人，数值为下限）' : '')
      + ' / 署名机构 ' + s.with_institution + ' 人';
  }
  if (s.data_source && s.data_source !== 'dblp') {
    txt += ' · 数据源 ' + s.data_source + '（备用）';
  } else if (s.data_source === 'dblp') {
    txt += ' · 数据源 dblp';
  }
  if (s.degraded || s.enrich_status === 'degraded' || s.enrich_status === 'partial'
      || s.enrich_status === 'fallback' || (s.source_message && s.data_source && s.data_source !== 'dblp')) {
    if (s.enrich_status === 'degraded' || s.enrich_status === 'partial') {
      txt += ' · 补充已降级';
    }
    // 数据源回退与补充降级说的不是同一件事时两句都显示（同一行内用「；」分隔）
    var parts = [];
    if (s.source_message) parts.push(s.source_message);
    if (s.enrich_message && s.enrich_message !== s.source_message) parts.push(s.enrich_message);
    setNotice('degradeBanner', parts.join('；')
      || '引用/机构补充已降级：配额用尽或接口失败（主图仍可用，详情与短名单会标缺口）',
      { level: 'warn', dismissible: true });
  } else {
    clearNotice('degradeBanner');
  }
  $('stats').textContent = txt;
  $('stats').style.display = 'block';
}

function computeSizes() {
  var mode = $('sizeMode').value;
  var maxP = 1, maxC = 1;
  allNodes.forEach(function (n) {
    if (n.papers > maxP) maxP = n.papers;
    if (n.citations > maxC) maxC = n.citations;
  });
  sizeById = {};
  allNodes.forEach(function (n) {
    if (n.is_seed === false) { sizeById[n.id] = 10; return; }
    if (mode === 'citations') {
      sizeById[n.id] = citeStatus(n) === 'missing' ? 9 : Math.round((12 + 28 * (n.citations / maxC)) * 10) / 10;
    } else {
      sizeById[n.id] = Math.round((12 + 28 * (n.papers / maxP)) * 10) / 10;
    }
  });
}

function applySizeMode() {
  if (!nodesDS || !allNodes.length) return;
  computeSizes();
  restore();
  nodesDS.update(allNodes.map(function (n) {
    return { id: n.id, size: sizeById[n.id] };
  }));
}

function restore() {
  if (!modified.length) return;
  nodesDS.update(modified.map(function (id) {
    return { id: id, color: colorById[id], size: sizeById[id] };
  }));
  modified = [];
}

function highlight(id) {
  restore(); setActive(null); activeCid = null;
  var nb = network.getConnectedNodes(id);
  var ups = [{ id: id, color: '#f9e2af', size: sizeById[id] * 1.6 }];
  nb.forEach(function (x) { ups.push({ id: x, color: '#a6e3a1', size: sizeById[x] * 1.2 }); });
  nodesDS.update(ups);
  modified = [id].concat(nb);
  network.focus(id, { scale: 1.15, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
  openDetail(id);
}

function filterCommunity(cid) {
  restore();
  if (activeCid === cid) { activeCid = null; setActive(null); return; }
  nodesDS.update(allNodes.map(function (n) {
    var inC = n.community === cid;
    return { id: n.id, color: inC ? colorById[n.id] : DIM, size: inC ? sizeById[n.id] : 8 };
  }));
  modified = allNodes.map(function (n) { return n.id; });
  activeCid = cid; setActive(cid);
}

function setActive(cid) {
  Array.prototype.forEach.call(document.querySelectorAll('.legendItem'), function (it) {
    it.classList.toggle('active', cid !== null && Number(it.getAttribute('data-cid')) === cid);
  });
}

function buildLegend(legend) {
  var list = $('legendList'); list.innerHTML = '';
  var frag = document.createDocumentFragment();
  legend.forEach(function (e) {
    var it = document.createElement('div');
    it.className = 'legendItem'; it.setAttribute('data-cid', e.id);
    var d = document.createElement('div'); d.className = 'dot'; d.style.background = e.color;
    var l = document.createElement('div'); l.className = 'lbl'; l.textContent = e.label; l.title = e.label;
    var c = document.createElement('div'); c.className = 'cnt'; c.textContent = e.size;
    it.appendChild(d); it.appendChild(l); it.appendChild(c);
    it.onclick = function () { filterCommunity(e.id); };
    frag.appendChild(it);
  });
  list.appendChild(frag);
}

function buildInstitutions() {
  var box = $('instList');
  var counts = {};
  allNodes.forEach(function (n) {
    if (n.institution) counts[n.institution] = (counts[n.institution] || 0) + 1;
  });
  var ranked = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
  box.innerHTML = '';
  if (!ranked.length) {
    box.innerHTML = '<div class="empty">这批结果没有拿到论文署名机构（发表当时）</div>';
    return;
  }
  var missing = allNodes.filter(function (n) { return !n.institution; }).length;
  var frag = document.createDocumentFragment();
  ranked.slice(0, 40).forEach(function (name) {
    var it = document.createElement('div');
    it.className = 'instItem';
    it.setAttribute('data-inst', name);
    var l = document.createElement('div');
    l.className = 'lbl'; l.textContent = name; l.title = name + '（论文署名机构，发表当时）';
    var c = document.createElement('div');
    c.className = 'cnt'; c.textContent = counts[name];
    it.appendChild(l); it.appendChild(c);
    it.onclick = function () { filterInstitution(name); };
    frag.appendChild(it);
  });
  if (missing) {
    var note = document.createElement('div');
    note.className = 'empty';
    note.textContent = '另有 ' + missing + ' 位未获取到论文署名机构（发表当时；OpenAlex 缺元数据或补充降级）';
    frag.appendChild(note);
  }
  box.appendChild(frag);
}

function filterInstitution(name) {
  restore();
  if (activeInst === name) { activeInst = null; setActiveInst(null); return; }
  nodesDS.update(allNodes.map(function (n) {
    var hit = n.institution === name;
    return { id: n.id, color: hit ? colorById[n.id] : DIM, size: hit ? sizeById[n.id] : 8 };
  }));
  modified = allNodes.map(function (n) { return n.id; });
  activeInst = name; activeCid = null; setActive(null); setActiveInst(name);
}

function setActiveInst(name) {
  Array.prototype.forEach.call(document.querySelectorAll('.instItem'), function (it) {
    it.classList.toggle('active', name !== null && it.getAttribute('data-inst') === name);
  });
}

function copyText(text, btn) {
  if (!text) return;
  function ok() {
    if (!btn) return;
    var old = btn.textContent;
    btn.textContent = '已复制';
    setTimeout(function () { btn.textContent = old; }, 1200);
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(ok).catch(function () {
      fallbackCopy(text); ok();
    });
  } else {
    fallbackCopy(text); ok();
  }
}

function fallbackCopy(text) {
  var ta = document.createElement('textarea');
  ta.value = text; document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); } catch (e) {}
  document.body.removeChild(ta);
}

function openDetail(id) {
  var n = allNodes.find(function (x) { return x.id === id; });
  if (!n) return;
  var gaps = gapMarkers(n);
  var scholarUrl = scholarSearchUrl(n);
  var samples = (n.samples || []).slice(0, 5);
  var html = '';
  html += '<h3>' + esc(n.label) + '</h3>';
  html += '<div class="roleLine">' + (n.is_seed === false
    ? '<span class="roleCollab">经关联引入的合作者</span>（不一定做该方向）'
    : '<span class="roleCore">方向核心</span>') + '</div>';

  function field(k, v, copyVal) {
    var row = '<div class="field"><div class="k">' + esc(k) + '</div><div class="v">' + v;
    if (copyVal) {
      row += ' <button type="button" class="copyBtn" data-copy="' + esc(copyVal) + '">复制</button>';
    }
    row += '</div></div>';
    return row;
  }

  html += field('角色', esc(roleLabel(n)), roleLabel(n));
  html += field('时间范围', esc(rangeFullText()), rangeFullText());
  var rSuffix = currentRange().years ? '（' + rangeShort() + '）' : '';
  if (n.is_seed !== false) {
    html += field('该方向发文' + rSuffix, esc(n.papers) + ' 篇', String(n.papers));
  } else {
    html += field('该方向发文' + rSuffix, '—（经关联引入，非该方向检索命中）', '');
  }
  var cst = citeStatus(n);
  html += field('该方向相关引用' + rSuffix,
    cst === 'missing'
      ? '<span style="color:#f9e2af">未补全</span><br><span style="color:#f9e2af;font-size:11px">' + esc(citationNote(n)) + '</span>'
      : esc(citationText(n)) + ' 次'
        + (cst === 'partial' ? '<br><span style="color:#f9e2af;font-size:11px">' + esc(citationNote(n)) + '</span>' : ''),
    citationText(n));
  html += field('论文署名机构（发表当时）',
    n.institution ? esc(n.institution) : '<span style="color:#f9e2af">未获取（≠无单位）</span>',
    n.institution || '');
  html += field('研究主题',
    (n.topics && n.topics.length) ? esc(n.topics.join('、')) : '—',
    (n.topics && n.topics.length) ? n.topics.join(', ') : '');
  if (samples.length) {
    html += '<div class="field"><div class="k">代表作（该方向样本）</div><div class="v">'
      + samples.map(function (s) {
          return '· ' + esc(s.year) + ' ' + esc((s.title || '').substring(0, 80));
        }).join('<br>') + '</div></div>';
  }
  html += field('dblp id', esc(n.id), n.id);
  html += field('团体/聚类 ID', esc(n.community), String(n.community));
  html += field('数据置信度 / 缺口',
    gaps.length
      ? gaps.map(function (g) { return '<span class="gapTag">' + esc(g) + '</span>'; }).join('')
      : '<span style="color:#a6e3a1">引用与署名机构均已补全</span>',
    gaps.join(', '));
  if (lastStats && (lastStats.degraded || lastStats.enrich_status === 'degraded' || lastStats.enrich_status === 'partial')) {
    html += '<div class="field"><div class="k">全局补充状态</div><div class="v" style="color:#f9e2af">'
      + esc(lastStats.enrich_message || '引用/机构补充已降级') + '</div></div>';
  }
  html += field('Google Scholar',
    '<span style="color:#a6adc8;font-size:11px">作者搜索页（非精确主页；同名需人工甄别）</span><br>'
    + '<a href="' + esc(scholarUrl) + '" target="_blank" rel="noopener" style="color:#89b4fa;word-break:break-all;font-size:11px">'
    + esc(scholarUrl) + '</a>',
    scholarUrl);

  html += '<div class="actions">'
    + '<button type="button" class="btnSecondary" id="detailScholarBtn">打开 Scholar 搜索</button>'
    + '<button type="button" class="btnSecondary" id="detailCopyName">复制姓名</button>'
    + '<button type="button" class="btnSecondary" id="detailCopyAll">复制摘要字段</button>'
    + '</div>';

  $('detailBody').innerHTML = html;
  $('detailCard').style.display = 'block';
  // Hide the search panel while detail is open to avoid overlap on the right
  $('searchPanel').style.display = 'none';

  Array.prototype.forEach.call($('detailBody').querySelectorAll('.copyBtn'), function (btn) {
    btn.onclick = function (e) {
      e.stopPropagation();
      copyText(btn.getAttribute('data-copy'), btn);
    };
  });
  $('detailScholarBtn').onclick = function () {
    window.open(scholarUrl, '_blank', 'noopener');
  };
  $('detailCopyName').onclick = function () { copyText(n.label, $('detailCopyName')); };
  $('detailCopyAll').onclick = function () {
    var lines = [
      '姓名: ' + n.label,
      '角色: ' + roleLabel(n),
      '时间范围: ' + rangeFullText(),
      '该方向发文: ' + (n.is_seed === false ? '' : n.papers),
      '该方向相关引用: ' + citationText(n) + (citationNote(n) ? '（' + citationNote(n) + '）' : ''),
      '论文署名机构（发表当时）: ' + (n.institution || ''),
      '主题: ' + ((n.topics && n.topics.length) ? n.topics.join(', ') : ''),
      'dblp id: ' + n.id,
      '团体/聚类 ID: ' + n.community,
      '缺口: ' + gaps.join(', '),
      'Scholar 搜索: ' + scholarUrl
    ];
    copyText(lines.join('\\n'), $('detailCopyAll'));
  };
}

function closeDetail() {
  $('detailCard').style.display = 'none';
  if (currentView === 'graph' && allNodes.length) {
    $('searchPanel').style.display = 'block';
  }
}

function setView(view) {
  currentView = view;
  $('viewGraphBtn').classList.toggle('active', view === 'graph');
  $('viewListBtn').classList.toggle('active', view === 'list');
  if (view === 'list') {
    $('listView').classList.add('visible');
    $('legendPanel').style.display = 'none';
    $('searchPanel').style.display = 'none';
    closeDetail();
    renderShortlist();
  } else {
    $('listView').classList.remove('visible');
    if (allNodes.length) {
      $('legendPanel').style.display = 'block';
      $('searchPanel').style.display = 'block';
      if (network) network.fit({ animation: false });
    }
  }
}

function filteredNodes() {
  var nameQ = ($('listNameFilter').value || '').trim().toLowerCase();
  var minP = parseInt($('listMinPapers').value, 10); if (isNaN(minP) || minP < 0) minP = 0;
  var cluster = $('listClusterFilter').value;
  var coreOnly = $('listCoreOnly').checked;
  var hasInst = $('listHasInst').checked;
  return allNodes.filter(function (n) {
    if (nameQ && labelById[n.id].indexOf(nameQ) === -1) return false;
    if (n.papers < minP) return false;
    if (cluster !== '' && String(n.community) !== cluster) return false;
    if (coreOnly && n.is_seed === false) return false;
    if (hasInst && !n.institution) return false;
    return true;
  });
}

function refreshListClusterOptions() {
  var sel = $('listClusterFilter');
  var cur = sel.value;
  var ids = {};
  allNodes.forEach(function (n) { ids[n.community] = true; });
  var keys = Object.keys(ids).map(Number).sort(function (a, b) { return a - b; });
  sel.innerHTML = '<option value="">全部</option>';
  keys.forEach(function (cid) {
    var o = document.createElement('option');
    o.value = String(cid);
    o.textContent = '聚类 ' + cid;
    sel.appendChild(o);
  });
  if (cur && ids[Number(cur)] !== undefined) sel.value = cur;
}

function renderShortlist() {
  var rows = filteredNodes().slice().sort(compareShortlist);
  $('listMeta').textContent = '显示 ' + rows.length + ' / ' + allNodes.length + ' 人 · 时间范围：' + currentRange().label;
  var body = $('listBody');
  body.innerHTML = '';
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty">没有符合筛选条件的学者</td></tr>';
    return;
  }
  var frag = document.createDocumentFragment();
  rows.forEach(function (n) {
    var tr = document.createElement('tr');
    var gaps = gapMarkers(n);
    var url = scholarSearchUrl(n);
    tr.innerHTML =
      '<td>' + esc(n.label) + '</td>'
      + '<td>' + esc(n.institution || '') + '</td>'
      + '<td>' + (n.is_seed === false ? '' : esc(n.papers)) + '</td>'
      + '<td' + (citeStatus(n) === 'ok' ? '' : ' class="citeGap" title="' + esc(citationNote(n)) + '"') + '>'
        + esc(citationText(n)) + '</td>'
      + '<td class="' + (n.is_seed === false ? 'roleCollab' : 'roleCore') + '">' + esc(roleLabel(n)) + '</td>'
      + '<td>' + esc(n.community) + '</td>'
      + '<td><a href="' + esc(url) + '" target="_blank" rel="noopener">搜索页</a></td>'
      + '<td>' + (gaps.length
          ? gaps.map(function (g) { return '<span class="gapTag">' + esc(g) + '</span>'; }).join('')
          : '') + '</td>';
    frag.appendChild(tr);
  });
  body.appendChild(frag);
}

// 导出时间：固定格式的 ISO 8601（本地时间 + 时区偏移），如 2026-10-01T13:25:00+08:00。
// 不用 toLocaleString：它随浏览器语言变化，英文环境会带逗号，导致 Excel 错列。
function isoLocalTimestamp(d) {
  d = d || new Date();
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  var off = -d.getTimezoneOffset();
  var sign = off >= 0 ? '+' : '-';
  off = Math.abs(off);
  return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate())
    + 'T' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds())
    + sign + pad(Math.floor(off / 60)) + ':' + pad(off % 60);
}

// CSV 公式注入防护：文本列里以 = + - @ 制表符 回车 开头的单元格前面加单引号，
// 表格软件会把它当文本而不是公式。只用于文本列；数字列（方向发文数、方向相关引用、团体/聚类ID）不经过这里。
var CSV_FORMULA_START = /^[=+\\-@\\t\\r]/;
function csvText(v) {
  var s = String(v == null ? '' : v);
  return CSV_FORMULA_START.test(s) ? "'" + s : s;
}

// RFC 4180：含逗号、双引号、CR、LF 的字段整体加双引号，内部双引号写成两个
function csvCell(v) {
  var s = String(v == null ? '' : v);
  if (/[",\\r\\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

var CSV_HEADER = ['姓名', '论文署名机构（发表当时）', '方向发文数', '方向相关引用',
                  '角色标签', '团体/聚类ID', 'Scholar搜索URL', '数据置信/缺口标记',
                  '时间范围', '查询词', '导出时间', '免责声明'];

function csvDisclaimer() {
  var tr = currentRange();
  return '本短名单仅供学术结构探索与试用筛选；'
    + '论文署名机构（发表当时）反映论文元数据中的署名单位，不等于现职担保；'
    + 'Google Scholar 列为作者搜索页 URL，非精确个人主页，同名需人工甄别；'
    + '方向相关引用/机构可能因 OpenAlex 配额用尽、接口失败或元数据缺口而缺失：引用缺失时留空（勿当作 0），'
    + '部分论文缺失时为已取到部分之和（下限，下限为 0 时留空），均在「数据置信/缺口标记」列注明；引用为 0 表示真实被引 0 次；'
    + '文本列中以 = + - @ 制表符或回车开头的单元格已在前面加单引号，防止被当作公式执行；'
    + '时间范围口径：' + (tr.years ? tr.basis : '全部年份，不按年份过滤')
    + '，发文数、方向核心判定、合作关系、聚类、引用与排序均只统计该范围内论文'
    + (tr.years ? '（引用 = 范围内论文至今的累计被引）' : '');
}

// 第一行就是表头（没有注释行、没有尾部说明），pandas.read_csv / Excel 都能直接读；
// 查询词、时间范围、导出时间、免责声明作为独立列每行重复。
function buildCsv(rows, exportedAt) {
  var query = (lastStats && lastStats.query) || lastQuery || '';
  var rangeLabel = currentRange().label;
  var disclaimer = csvDisclaimer();
  var lines = [CSV_HEADER.map(csvCell).join(',')];
  rows.forEach(function (n) {
    lines.push([
      csvText(n.label),
      csvText(n.institution || ''),
      n.is_seed === false ? '' : n.papers,   // 数字列，不转义
      citationCsv(n),                         // 数字列：缺失留空，真实 0 写 0
      csvText(roleLabel(n)),
      n.community,                            // 数字列
      csvText(scholarSearchUrl(n)),
      csvText(gapMarkers(n).join('; ')),
      csvText(rangeLabel),
      csvText(query),
      csvText(exportedAt),
      csvText(disclaimer)
    ].map(csvCell).join(','));
  });
  return lines.join('\\r\\n') + '\\r\\n';
}

function exportCsv() {
  var rows = filteredNodes().slice().sort(compareShortlist);
  var text = buildCsv(rows, isoLocalTimestamp(new Date()));
  // UTF-8 BOM so Excel on Windows opens Chinese correctly
  var blob = new Blob(['\\ufeff' + text], { type: 'text/csv;charset=utf-8' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  var tr = currentRange();
  var query = (lastStats && lastStats.query) || lastQuery || 'export';
  a.download = 'talent-shortlist-' + query.replace(/ +/g, '_')
    + '-' + (tr.years ? (tr.start + '-' + tr.end) : 'all-years') + '.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
}

$('tabTeam').onclick = function () {
  $('tabTeam').classList.add('active'); $('tabInst').classList.remove('active');
  $('legendList').style.display = 'block'; $('instList').style.display = 'none';
};
$('tabInst').onclick = function () {
  $('tabInst').classList.add('active'); $('tabTeam').classList.remove('active');
  $('instList').style.display = 'block'; $('legendList').style.display = 'none';
};
$('sizeMode').onchange = applySizeMode;
$('resetBtn').onclick = function () {
  restore(); activeCid = null; activeInst = null; setActive(null); setActiveInst(null);
};
$('goBtn').onclick = runSearch;
$('topicInput').addEventListener('keydown', function (e) { if (e.key === 'Enter') runSearch(); });
$('detailClose').onclick = closeDetail;
// 报错提示还盖着时点「结构图 / 短名单」，先恢复上一次结果再切视图
function overlayBlocking() { return !$('overlay').classList.contains('hidden') && !currentES; }
$('viewGraphBtn').onclick = function () { if (overlayBlocking()) restoreLastResult(); setView('graph'); };
$('viewListBtn').onclick = function () { if (overlayBlocking()) restoreLastResult(); setView('list'); };
$('restoreLastBtn').onclick = restoreLastResult;
$('listNameFilter').addEventListener('input', function () { renderShortlist(); });
$('listMinPapers').addEventListener('input', function () { renderShortlist(); });
$('listClusterFilter').onchange = function () { renderShortlist(); };
$('listCoreOnly').onchange = function () { renderShortlist(); };
$('listHasInst').onchange = function () { renderShortlist(); };
$('exportCsvBtn').onclick = exportCsv;
// 已有结果时切换时间范围直接重跑：后端复用已抓取的论文，只按新范围重新统计。
// 检索进行中不打断（同一访客不能并发检索），新范围在下次检索时生效。
$('rangeSelect').onchange = function () {
  if (allNodes.length && lastQuery && !$('goBtn').disabled) runSearch();
};

var nsTimer;
$('nodeSearch').addEventListener('input', function (e) {
  clearTimeout(nsTimer);
  var q = e.target.value.trim().toLowerCase();
  nsTimer = setTimeout(function () {
    var box = $('nodeResults');
    if (!q) { box.innerHTML = ''; return; }
    var hits = allNodes.filter(function (n) { return labelById[n.id].indexOf(q) !== -1; });
    box.innerHTML = '';
    if (!hits.length) { box.innerHTML = '<div class="empty">没有匹配的学者</div>'; return; }
    var frag = document.createDocumentFragment();
    hits.slice(0, 30).forEach(function (n) {
      var d = document.createElement('div');
      d.className = 'hit';
      var meta = n.is_seed === false
        ? '经关联引入的合作者'
        : ('方向核心 · 该方向 ' + n.papers + ' 篇');
      d.innerHTML = '<div class="n">' + esc(n.label) + '</div><div class="m">' + esc(meta) + '</div>';
      d.onclick = function () { highlight(n.id); box.innerHTML = ''; e.target.value = n.label; };
      frag.appendChild(d);
    });
    if (hits.length > 30) {
      var m = document.createElement('div');
      m.className = 'empty'; m.textContent = '还有 ' + (hits.length - 30) + ' 位，请输入更精确的姓名';
      frag.appendChild(m);
    }
    box.appendChild(frag);
  }, 180);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # 部署到容器时（Hugging Face Spaces / Render 等）平台会通过 PORT 环境变量
    # 指定端口，且必须监听 0.0.0.0 才能被外部访问；本地不设置时仍然默认 8000，
    # 行为跟原来一样。
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("人才地图 Web 应用启动中...")
    print(f"浏览器打开: http://127.0.0.1:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
