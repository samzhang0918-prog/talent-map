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
"""

import asyncio
import json
import threading
import time

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

# key -> (存入时间, 结果)。进程内缓存，重启即失效，够用了。
_CACHE = {}
_CACHE_TTL = 30 * 60
_CACHE_LOCK = threading.Lock()

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


def cache_get(key):
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < _CACHE_TTL:
            return hit[1]
        if hit:
            del _CACHE[key]
    return None


def cache_put(key, value):
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


def strip_internal(payload):
    """去掉内部字段（networkx 图对象无法 JSON 序列化）。"""
    return {k: v for k, v in payload.items() if not k.startswith("_")}


@app.get("/api/search")
async def search(request: Request, q: str, papers: int = 300, min_papers: int = 2,
                 deep: bool = False, seeds: int = 10):
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
    """
    q = (q or "").strip()
    # 公开部署时把上限收紧一些：papers=10000 (dblp 硬上限) 单次要抓 7 分钟以上，
    # seeds=40 的深度模式要跑将近 4 分钟，一个访客就能把全局节流闸占满，
    # 让其他人排很久的队。命令行版（talent_map_by_topic.py）不走这个接口，不受影响。
    papers = max(100, min(papers, PUBLIC_MAX_PAPERS))
    min_papers = max(1, min(min_papers, 20))
    seeds = max(3, min(seeds, PUBLIC_MAX_SEEDS))
    client_ip = get_client_ip(request)

    async def event_stream():
        if not q:
            yield sse("error", {"message": "请输入方向关键词"})
            return

        # 同一访客不能同时开好几个检索：dblp 请求是全局串行的（见 topic_graph.dblp_get），
        # 一个人开几个标签页会把队列占满，让其他访客等更久。
        with _INFLIGHT_LOCK:
            if client_ip in _INFLIGHT_IPS:
                yield sse("error", {"message": "你有一个检索还在进行中，请等它结束后再发起新的搜索"})
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
                    fast_key = f"{q}|{papers}|{min_papers}"
                    fast = cache_get(fast_key)
                    if fast is None:
                        fast = topic_graph.search_topic(
                            q, papers, min_papers, make_reporter("progress"))
                        if not fast.get("error"):
                            cache_put(fast_key, fast)
                    else:
                        make_reporter("progress")(0, 0, "命中缓存，直接返回")

                    loop.call_soon_threadsafe(queue.put_nowait, ("fast_result", fast))
                    if fast.get("error") or not deep:
                        loop.call_soon_threadsafe(queue.put_nowait, ("finish", None))
                        return

                    deep_key = f"{q}|{papers}|{min_papers}|deep{seeds}"
                    deep_res = cache_get(deep_key)
                    if deep_res is None:
                        deep_res = topic_graph.deep_expand(
                            fast, seeds, on_progress=make_reporter("deep_progress"))
                        if not deep_res.get("error"):
                            cache_put(deep_key, deep_res)
                    loop.call_soon_threadsafe(queue.put_nowait, ("deep_result", deep_res))
                    loop.call_soon_threadsafe(queue.put_nowait, ("finish", None))
                except Exception as exc:
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
                        yield sse("error", {"message": data["error"]})
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
    return {"topics": topic_graph.PRESET_TOPICS}


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
  .instItem { display: flex; align-items: center; padding: 5px 6px; border-radius: 4px;
              cursor: pointer; font-size: 12px; }
  .instItem:hover { background: #313244; }
  .instItem.active { background: #45475a; }
  #deepBanner {
    position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(137,180,250,.14); border: 1px solid #89b4fa; color: #cdd6f4;
    padding: 7px 16px; border-radius: 18px; font-size: 12px; z-index: 6; display: none;
  }
  #presets { margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap; }
  .chip {
    padding: 4px 11px; font-size: 12px; background: #313244; border: 1px solid #45475a;
    border-radius: 20px; cursor: pointer; color: #bac2de;
  }
  .chip:hover { background: #45475a; color: #cdd6f4; }
  /* min-height:0 + 绝对定位撑满：flex item 的子元素用 height:100% 在部分浏览器里
     解析不出高度（父元素 height 是 auto，由 flex 算法决定），容器一旦算成 0 高，
     vis-network 就会画在一块看不见的画布上，页面表现为"图没出来"。 */
  main { flex: 1; min-height: 0; position: relative; }
  #graph { position: absolute; inset: 0; }
  .panel {
    position: absolute; background: rgba(30,30,46,.96); padding: 14px;
    border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,.4); z-index: 5;
  }
  #legendPanel { top: 16px; left: 16px; width: 250px; display: none; }
  #searchPanel  { top: 16px; right: 16px; width: 260px; display: none; }
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
  .bar { width: 300px; height: 4px; background: #313244; border-radius: 2px; margin: 14px auto 0; overflow: hidden; }
  .bar div { height: 100%; background: #89b4fa; width: 0; transition: width .3s; }
  .bar.hidden { display: none; }
  #stats {
    position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
    background: rgba(24,24,37,.94); padding: 7px 18px; border-radius: 18px;
    font-size: 12px; color: #a6adc8; z-index: 5; display: none; white-space: nowrap;
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
    <div class="opt">节点大小
      <select id="sizeMode">
        <option value="papers">按发文量</option>
        <option value="citations">按引用量</option>
      </select>
    </div>
    <label class="opt deepOpt" title="先出快速结果，再用该方向 top 学者做滚雪球扩展，把散落的研究组连成网络。实测每位种子学者约需 14 秒（dblp 限速），10 位约 2~3 分钟，期间可以先看快速结果">
      <input type="checkbox" id="deepInput" /> 深度模式
    </label>
    <button id="goBtn">检索</button>
  </div>
  <div id="presets"></div>
</header>

<main>
  <div id="graph"></div>
  <div class="panel" id="legendPanel">
    <div class="tabs">
      <div class="tab active" id="tabTeam">研究团体</div>
      <div class="tab" id="tabInst">机构分布</div>
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
  <div id="stats"></div>
  <div id="deepBanner"></div>
  <div id="overlay">
    <div class="box">
      <div class="spinner hidden" id="spinner"></div>
      <div id="msg">输入一个论文方向，开始检索</div>
      <div id="sub">
        数据来自 dblp 全网论文库，不限于任何预设名单。<br />
        关键词建议只用 1~2 个核心词 —— dblp 多个词之间是 AND 关系，词越多结果越少
        （"BEV" 有 4979 篇，"BEV perception autonomous driving" 只剩 19 篇）。<br />
        dblp 要求每次请求间隔 4 秒，检索 300 篇约需 10 秒。
      </div>
      <div class="bar hidden" id="bar"><div id="barFill"></div></div>
    </div>
  </div>
</main>

<script>
var network = null, nodesDS = null, edgesDS = null;
var colorById = {}, sizeById = {}, labelById = {}, allNodes = [];
var modified = [], activeCid = null, currentES = null, deepStart = null;
var DIM = '#3a3a4a';

var $ = function (id) { return document.getElementById(id); };

fetch('/api/presets').then(function (r) { return r.json(); }).then(function (d) {
  var box = $('presets');
  d.topics.forEach(function (t) {
    var c = document.createElement('span');
    c.className = 'chip'; c.textContent = t;
    c.onclick = function () { $('topicInput').value = t; runSearch(); };
    box.appendChild(c);
  });
});

function setOverlay(show, opts) {
  opts = opts || {};
  $('overlay').classList.toggle('hidden', !show);
  if (!show) return;
  $('msg').textContent = opts.msg || '';
  $('sub').innerHTML = opts.sub || '';
  $('spinner').classList.toggle('hidden', !opts.busy);
  $('bar').classList.toggle('hidden', !opts.progress);
  if (opts.progress) $('barFill').style.width = opts.percent + '%';
}

function runSearch() {
  var q = $('topicInput').value.trim();
  if (!q) { $('topicInput').focus(); return; }
  if (currentES) currentES.close();

  $('goBtn').disabled = true;
  $('legendPanel').style.display = 'none';
  $('searchPanel').style.display = 'none';
  $('stats').style.display = 'none';
  setOverlay(true, { busy: true, msg: '正在检索「' + q + '」...',
                     sub: '正在联网查询 dblp 论文库', progress: true, percent: 0 });

  var deep = $('deepInput').checked;
  $('deepBanner').style.display = 'none';
  var url = '/api/search?q=' + encodeURIComponent(q)
          + '&papers=' + encodeURIComponent($('papersInput').value)
          + '&min_papers=' + encodeURIComponent($('minInput').value)
          + '&deep=' + (deep ? 'true' : 'false');
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
    if (deep) {
      // 深度模式下先让用户看到快速结果，滚雪球在后台继续，完成后再替换整张图
      deepStart = Date.now();
      $('deepBanner').style.display = 'block';
      $('deepBanner').textContent = '深度模式：正在扩展合作网络，完成后自动更新（可先浏览当前结果）';
    } else {
      es.close(); currentES = null; $('goBtn').disabled = false;
    }
  });

  es.addEventListener('deep_progress', function (e) {
    var d = JSON.parse(e.data);
    // 用已完成种子的实际耗时推算剩余时间，比写死一个估计值可靠
    var extra = '';
    if (deepStart && d.done > 0 && d.total > d.done) {
      var perSeed = (Date.now() - deepStart) / d.done;
      var leftSec = Math.round(perSeed * (d.total - d.done) / 1000);
      extra = leftSec >= 60
        ? '，预计还需 ' + Math.ceil(leftSec / 60) + ' 分钟'
        : '，预计还需 ' + leftSec + ' 秒';
    }
    $('deepBanner').style.display = 'block';
    $('deepBanner').textContent = '深度模式：' + d.message + extra;
  });

  es.addEventListener('deep_done', function (e) {
    es.close(); currentES = null; $('goBtn').disabled = false;
    render(JSON.parse(e.data));
    $('deepBanner').style.display = 'none';
  });

  es.addEventListener('deep_error', function (e) {
    es.close(); currentES = null; $('goBtn').disabled = false;
    var m = '扩展失败';
    try { m = JSON.parse(e.data).message; } catch (err) {}
    $('deepBanner').textContent = '深度扩展失败：' + m + '（快速结果仍可用）';
  });

  es.addEventListener('error', function (e) {
    es.close(); currentES = null; $('goBtn').disabled = false;
    var m = '检索失败';
    try { m = JSON.parse(e.data).message; } catch (err) {}
    setOverlay(true, { busy: false, msg: '没有结果', sub: m });
  });

  // EventSource 自身的连接错误（服务没起来 / 断网）走 onerror，
  // 和上面服务端主动推的 error 事件是两回事，要分开处理。
  es.onerror = function () {
    if (currentES !== es) return;
    es.close(); currentES = null; $('goBtn').disabled = false;
    setOverlay(true, { busy: false, msg: '连接中断',
                       sub: '与本地服务的连接断开了，确认 app.py 仍在运行后重试。' });
  };
}

function render(data) {
  // 没有 vis 就直接说清楚，别让页面静默停在一片空白上
  if (typeof vis === 'undefined' || !vis.Network) {
    setOverlay(true, { busy: false, msg: '图形库未能加载',
      sub: 'vis-network 没加载成功，图无法绘制。<br>请确认 app.py 与 lib/ 目录在同一个文件夹下，然后刷新页面。' });
    return;
  }
  allNodes = data.nodes;
  colorById = {}; sizeById = {}; labelById = {};
  allNodes.forEach(function (n) {
    // 记住每个节点本来的颜色：节点按团体上色，高亮/筛选之后必须各自还原，
    // 不能统一刷成同一个值。大小由 computeSizes() 按当前口径算。
    colorById[n.id] = n.color;
    labelById[n.id] = (n.label || '').toLowerCase();
  });

  computeSizes();
  nodesDS = new vis.DataSet(allNodes.map(function (n) {
    var samples = (n.samples || []).slice(0, 3).map(function (s) {
      return '· ' + s.year + ' ' + (s.title || '').substring(0, 60);
    }).join('<br>');
    // 深度模式里 is_seed=false 的是滚雪球带进来的合作者，他们不一定做这个方向，
    // 提示里要说清楚，不能让人误以为是该方向的核心学者
    var head = n.is_seed === false
      ? '<b>' + n.label + '</b><br><i>合作者（经该方向学者关联引入）</i>'
      : '<b>' + n.label + '</b><br>该方向发文: ' + n.papers + ' 篇';
    var extra = '';
    if (n.citations) extra += '<br>该方向被引: ' + n.citations + ' 次';
    if (n.institution) extra += '<br>机构: ' + n.institution;
    if (n.topics && n.topics.length) extra += '<br>主题: ' + n.topics.join('、');
    return {
      id: n.id, label: n.label, color: n.color, size: sizeById[n.id], community: n.community,
      title: head + extra + '<br>dblp: ' + n.id + (samples ? '<br><br>代表作:<br>' + samples : '')
        + '<br><br><i>点击跳转 Google Scholar</i>'
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
    // 一个方向的检索结果往往包含很多互不相连的小研究组（彼此没有共同作者）。
    // 独立组之间只有斥力、没有弹簧拉力，斥力一大就会把它们推到很远，
    // 整张图缩放后节点只剩几个像素、什么都看不清。所以这里把斥力压到 vis 默认量级，
    // 同时把向心力调高，让这些散落的小组聚拢到画面中央。
    physics: {
      stabilization: { iterations: 200 },
      barnesHut: { gravitationalConstant: -3000, centralGravity: 0.75,
                   springLength: 70, springConstant: 0.05, damping: 0.5 }
    },
    interaction: { hover: true, tooltipDelay: 120 }
  });

  // dblp 不提供作者的 Google Scholar 主页链接，也没有可靠的第三方 API 能把姓名
  // 精确映射到某个 Scholar profile（同名学者很常见）。所以不去猜一个具体链接，
  // 而是跳转到 Google Scholar 的"作者搜索"结果页，让用户自己从候选里选——
  // 带上机构信息（如果有）帮助缩小范围，但不保证第一个结果就是本人。
  network.on('click', function (params) {
    if (!params.nodes.length) return;
    var n = allNodes.find(function (x) { return x.id === params.nodes[0]; });
    if (!n) return;
    var q = '"' + n.label + '"' + (n.institution ? ' "' + n.institution + '"' : '');
    window.open('https://scholar.google.com/citations?view_op=search_authors&mauthors='
      + encodeURIComponent(q), '_blank', 'noopener');
  });

  modified = []; activeCid = null; activeInst = null;
  buildLegend(data.legend);
  buildInstitutions();
  $('nodeSearch').value = '';
  $('nodeResults').innerHTML = '';
  $('legendPanel').style.display = 'block';
  $('searchPanel').style.display = 'block';

  var s = data.stats;
  var txt;
  if (s.mode === 'deep') {
    txt = '深度模式 · 检索 ' + s.papers_fetched + ' / ' + s.papers_available + ' 篇论文 · '
        + s.seed_count + ' 位方向核心学者 + 合作圈共 ' + s.scholars + ' 人 · 合作关系 '
        + s.relations + ' 条 · 独立网络 ' + s.components + ' 个 · 研究团体 ' + s.communities + ' 个';
  } else {
    txt = '检索 ' + s.papers_fetched + ' / ' + s.papers_available
        + ' 篇论文 · 核心学者 ' + s.scholars + ' 位（该方向发文 ≥ ' + s.min_papers + ' 篇）· 合作关系 '
        + s.relations + ' 条 · 独立网络 ' + s.components + ' 个 · 研究团体 ' + s.communities + ' 个';
  }
  if (s.enriched) {
    txt += ' · 引用数 ' + s.with_citations + ' 人 / 机构 ' + s.with_institution + ' 人';
  }
  $('stats').textContent = txt;
  $('stats').style.display = 'block';

  // 布局稳定后必须 fit 一次，否则初始视口是按空图定的，节点会跑到画面外
  network.once('stabilizationIterationsDone', function () {
    network.fit({ animation: false });
    setOverlay(false);
  });
  // 兜底：万一 stabilization 事件没如期触发，也不能让遮罩一直盖着
  setTimeout(function () {
    if (network) network.fit({ animation: false });
    setOverlay(false);
  }, 6000);
}

function computeSizes() {
  // 两种口径分别归一化：按发文量看谁在这个方向产出多，按引用量看谁影响力大。
  // 引用数缺失的（OpenAlex 没匹配到）统一给最小尺寸，不能当成 0 影响力去参与归一化。
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
      sizeById[n.id] = n.citations ? Math.round((12 + 28 * (n.citations / maxC)) * 10) / 10 : 9;
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
}

function filterCommunity(cid) {
  restore();
  if (activeCid === cid) { activeCid = null; setActive(null); return; }
  // 一次性算好全部节点的目标样式再单批提交，避免上千节点逐个 update 造成卡顿
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
    box.innerHTML = '<div class="empty">这批结果没有拿到机构信息</div>';
    return;
  }
  var missing = allNodes.filter(function (n) { return !n.institution; }).length;
  var frag = document.createDocumentFragment();
  ranked.slice(0, 40).forEach(function (name) {
    var it = document.createElement('div');
    it.className = 'instItem';
    it.setAttribute('data-inst', name);
    var l = document.createElement('div');
    l.className = 'lbl'; l.textContent = name; l.title = name;
    var c = document.createElement('div');
    c.className = 'cnt'; c.textContent = counts[name];
    it.appendChild(l); it.appendChild(c);
    it.onclick = function () { filterInstitution(name); };
    frag.appendChild(it);
  });
  if (missing) {
    var note = document.createElement('div');
    note.className = 'empty';
    // 覆盖率要如实说明，不能让人以为没列出的机构就是不存在
    note.textContent = '另有 ' + missing + ' 位未获取到机构（OpenAlex 缺少该论文的机构元数据）';
    frag.appendChild(note);
  }
  box.appendChild(frag);
}

var activeInst = null;
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
      d.innerHTML = '<div class="n">' + n.label + '</div><div class="m">该方向 ' + n.papers + ' 篇</div>';
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
