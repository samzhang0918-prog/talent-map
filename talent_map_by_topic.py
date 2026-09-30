"""
按论文方向联网检索人才地图（命令行版，生成静态 HTML 文件）。

用法：
    python3 talent_map_by_topic.py "BEV perception"
    python3 talent_map_by_topic.py "occupancy prediction" --papers 2000 --min-papers 3
    python3 talent_map_by_topic.py "lane detection" --years 3   # 只统计近 3 年（含当年）的论文
    python3 talent_map_by_topic.py --list          # 查看预置方向词表

如果想要"在网页里输入关键词、实时出图"的交互式版本，用 app.py（Web 应用）。
两者共用 topic_graph.py 里的检索与建图逻辑，区别只在结果怎么呈现：
这个脚本落成静态 HTML 文件，Web 应用则把数据推给前端实时渲染。
"""

import argparse
import json
import re

from pyvis.network import Network

import topic_graph
from topic_graph import PRESET_TOPICS, MAX_OFFSET


UI_CSS = """
<style>
#topicBar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 1001;
    background: rgba(24, 24, 37, 0.97); padding: 12px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4); color: #cdd6f4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#topicBar .title { font-size: 17px; font-weight: 600; }
#topicBar .title .kw { color: #89b4fa; }
#topicBar .stats { font-size: 12px; color: #a6adc8; margin-top: 3px; }
#searchContainer {
    position: fixed; top: 78px; right: 20px; z-index: 1000;
    background: rgba(30,30,46,0.95); padding: 15px; border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); max-width: 300px;
}
#searchInput {
    width: 100%; padding: 10px; border: 1px solid #45475a; border-radius: 5px;
    background: #313244; color: #cdd6f4; font-size: 14px; box-sizing: border-box;
}
#searchInput:focus { outline: none; border-color: #89b4fa; }
#searchResults { margin-top: 10px; max-height: 280px; overflow-y: auto; background: #1e1e2e; border-radius: 5px; }
.searchResult { padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #313244; color: #cdd6f4; font-size: 13px; }
.searchResult:hover { background: #313244; }
.searchResult .name { font-weight: bold; color: #89b4fa; }
.searchResult .info { font-size: 11px; color: #a6adc8; margin-top: 2px; }
.noResults { padding: 10px; text-align: center; color: #a6adc8; font-size: 13px; }
#legendContainer {
    position: fixed; top: 78px; left: 20px; z-index: 1000;
    background: rgba(30,30,46,0.95); padding: 15px; border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); max-width: 260px; color: #cdd6f4;
}
#legendContainer h3 { margin: 0 0 8px 0; font-size: 14px; }
#legendResetBtn {
    display: inline-block; margin-bottom: 8px; padding: 4px 10px; font-size: 12px;
    color: #cdd6f4; background: #313244; border: 1px solid #45475a;
    border-radius: 4px; cursor: pointer;
}
#legendResetBtn:hover { background: #45475a; }
#legendList { max-height: 340px; overflow-y: auto; }
.legendItem { display: flex; align-items: center; padding: 5px 6px; cursor: pointer; border-radius: 4px; font-size: 12px; }
.legendItem:hover { background: #313244; }
.legendItem.active { background: #45475a; }
.legendDot { width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
.legendLabel { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.legendCount { color: #a6adc8; margin-left: 6px; flex-shrink: 0; }
</style>
"""

UI_JS_TEMPLATE = """
<script type="text/javascript">
(function () {
    var originalNodesData = [], nodeLabels = {}, colorById = {}, sizeById = {};
    var legendData = __LEGEND_DATA__;
    var modifiedNodeIds = [], activeCommunityId = null;
    var MAX_RESULTS_SHOWN = 30, DIMMED = '#3a3a4a';

    function initGraphData() {
        originalNodesData = nodes.get();
        originalNodesData.forEach(function (n) {
            nodeLabels[n.id] = (n.label || '').toLowerCase();
            // 记住每个节点本来的颜色和大小：节点按团体上色、按发文数定大小，
            // 重置时必须各自还原，不能统一刷成一个值。
            colorById[n.id] = n.color;
            sizeById[n.id] = n.size;
        });
        console.log('图数据就绪，共 ' + originalNodesData.length + ' 位学者');
    }

    function restoreModified() {
        if (!modifiedNodeIds.length) return;
        nodes.update(modifiedNodeIds.map(function (id) {
            return { id: id, color: colorById[id], size: sizeById[id] };
        }));
        modifiedNodeIds = [];
    }

    function highlightNode(nodeId) {
        restoreModified();
        setActiveLegend(null);
        activeCommunityId = null;
        var neighbors = network.getConnectedNodes(nodeId);
        var updates = [{ id: nodeId, color: '#f9e2af', size: sizeById[nodeId] * 1.6 }];
        neighbors.forEach(function (nb) {
            updates.push({ id: nb, color: '#a6e3a1', size: sizeById[nb] * 1.2 });
        });
        nodes.update(updates);
        modifiedNodeIds = [nodeId].concat(neighbors);
        network.focus(nodeId, { scale: 1.2, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
    }

    function filterCommunity(cid) {
        restoreModified();
        if (activeCommunityId === cid) { activeCommunityId = null; setActiveLegend(null); return; }
        // 一次性算好全部节点的目标样式再单批提交，避免逐个 update 造成的卡顿
        nodes.update(originalNodesData.map(function (n) {
            var inC = n.community === cid;
            return { id: n.id, color: inC ? colorById[n.id] : DIMMED, size: inC ? sizeById[n.id] : 8 };
        }));
        modifiedNodeIds = originalNodesData.map(function (n) { return n.id; });
        activeCommunityId = cid;
        setActiveLegend(cid);
    }

    function setActiveLegend(cid) {
        document.querySelectorAll('.legendItem').forEach(function (it) {
            it.classList.toggle('active', cid !== null && Number(it.getAttribute('data-cid')) === cid);
        });
    }

    function buildLegend() {
        var list = document.getElementById('legendList');
        if (!list) return;
        var frag = document.createDocumentFragment();
        legendData.forEach(function (e) {
            var item = document.createElement('div');
            item.className = 'legendItem';
            item.setAttribute('data-cid', e.id);
            var dot = document.createElement('div');
            dot.className = 'legendDot'; dot.style.background = e.color;
            var label = document.createElement('div');
            label.className = 'legendLabel'; label.textContent = e.label; label.title = e.label;
            var cnt = document.createElement('div');
            cnt.className = 'legendCount'; cnt.textContent = e.size;
            item.appendChild(dot); item.appendChild(label); item.appendChild(cnt);
            item.onclick = function () { filterCommunity(e.id); };
            frag.appendChild(item);
        });
        list.appendChild(frag);
        var btn = document.getElementById('legendResetBtn');
        if (btn) btn.onclick = function () { restoreModified(); activeCommunityId = null; setActiveLegend(null); };
    }

    function findNodes(q) {
        var lower = q.toLowerCase(), out = [];
        originalNodesData.forEach(function (n) {
            if ((nodeLabels[n.id] || '').indexOf(lower) !== -1) {
                out.push({ id: n.id, name: n.label, info: (n.title || '').replace(/<[^>]*>/g, '').split('\\n')[0] });
            }
        });
        return out;
    }

    function showResults(results) {
        var div = document.getElementById('searchResults');
        div.innerHTML = '';
        if (!results.length) { div.innerHTML = '<div class="noResults">未找到匹配的学者</div>'; return; }
        var frag = document.createDocumentFragment();
        results.slice(0, MAX_RESULTS_SHOWN).forEach(function (r) {
            var d = document.createElement('div');
            d.className = 'searchResult';
            d.innerHTML = '<div class="name">' + r.name + '</div><div class="info">' + r.info + '</div>';
            d.onclick = function () {
                highlightNode(r.id);
                div.innerHTML = '';
                document.getElementById('searchInput').value = r.name;
            };
            frag.appendChild(d);
        });
        if (results.length > MAX_RESULTS_SHOWN) {
            var more = document.createElement('div');
            more.className = 'noResults';
            more.textContent = '还有 ' + (results.length - MAX_RESULTS_SHOWN) + ' 个结果，请输入更精确的关键词';
            frag.appendChild(more);
        }
        div.appendChild(frag);
    }

    // drawGraph() 已在此脚本插入点之前同步执行完毕，nodes / network 都已就绪
    initGraphData();
    buildLegend();

    var input = document.getElementById('searchInput'), timer;
    if (input) {
        input.addEventListener('input', function (e) {
            var q = e.target.value.trim();
            clearTimeout(timer);
            if (!q) { document.getElementById('searchResults').innerHTML = ''; return; }
            timer = setTimeout(function () { showResults(findNodes(q)); }, 200);
        });
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                var rs = findNodes(e.target.value.trim());
                if (rs.length) { highlightNode(rs[0].id); document.getElementById('searchResults').innerHTML = ''; }
            } else if (e.key === 'Escape') {
                e.target.value = '';
                document.getElementById('searchResults').innerHTML = '';
            }
        });
    }
    document.addEventListener('click', function (e) {
        var c = document.getElementById('searchContainer');
        if (c && !c.contains(e.target)) document.getElementById('searchResults').innerHTML = '';
    });
})();
</script>
"""


def inject_ui(html, legend_data, topic, stats):
    """把顶栏 / 搜索框 / 图例注入 pyvis 生成的 HTML。"""
    topic_bar = f"""
<div id="topicBar">
    <div class="title">论文方向人才地图：<span class="kw">{topic}</span></div>
    <div class="stats">{stats}</div>
</div>
<div id="searchContainer">
    <input type="text" id="searchInput" placeholder="搜索学者姓名..." autocomplete="off" />
    <div id="searchResults"></div>
</div>
<div id="legendContainer">
    <h3>研究团体（按合作紧密度聚类）</h3>
    <div id="legendResetBtn">重置视图</div>
    <div id="legendList"></div>
</div>
"""
    html = html.replace("</head>", UI_CSS + "\n</head>", 1)
    html = html.replace("<body>", "<body>\n" + topic_bar, 1)

    # 脚本必须插在包含 drawGraph(); 的那个 <script> 块闭合之后。
    # 直接接在 drawGraph(); 后面会把新的 <script> 标签嵌进尚未闭合的脚本块里，
    # 触发 "Unexpected token '<'" 语法错误，导致 network/nodes 根本不会被赋值、页面卡在进度条。
    ui_js = UI_JS_TEMPLATE.replace("__LEGEND_DATA__", json.dumps(legend_data, ensure_ascii=False))
    marker = html.find("drawGraph();")
    close = html.find("</script>", marker) if marker != -1 else -1
    if close != -1:
        at = close + len("</script>")
        html = html[:at] + "\n" + ui_js + html[at:]
    else:
        html = html.replace("</body>", ui_js + "\n</body>", 1)
    return html


def render(payload, topic, out_file):
    """把共享模块产出的图数据落成一个静态 HTML 文件。"""
    net = Network(height="calc(100vh - 70px)", width="100%",
                  bgcolor="#1e1e2e", font_color="white")

    legend = payload["legend"]
    for n in payload["nodes"]:
        samples = "<br>".join(
            f"· {s['year']} {(s['title'] or '')[:60]}" for s in n["samples"][:3]
        )
        # 深度模式里 is_seed=False 的是滚雪球带进来的合作者，他们不一定做这个方向，
        # 提示里要说清楚，别让人误以为是该方向的核心学者
        head = (f"<b>{n['label']}</b><br><i>合作者（经该方向学者关联引入）</i>"
                if n.get("is_seed") is False
                else f"<b>{n['label']}</b><br>该方向发文: {n['papers']} 篇")
        extra = ""
        if n.get("citations"):
            extra += f"<br>该方向被引: {n['citations']} 次"
        if n.get("institution"):
            extra += f"<br>机构: {n['institution']}"
        if n.get("topics"):
            extra += f"<br>主题: {'、'.join(n['topics'])}"
        title = (f"{head}{extra}<br>团体: {legend[n['community']]['label']}"
                 f"<br>dblp: {n['id']}" + (f"<br><br>代表作:<br>{samples}" if samples else ""))
        net.add_node(n["id"], label=n["label"], color=n["color"],
                     title=title, size=n["size"], community=n["community"])

    for e in payload["edges"]:
        net.add_edge(e["from"], e["to"], value=e["value"],
                     title=f"共同发表 {e['value']} 篇论文", color="#585b70")

    net.toggle_physics(True)

    s = payload["stats"]
    tr = s.get("time_range") or {}
    if tr.get("years"):
        range_prefix = (f"时间范围 {tr['label']}（{tr['basis']}，范围内 "
                        f"{s.get('papers_in_range')} 篇）· ")
    else:
        range_prefix = "时间范围 全部年份 · "
    if s.get("mode") == "deep":
        stats = (f"深度模式 · 检索 {s['papers_fetched']} / {s['papers_available']} 篇论文 · "
                 f"{s['seed_count']} 位方向核心学者 + 合作圈共 {s['scholars']} 人 · "
                 f"合作关系 {s['relations']} 条 · 独立网络 {s['components']} 个 · "
                 f"研究团体 {s['communities']} 个")
    else:
        stats = (f"检索 {s['papers_fetched']} / {s['papers_available']} 篇论文 · "
                 f"核心学者 {s['scholars']} 位（该方向发文 ≥ {s['min_papers']} 篇）· "
                 f"合作关系 {s['relations']} 条 · 独立网络 {s['components']} 个 · "
                 f"研究团体 {s['communities']} 个")
    if s.get("enriched"):
        stats += f" · 引用数 {s['with_citations']} 人 / 机构 {s['with_institution']} 人"
    stats = range_prefix + stats
    html = inject_ui(net.generate_html(), legend, topic, stats)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    return stats


def safe_filename(topic):
    return re.sub(r"[^\w\-]+", "_", topic).strip("_").lower()


def main():
    parser = argparse.ArgumentParser(
        description="按论文方向联网检索并生成人才地图（静态 HTML）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n  python3 talent_map_by_topic.py "BEV perception"\n'
               '  python3 talent_map_by_topic.py "occupancy prediction" --papers 2000 --min-papers 3\n\n'
               '想要网页里实时检索的交互版本，改用：python3 app.py',
    )
    parser.add_argument("topic", nargs="?", help="论文方向关键词（1~2 个核心词效果最好）")
    parser.add_argument("--papers", type=int, default=1000,
                        help=f"最多抓取多少篇论文（默认 1000，dblp 上限 {MAX_OFFSET}）")
    parser.add_argument("--min-papers", type=int, default=2,
                        help="学者在该方向至少发过几篇才入图（默认 2）")
    parser.add_argument("--deep", action="store_true",
                        help="深度模式：用该方向 top 学者做滚雪球扩展，把散落的研究组连成网络"
                             "（每位种子约 14 秒，默认 10 位约 2~3 分钟）")
    parser.add_argument("--seeds", type=int, default=10,
                        help="深度模式的种子学者数量（默认 10）")
    parser.add_argument("--years", default="all", choices=["all", "5", "3"],
                        help="时间范围：all=全部（默认）/ 5=近 5 年 / 3=近 3 年；按论文发表年份、"
                             "含当年，发文数/方向核心/合作边/引用都只统计范围内论文")
    parser.add_argument("--list", action="store_true", help="列出预置方向词表")
    args = parser.parse_args()

    if args.list:
        print("预置方向词表（也可以直接传任意关键词，不限于这些）：\n")
        for t in PRESET_TOPICS:
            print(f"  {t}")
        return

    if not args.topic:
        parser.print_help()
        print("\n提示：用 --list 查看预置方向词表")
        return

    def on_progress(fetched, total, message):
        print(f"  {message}")

    time_range = topic_graph.resolve_time_range(args.years)
    print(f"正在 dblp 检索方向：{args.topic}（时间范围：{time_range['label']}）")
    payload = topic_graph.search_topic(args.topic, args.papers, args.min_papers, on_progress,
                                       time_range=time_range)

    if payload.get("error"):
        print(f"\n{payload['error']}")
        return

    if args.deep:
        print(f"\n开始深度扩展（{args.seeds} 位种子学者，dblp 限速下约需 "
              f"{args.seeds * 14 // 60 + 1} 分钟）...")
        deep_payload = topic_graph.deep_expand(payload, args.seeds, on_progress=on_progress)
        if deep_payload.get("error"):
            print(f"深度扩展失败：{deep_payload['error']}（改用快速结果）")
        else:
            payload = deep_payload

    suffix = "" if time_range["years"] is None else f"_{time_range['start']}-{time_range['end']}"
    out_file = f"talent_map_{safe_filename(args.topic)}{suffix}.html"
    stats = render(payload, args.topic, out_file)
    print(f"\n[成功] {stats}")
    print(f"已生成 {out_file}，双击即可打开。")


if __name__ == "__main__":
    main()
