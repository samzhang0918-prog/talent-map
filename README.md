# 论文方向人才地图

输入一个论文方向关键词，实时联网检索全网论文，把该方向的研究者和他们的合作关系画成一张可交互的人才地图。

不依赖任何预设名单——每次检索都是现场从 dblp 全库拉数据构建的。

## 功能

- **按方向实时检索**：输入 `BEV perception`、`occupancy prediction` 之类的关键词，十几秒出图
- **合作网络**：节点是研究者，边是共同署名过论文，边粗细代表合作篇数
- **研究团体聚类**：用 Louvain 社区发现把合作紧密的人分组，按团体着色
- **多维度信息**：每位学者的该方向发文量、被引次数、所属机构、研究主题
- **节点大小可切换**：按发文量看谁产出多，按引用量看谁影响力大——两种口径的排序往往差别很大
- **机构分布**：列出机构及人数，点击可高亮该机构的学者
- **深度模式**：把散落的研究组连成一张连通网络（见下文）
- **跳转 Google Scholar**：点击图上任意一个学者节点，新标签页打开该学者的 Google
  Scholar 作者搜索结果（带姓名和机构，帮助缩小范围——dblp 没有 Scholar 主页链接，
  姓名到 profile 没有可靠的精确映射，所以是搜索结果页而不是直接跳到某个具体页面）

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 网页版（推荐）

```bash
python3 app.py
```

然后打开 http://127.0.0.1:8000

### 公开部署（让别人不用你开着电脑也能访问）

这个应用是 SSE 长连接、深度模式单次检索能跑几分钟，Vercel / Netlify /
Cloudflare Workers 这类 serverless 平台都有几十秒的执行时间上限，检索到一半
就会被掐断，所以只能部署到常驻容器平台。推荐 **[Hugging Face
Spaces](https://huggingface.co/new-space)**（免费常驻、Docker SDK 直接能跑、
受众也是学术圈），Render / Fly.io / Railway 这些同样能用同一个 Dockerfile。

仓库根目录带了 `Dockerfile`，本地验证：

```bash
docker build -t talent-map .
docker run -p 7860:7860 -e OPENALEX_MAILTO="you@example.com" talent-map
```

部署到 Hugging Face Spaces：

1. 打开 [huggingface.co/new-space](https://huggingface.co/new-space)，SDK 选
   **Docker**，关联这个 GitHub 仓库（或者 `git remote add hf <space 地址>` 后
   `git push hf main`）
2. 在 Space 的 Settings → Variables and secrets 里按需加：
   - `OPENALEX_MAILTO`：你的邮箱，不设也能用，只是配额更保守
   - `CONTACT_URL`：能联系到你的一个链接，会附在请求的 User-Agent 里——公开
     长期跑之后请求量不再是偶发的自用流量，让 dblp/OpenAlex 出问题时能找到人，
     而不是直接封 IP
3. Space 会自动 build 这个 Dockerfile，几分钟后给你一个固定公网网址

跟"本地 + cloudflared 隧道临时分享"（见上面 `python3 app.py` 那种用法）的区别：
隧道网址每次重启都变、你关电脑就失效，适合发给几个人临时试用；公开部署网址固定
长期有效，电脑不用开着，适合真的想让不特定的人都能访问。

公开部署时几处内建的保护机制：

| 机制 | 做什么 | 为什么 |
|---|---|---|
| 全局 dblp 节流 | 同一时刻整个进程只有一个 dblp 请求在飞 | 访客多的时候如果各自独立限速，叠加起来会集体触发 dblp 的限流，一个人被封连累所有人 |
| 单访客并发限制 | 同一 IP 有检索在跑时，新请求会被拒绝并提示稍等 | 防止一个人开好几个标签页占满上面这条全局队列 |
| 参数上限收紧 | `papers` 上限从 dblp 的技术上限 10000 降到 2000，`seeds` 从 40 降到 15 | 不收紧的话一次请求能占住队列 7~9 分钟，命令行版不受此限制 |
| OpenAlex 配额降级 | 每天 1000 credits 是全站共享的，用完后引用数/机构补充会静默跳过 | 不影响检索本身，只是那部分数据缺失——界面本来就如实标注覆盖率，不会误导成"没有" |

以及一条限制：进程内缓存（30 分钟 TTL）只在单实例里有效，**只能部署单副本**
（`--workers 1` 或平台默认的单实例设置），开多副本会让缓存命中率下降、
连带把 dblp 压力推高。

### 命令行版（生成静态 HTML 文件）

```bash
python3 talent_map_by_topic.py "BEV perception"
python3 talent_map_by_topic.py "occupancy prediction" --papers 2000 --min-papers 3
python3 talent_map_by_topic.py "corner case" --deep --seeds 10
python3 talent_map_by_topic.py --list          # 查看预置方向词表
```

## 关键词怎么写

**只用 1~2 个核心词。** dblp 的多个关键词之间是 AND 关系，词堆得越多结果越少：

| 查询 | 命中论文 |
|---|---|
| `BEV` | 4979 篇 |
| `BEV perception` | 135 篇 |
| `BEV perception autonomous driving` | 19 篇 |

## 深度模式解决什么问题

直接按方向检索建出来的图是高度碎片化的。实测 `lane detection` 的 123 位学者散成 30 个互不相连的小组，最大的一组才 16 人。

这不是采集方式的问题，而是客观事实：同一方向的研究组之间本来就各写各的，没有共同作者就连不起来。试过"只在这些核心学者之间补边"——查了 top 20 人的全部论文，只补出 1 条边，他们是真的互不相识。

深度模式换了个思路：以该方向发文最多的学者为**种子**，把他们各自的合作者也拉进图里。因为每个新节点都是顺着一条边被引入的，连通性是构造出来的：

| 方向 | | 人数 | 独立网络数 |
|---|---|---|---|
| lane detection | 快速 | 97 | 20 |
| | 深度 | 216 | **8** |
| occupancy prediction | 快速 | 224 | 37 |
| | 深度 | 75 | **4** |
| corner case | 深度 | 37 | **1** |

代价是图里会混入不做这个方向的人（种子在其他领域的合作者），所以界面上做了区分：种子是该方向的核心学者，其余标注为"合作者（经该方向学者关联引入）"。

耗时：每位种子约 14 秒（dblp 限速所致），10 位约 2~3 分钟。网页版会先把快速结果给你看，滚雪球在后台跑完再自动替换整张图。

## 数据来源

**dblp**（`dblp.org/search/publ/api`）— 论文、作者身份、合作关系

用官方检索 API 而不是抓 `/pid/*.xml` 页面，因为 dblp 的 robots.txt 里写了 `Disallow: /*.xml`。

**OpenAlex**（`api.openalex.org`）— 被引次数、署名机构、研究主题

按**论文 DOI** 批量精确匹配，而不是按人名查。这个区别很关键：按人名查会撞上同名问题（"Jianping Shi" 在 OpenAlex 上有 5 个同名候选，判不出是哪一个）；按 DOI 定位到具体论文后，再在这一篇论文内部把作者对上号，候选集就只剩这几位作者，实测 94% 的 dblp 署名能正确匹配。

成本也差得远：按人名查用的是全文 `search`（10 credits/次），按 DOI 用 `filter` 精确匹配，一次请求带 50 个 DOI 只花 1 credit。

### 可选：设置 OpenAlex polite pool 标识

```bash
export OPENALEX_MAILTO="your@email.com"
```

带上邮箱能拿到更稳定的配额和更快的响应。**不是密钥，不设也能正常用**，只是会被归到匿名池。

## 数据覆盖率

补充数据并非人人都有，界面上会如实标注（"引用数 89 人 / 机构 88 人"、"另有 16 位未获取到机构"）。

缺口来自逐层衰减：dblp 论文约 84% 带 DOI → OpenAlex 命中其中约 96% → 作者匹配率 94% → 但并非所有论文都有机构元数据。

**所以机构列表里没出现的机构，不代表那个人没有单位，只代表这批论文的元数据里没有。** 拿它做判断时请注意这个区别。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| 论文数 | 300 | 最多抓多少篇。dblp 分页硬上限 10000 |
| 最少发文 | 2 | 学者在该方向至少发过几篇才入图。调高得到核心圈，调低更全 |
| 深度模式种子数 | 10 | 滚雪球的种子学者数量，每位约 14 秒 |

## 已知限制

- dblp 单个方向最多返回 10000 篇论文，超大方向会被截断
- dblp 会间歇性返回 HTTP 500（与请求本身无关，重试即可），代码里已做退避重试
- dblp 要求 `Crawl-delay: 4`，所以抓取速度有硬性下限
- 机构信息来自论文署名，反映的是**发表当时**的单位，不是当前任职

## 文件说明

| 文件 | 作用 |
|---|---|
| `app.py` | Web 应用（FastAPI + SSE 进度推送） |
| `topic_graph.py` | 检索、建图、聚类、数据补充的核心逻辑 |
| `talent_map_by_topic.py` | 命令行版，生成静态 HTML |
| `lib/` | vis-network 前端库（本地提供，避免 CDN 不可达时页面空白） |
| `Dockerfile` | 公开部署用（Hugging Face Spaces / Render 等），见上文"公开部署" |

## 许可

[MIT](LICENSE)

本项目仅使用 dblp 与 OpenAlex 的公开学术数据。这些数据各自的使用条款以其官方说明为准，
MIT 许可仅适用于本仓库的代码。
