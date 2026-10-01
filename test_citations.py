"""
被引数解析（citation_count_or_none）的取值矩阵 + 三条数据路径的端到端小测试。不联网。

    python test_citations.py        # 直接跑
    python -m pytest test_citations.py

端到端部分把 topic_graph.SESSION.get 换成本地假响应，响应体是原始 JSON 文本
（含 1e400 / Infinity / NaN 这类 Python json 会解析成 inf / nan 的写法），
覆盖 Crossref 备用源、OpenAlex 备用源、dblp 主路径 + OpenAlex 补充三条路径。
"""
import json
import math
import os
import sys
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import topic_graph as tg  # noqa: E402

f = tg.citation_count_or_none
MAX = tg.CITATION_MAX


def test_value_matrix():
    cases = [
        # (输入, 期望)
        (None, None), (True, None), (False, None),
        (0, 0), (5, 5), (-1, None), (MAX, MAX), (MAX + 1, None), (10 ** 12, None), (10 ** 5000, None),
        (5.0, 5), (0.0, 0), (-0.0, 0), (5.5, None), (-5.0, None), (1e12, None),
        (float("inf"), None), (float("-inf"), None), (float("nan"), None),
        ("5", 5), (" 5 ", 5), ("0", 0), ("\t7\n", 7), ("1000000000", MAX), ("1000000001", None),
        ("999999999999", None),            # 12 位，长度允许但超上限
        ("1" * 13, None),                  # 13 位，超长度
        ("5" * 5000, None),                # 5000 位，超过 Python int 位数上限也不抛
        ("²", None), ("٥", None), ("５", None), ("-5", None), ("+5", None), ("5.0", None),
        ("1e3", None), ("", None), ("   ", None), ("abc", None),
        ([], None), ({}, None), ([5], None), (Decimal("5"), None), (b"5", None),
    ]
    bad = [(v if not isinstance(v, str) or len(v) < 20 else f"<{len(v)} 位字符串>", f(v), exp)
           for v, exp in cases if f(v) != exp or type(f(v)) is not type(exp)]
    assert not bad, bad


def test_json_parsed_specials():
    """JSON 里的 1e400 / -1e400 / Infinity / -Infinity / NaN 经 json.loads 后是 inf / nan，一律按缺失。"""
    for raw in ("1e400", "-1e400", "Infinity", "-Infinity", "NaN"):
        v = json.loads('{"c": %s}' % raw)["c"]
        assert isinstance(v, float) and not math.isfinite(v)
        assert f(v) is None, raw
    assert f(json.loads('{"c": 5.0}')["c"]) == 5
    assert f(json.loads('{"c": "5"}')["c"]) == 5


def test_never_raises():
    class Weird:
        def __int__(self):
            raise OverflowError
    for v in (Weird(), object(), type, lambda: 5, float("nan"), "²" * 3, "9" * 10000):
        assert f(v) is None


# ---------------------------------------------------------------------------
# 端到端：假的 HTTP 层
class FakeResp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text
        self.content = text.encode()
        self.headers = {"content-type": "application/json"}
        self.ok = 200 <= status < 400

    def json(self):
        return json.loads(self.text)   # 与 requests 一样用标准 json：1e400 -> inf，Infinity -> inf

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _raw(obj, specials):
    """json.dumps 后把占位字符串替换成 JSON 原文里的 1e400 / Infinity / NaN 等写法。"""
    text = json.dumps(obj)
    for k, v in specials.items():
        text = text.replace(json.dumps(k), v)
    return text


SPECIALS = {"__1E400__": "1e400", "__INF__": "Infinity", "__NAN__": "NaN", "__NEGINF__": "-1e400"}
# 每位作者两篇论文；第一篇的被引字段取下面的值，第二篇统一是 3
VALUES = {"Big": "__1E400__", "Inf": "__INF__", "Nan": "__NAN__", "Neg": "__NEGINF__",
          "Sup": "²", "Huge": 10 ** 12, "Five": 5.0, "Zero": 0, "Missing": "__ABSENT__"}


def _install(router):
    orig_get, orig_sleep = tg.SESSION.get, tg.time.sleep
    tg.SESSION.get = lambda url, params=None, **kw: router(url, params or {})
    tg.time.sleep = lambda s: None
    tg._DBLP_RESP_CACHE.clear() if hasattr(tg._DBLP_RESP_CACHE, "clear") else None
    return orig_get, orig_sleep


def _restore(saved):
    tg.SESSION.get, tg.time.sleep = saved


def _check_payload(pl, label_of):
    assert not pl.get("error"), pl.get("error")
    # 严格 JSON：任何 inf / nan 漏进 payload 都会让这里抛 ValueError
    json.dumps({"nodes": pl["nodes"], "stats": pl["stats"]}, allow_nan=False)
    N = {label_of(n["label"]): n for n in pl["nodes"]}
    for key in ("Big", "Inf", "Nan", "Neg", "Sup", "Huge"):
        assert (N[key]["citations"], N[key]["citation_status"], N[key]["citation_papers"]) == (3, "partial", [1, 2]), (key, N[key])
    assert (N["Five"]["citations"], N["Five"]["citation_status"]) == (8, "ok"), N["Five"]
    assert (N["Zero"]["citations"], N["Zero"]["citation_status"]) == (3, "ok"), N["Zero"]
    assert (N["Missing"]["citations"], N["Missing"]["citation_status"]) == (3, "partial"), N["Missing"]
    return N


def _dblp_fail(url, params):
    return FakeResp(404, '{"error":"blocked"}')


def test_e2e_crossref_fallback_with_1e400():
    items = []
    for i, (key, val) in enumerate(VALUES.items()):
        for j, v in enumerate((val, 3)):
            w = {"title": [f"x {key} {j}"], "DOI": f"10.123/{key}{j}", "published": {"date-parts": [[2025]]},
                 "author": [{"given": key, "family": "Xr"}, {"given": key + "b", "family": "Xr"}]}
            if v != "__ABSENT__":
                w["is-referenced-by-count"] = v
            items.append(w)

    def router(url, params):
        if "dblp.org" in url:
            return _dblp_fail(url, params)
        if "api.openalex.org" in url:
            return FakeResp(404, '{"error":"no"}')
        if "api.crossref.org" in url:
            body = {"message": {"total-results": len(items), "items": items if int(params.get("offset", 0)) == 0 else []}}
            return FakeResp(200, _raw(body, SPECIALS))
        return FakeResp(404, "{}")
    saved = _install(router)
    try:
        pl = tg.search_topic("citeinf crossref", 300, 2, None)
    finally:
        _restore(saved)
    assert pl["stats"]["data_source"] == "crossref"
    _check_payload(pl, lambda lab: lab.split()[0])


def test_e2e_openalex_fallback_with_1e400():
    results = []
    for key, val in VALUES.items():
        for j, v in enumerate((val, 3)):
            w = {"id": f"W{key}{j}", "display_name": f"o {key} {j}", "publication_year": 2025,
                 "doi": f"https://doi.org/10.124/{key}{j}", "topics": [],
                 "authorships": [{"author": {"id": f"https://openalex.org/A{key}", "display_name": f"{key} Oa"}, "institutions": []},
                                 {"author": {"id": f"https://openalex.org/A{key}b", "display_name": f"{key}b Oa"}, "institutions": []}]}
            if v != "__ABSENT__":
                w["cited_by_count"] = v
            results.append(w)

    def router(url, params):
        if "dblp.org" in url:
            return _dblp_fail(url, params)
        if "api.openalex.org" in url and params.get("search"):
            body = {"meta": {"count": len(results), "next_cursor": None}, "results": results}
            return FakeResp(200, _raw(body, SPECIALS))
        return FakeResp(404, "{}")
    saved = _install(router)
    try:
        pl = tg.search_topic("citeinf openalex", 300, 2, None)
    finally:
        _restore(saved)
    assert pl["stats"]["data_source"] == "openalex"
    _check_payload(pl, lambda lab: lab.split()[0])


def test_e2e_dblp_with_openalex_enrich_1e400():
    hits, works = [], []
    for key, val in VALUES.items():
        for j, v in enumerate((val, 3)):
            doi = f"10.125/{key}{j}".lower()
            hits.append({"info": {"title": f"d {key} {j}", "year": "2025", "doi": doi,
                                  "authors": {"author": [{"@pid": f"d/{key}", "text": f"{key} Db"},
                                                         {"@pid": f"d/{key}b", "text": f"{key}b Db"}]}}})
            w = {"doi": "https://doi.org/" + doi, "authorships": [], "topics": []}
            if v != "__ABSENT__":
                w["cited_by_count"] = v
            works.append(w)

    def router(url, params):
        if "dblp.org" in url:
            first = str(params.get("f", "0")) == "0" or "f=0" in url
            body = {"result": {"hits": {"@total": str(len(hits)), "hit": hits if first else []}}}
            return FakeResp(200, json.dumps(body))
        if "api.openalex.org" in url and str(params.get("filter", "")).startswith("doi:"):
            want = set(params["filter"][4:].split("|"))
            sel = [w for w in works if w["doi"].replace("https://doi.org/", "") in want]
            return FakeResp(200, _raw({"results": sel}, SPECIALS))
        return FakeResp(404, "{}")
    saved = _install(router)
    try:
        pl = tg.search_topic("citeinf dblp", 300, 2, None)
    finally:
        _restore(saved)
    assert pl["stats"]["data_source"] == "dblp"
    _check_payload(pl, lambda lab: lab.split()[0])


if __name__ == "__main__":
    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, repr(exc)[:400])
    print(f"{len(tests) - failed} passed" + (f", {failed} failed" if failed else ""))
    sys.exit(1 if failed else 0)
