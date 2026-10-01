"""
进程内缓存（TTL + LRU 容量上限）的小测试。不联网。

    python test_cache.py        # 直接跑
    python -m pytest test_cache.py
"""
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import topic_graph as tg  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_lru_evicts_least_recently_used():
    c = tg.TTLLRUCache(maxsize=3, ttl=60, clock=FakeClock())
    for k in "abc":
        c.put(k, k.upper())
    assert c.get("a") == "A"          # a 变成最近使用，b 成为最久未用
    c.put("d", "D")                   # 超过上限 -> 淘汰 b
    assert len(c) == 3
    assert c.get("b") is None
    assert c.keys() == ["c", "a", "d"]
    assert c.evictions == 1
    for i in range(10):               # 持续写入，始终不超过上限
        c.put(f"k{i}", i)
        assert len(c) <= 3
    assert c.evictions == 11


def test_put_existing_key_refreshes_without_growing():
    c = tg.TTLLRUCache(maxsize=2, ttl=60, clock=FakeClock())
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 3)                     # 覆盖，不算新条目
    c.put("c", 4)                     # 淘汰最久未用的 b
    assert c.get("a") == 3 and c.get("b") is None and len(c) == 2


def test_ttl_expiry_and_purge_on_put():
    clock = FakeClock()
    c = tg.TTLLRUCache(maxsize=10, ttl=30, clock=clock)
    c.put("old1", 1)
    c.put("old2", 2)
    clock.t += 20
    c.put("mid", 3)
    assert c.get("old1") == 1         # 未过期
    clock.t += 15                     # old* 已 35s（过期），mid 15s
    assert "old2" not in c
    c.put("new", 4)                   # 写入时顺带清掉所有过期项
    assert sorted(c.keys()) == ["mid", "new"]
    assert c.expirations == 2 and c.evictions == 0
    clock.t += 100
    assert c.get("mid") is None and c.get("new") is None


def test_contains_does_not_refresh_recency():
    c = tg.TTLLRUCache(maxsize=2, ttl=60, clock=FakeClock())
    c.put("a", 1)
    c.put("b", 2)
    assert "a" in c                   # 只查询
    c.put("c", 3)                     # a 仍是最久未用 -> 被淘汰
    assert c.get("a") is None and c.get("b") == 2


def test_thread_safety_under_concurrency():
    c = tg.TTLLRUCache(maxsize=50, ttl=60)
    errors = []

    def worker(n):
        try:
            for i in range(2000):
                c.put((n, i % 120), i)
                c.get((n, (i * 7) % 120))
                len(c)
                c.keys()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(c) == 50
    assert c.evictions > 0


def test_dblp_response_cache_is_bounded():
    cache = tg._DBLP_RESP_CACHE
    old_max = cache.maxsize
    cache.clear()
    cache.maxsize = 5
    try:
        for i in range(12):
            tg._dblp_cache_put(f"https://dblp.org/search/publ/api?q=x&f={i}", ([], 0))
        assert len(cache) == 5
        assert tg._dblp_cache_get("https://dblp.org/search/publ/api?q=x&f=0") is None
        assert tg._dblp_cache_get("https://dblp.org/search/publ/api?q=x&f=11") == ([], 0)
    finally:
        cache.clear()
        cache.maxsize = old_max


def test_env_overrides_and_app_caches_evict():
    """子进程里用环境变量把上限改小，确认 app 的三份缓存和 dblp 缓存都按上限淘汰。"""
    code = r'''
import app, topic_graph as tg
assert (app._RESULT_CACHE.maxsize, app._RAW_CACHE.maxsize, app._DEEP_CACHE.maxsize) == (3, 2, 4), \
    (app._RESULT_CACHE.maxsize, app._RAW_CACHE.maxsize, app._DEEP_CACHE.maxsize)
assert tg._DBLP_RESP_CACHE.maxsize == 6
for i in range(10):
    app._RESULT_CACHE.put(f"q{i}|300|2|all:None-None", {"i": i})
    app._RAW_CACHE.put(f"raw|q{i}|300", {"papers": [i]})
    app._DEEP_CACHE.put(f"q{i}|300|2|all:None-None|deep10", {"i": i})
assert len(app._RESULT_CACHE) == 3 and len(app._RAW_CACHE) == 2 and len(app._DEEP_CACHE) == 4
assert app._RAW_CACHE.keys() == ["raw|q8|300", "raw|q9|300"]
assert len(app._CACHE) == 9 and len(app._CACHE.keys()) == 9
print("ok")
'''
    env = dict(os.environ,
               TALENT_MAP_RESULT_CACHE_MAX="3", TALENT_MAP_RAW_CACHE_MAX="2",
               TALENT_MAP_DEEP_CACHE_MAX="4", TALENT_MAP_DBLP_CACHE_MAX="6")
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr


def test_env_int_rejects_bad_values():
    os.environ["TM_TEST_INT"] = "abc"
    assert tg.env_int("TM_TEST_INT", 7) == 7
    os.environ["TM_TEST_INT"] = "0"
    assert tg.env_int("TM_TEST_INT", 7) == 7
    os.environ["TM_TEST_INT"] = " 12 "
    assert tg.env_int("TM_TEST_INT", 7) == 12
    del os.environ["TM_TEST_INT"]
    assert tg.env_int("TM_TEST_INT", 7) == 7


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} passed")
