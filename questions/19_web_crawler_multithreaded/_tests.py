"""Hidden tests for Question 19: Web Crawler Multithreaded
Run: python questions/19_web_crawler_multithreaded/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
from starter import HtmlParser, get_hostname, crawl_single_threaded, crawl


def build_large_graph(num_nodes: int = 50) -> tuple[dict[str, list[str]], str]:
    """Build a graph with many nodes to make single-threaded obviously slow."""
    hostname = "http://example.com"
    urls = [f"{hostname}/page/{i}" for i in range(num_nodes)]
    edges: dict[str, list[str]] = {}

    for i, url in enumerate(urls):
        links = []
        for j in range(1, 4):
            links.append(urls[(i + j) % num_nodes])
        links.append(urls[(i * 7) % num_nodes])
        if i % 10 == 0:
            links.append(f"http://other-site-{i}.com/page")
        edges[url] = links

    start_url = urls[0]
    return edges, start_url


def run_and_time(label: str, func, *args) -> tuple[list[str] | None, float]:
    """Run a function, measure wall-clock time, return (result, elapsed)."""
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    return result, elapsed


def test_basic():
    """Basic example from the problem description."""
    print("TEST: Basic example")

    edges = {
        "http://news.yahoo.com/news": [
            "http://news.yahoo.com/news/topics/",
            "http://news.google.com",
        ],
        "http://news.yahoo.com/news/topics/": [
            "http://news.yahoo.com/news",
            "http://news.yahoo.com",
        ],
        "http://news.google.com": [
            "http://news.yahoo.com",
            "http://news.yahoo.com/news",
        ],
        "http://news.yahoo.com": [],
    }
    parser = HtmlParser(edges, delay=0.1)
    start_url = "http://news.yahoo.com/news"

    expected = {
        "http://news.yahoo.com/news",
        "http://news.yahoo.com/news/topics/",
        "http://news.yahoo.com",
    }

    result_st, time_st = run_and_time(
        "Single-threaded", crawl_single_threaded, start_url, parser
    )
    assert set(result_st) == expected, f"Single-threaded wrong: {set(result_st)}"

    result_mt, time_mt = run_and_time("Multithreaded", crawl, start_url, parser)
    if result_mt is not None:
        assert set(result_mt) == expected, f"Multithreaded wrong: {set(result_mt)}"
        print(f"  Speedup: {time_st / time_mt:.1f}x")
    else:
        print("  crawl() returned None -- not yet implemented")

    print("  [PASS]\n")


def test_large_graph():
    """Larger graph to make the performance difference obvious."""
    print("TEST: Large graph (50 nodes, ~100ms per call)")

    edges, start_url = build_large_graph(num_nodes=50)
    parser = HtmlParser(edges, delay=0.1)

    result_st, time_st = run_and_time(
        "Single-threaded", crawl_single_threaded, start_url, parser
    )

    result_mt, time_mt = run_and_time(
        "Multithreaded", crawl, start_url, parser
    )

    if result_mt is not None:
        assert set(result_mt) == set(result_st), (
            f"Results differ!\n"
            f"  Single-threaded found {len(result_st)} URLs\n"
            f"  Multithreaded found {len(result_mt)} URLs\n"
            f"  Missing: {set(result_st) - set(result_mt)}\n"
            f"  Extra: {set(result_mt) - set(result_st)}"
        )
        print(f"  Correctness: PASS (both found {len(result_st)} URLs)")
        print(f"  Speedup: {time_st / time_mt:.1f}x")
        if time_mt < time_st / 3:
            print("  Performance: PASS (significant speedup)")
        else:
            print("  Performance: WARN (expected more speedup)")
    else:
        print("  crawl() returned None -- not yet implemented")
    print()


def test_single_url():
    """Edge case: start_url has no outgoing links."""
    print("TEST: Single URL (no outgoing links)")

    edges: dict[str, list[str]] = {"http://lonely.com": []}
    parser = HtmlParser(edges, delay=0.05)
    start_url = "http://lonely.com"

    result_mt, _ = run_and_time("Multithreaded", crawl, start_url, parser)
    if result_mt is not None:
        assert set(result_mt) == {"http://lonely.com"}, f"Wrong: {result_mt}"
        print("  Correctness: PASS")
    else:
        print("  crawl() returned None -- not yet implemented")
    print()


def test_all_different_hostnames():
    """Edge case: all linked URLs have different hostnames."""
    print("TEST: All different hostnames")

    edges = {
        "http://start.com/page": [
            "http://other1.com/page",
            "http://other2.com/page",
            "http://other3.com/page",
        ],
    }
    parser = HtmlParser(edges, delay=0.05)
    start_url = "http://start.com/page"

    result_mt, _ = run_and_time("Multithreaded", crawl, start_url, parser)
    if result_mt is not None:
        assert set(result_mt) == {"http://start.com/page"}, f"Wrong: {result_mt}"
        print("  Correctness: PASS")
    else:
        print("  crawl() returned None -- not yet implemented")
    print()


def run_tests():
    print("=" * 60)
    print("Web Crawler Multithreaded — Hidden Tests")
    print("=" * 60 + "\n")

    test_basic()
    test_single_url()
    test_all_different_hostnames()
    test_large_graph()

    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
