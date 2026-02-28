"""
Question 19: Web Crawler Multithreaded (LeetCode 1242)

Implement a multithreaded web crawler that crawls all links under the same
hostname as start_url. The get_urls() call is I/O-bound (~100ms per call),
so a single-threaded solution is too slow.

Run this file to see the single-threaded baseline, then implement crawl()
and compare performance.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class HtmlParser:
    """Simulates a web server that returns links for a given URL."""

    def __init__(self, edges: dict[str, list[str]], delay: float = 0.1):
        self.edges = edges
        self.delay = delay

    def get_urls(self, url: str) -> list[str]:
        time.sleep(self.delay)  # Simulate network latency
        return self.edges.get(url, [])


def get_hostname(url: str) -> str:
    """Extract hostname from URL.

    Examples:
        http://news.yahoo.com/news/topics/ -> news.yahoo.com
        http://news.yahoo.com/news         -> news.yahoo.com
        http://news.google.com             -> news.google.com
    """
    # Remove http://
    url = url.split("://", 1)[1] if "://" in url else url
    # Get everything before first /
    return url.split("/", 1)[0]


# ---------------------------------------------------------------------------
# Reference: Single-threaded solution (TOO SLOW for large graphs)
# ---------------------------------------------------------------------------

def crawl_single_threaded(start_url: str, parser: HtmlParser) -> list[str]:
    """Reference single-threaded solution. This is too slow for large inputs."""
    hostname = get_hostname(start_url)
    visited = set()
    queue = [start_url]
    visited.add(start_url)

    while queue:
        url = queue.pop(0)
        for next_url in parser.get_urls(url):
            if next_url not in visited and get_hostname(next_url) == hostname:
                visited.add(next_url)
                queue.append(next_url)

    return list(visited)


# ---------------------------------------------------------------------------
# TODO: Implement this function
# ---------------------------------------------------------------------------

def crawl(start_url: str, parser: HtmlParser) -> list[str]:
    """
    Multithreaded web crawler. Implement this.

    Must crawl all URLs under the same hostname as start_url.
    Must use multithreading for acceptable performance.
    Must be thread-safe (no duplicate visits, no race conditions).

    Hints:
    - Use ThreadPoolExecutor or threading.Thread to parallelize get_urls calls.
    - Protect shared state (the visited set) with a Lock or use thread-safe structures.
    - Think about how to detect when all work is done (no more URLs to crawl).

    Args:
        start_url: The starting URL to begin crawling from.
        parser: An HtmlParser instance that provides get_urls(url).

    Returns:
        A list of all URLs under the same hostname reachable from start_url.
    """
    pass


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def build_large_graph(num_nodes: int = 50) -> tuple[dict[str, list[str]], str]:
    """Build a graph with many nodes to make single-threaded obviously slow.

    Creates a fan-out structure where each node links to several others,
    plus some cross-links. All under the same hostname except a few decoys.
    """
    hostname = "http://example.com"
    urls = [f"{hostname}/page/{i}" for i in range(num_nodes)]
    edges: dict[str, list[str]] = {}

    for i, url in enumerate(urls):
        links = []
        # Each page links to the next 3 pages (wrapping)
        for j in range(1, 4):
            links.append(urls[(i + j) % num_nodes])
        # Add a couple of cross-links
        links.append(urls[(i * 7) % num_nodes])
        # Add one external link (different hostname) as a decoy
        if i % 10 == 0:
            links.append(f"http://other-site-{i}.com/page")
        edges[url] = links

    start_url = urls[0]
    return edges, start_url


def run_and_time(label: str, func, *args) -> tuple[list[str] | None, float]:
    """Run a function, measure wall-clock time, return (result, elapsed)."""
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    if result is not None:
        print(f"  Found {len(result)} URLs in {elapsed:.2f}s")
    else:
        print(f"  Returned None in {elapsed:.2f}s")
    return result, elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic():
    """Basic example from the problem description."""
    print("\n" + "#" * 60)
    print("TEST: Basic example")
    print("#" * 60)

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

    # Single-threaded
    result_st, time_st = run_and_time(
        "Single-threaded", crawl_single_threaded, start_url, parser
    )
    assert set(result_st) == expected, f"Single-threaded wrong: {set(result_st)}"

    # Multithreaded
    result_mt, time_mt = run_and_time("Multithreaded", crawl, start_url, parser)
    if result_mt is not None:
        assert set(result_mt) == expected, f"Multithreaded wrong: {set(result_mt)}"
        print(f"\n  Speedup: {time_st / time_mt:.1f}x")
    else:
        print("\n  crawl() returned None -- not yet implemented")


def test_large_graph():
    """Larger graph to make the performance difference obvious."""
    print("\n" + "#" * 60)
    print("TEST: Large graph (50 nodes, ~100ms per call)")
    print("#" * 60)

    edges, start_url = build_large_graph(num_nodes=50)
    parser = HtmlParser(edges, delay=0.1)

    # Single-threaded (this will take ~5 seconds)
    result_st, time_st = run_and_time(
        "Single-threaded (expect ~5s)", crawl_single_threaded, start_url, parser
    )

    # Multithreaded (should be much faster)
    result_mt, time_mt = run_and_time(
        "Multithreaded (should be <1s)", crawl, start_url, parser
    )

    if result_mt is not None:
        # Verify correctness: same set of URLs
        assert set(result_mt) == set(result_st), (
            f"Results differ!\n"
            f"  Single-threaded found {len(result_st)} URLs\n"
            f"  Multithreaded found {len(result_mt)} URLs\n"
            f"  Missing: {set(result_st) - set(result_mt)}\n"
            f"  Extra: {set(result_mt) - set(result_st)}"
        )
        print(f"\n  Correctness: PASS (both found {len(result_st)} URLs)")
        print(f"  Speedup: {time_st / time_mt:.1f}x")
        if time_mt < time_st / 3:
            print("  Performance: PASS (significant speedup)")
        else:
            print("  Performance: WARN (expected more speedup)")
    else:
        print("\n  crawl() returned None -- not yet implemented")


def test_single_url():
    """Edge case: start_url has no outgoing links."""
    print("\n" + "#" * 60)
    print("TEST: Single URL (no outgoing links)")
    print("#" * 60)

    edges: dict[str, list[str]] = {"http://lonely.com": []}
    parser = HtmlParser(edges, delay=0.05)
    start_url = "http://lonely.com"

    result_mt, _ = run_and_time("Multithreaded", crawl, start_url, parser)
    if result_mt is not None:
        assert set(result_mt) == {"http://lonely.com"}, f"Wrong: {result_mt}"
        print("  Correctness: PASS")
    else:
        print("  crawl() returned None -- not yet implemented")


def test_all_different_hostnames():
    """Edge case: all linked URLs have different hostnames."""
    print("\n" + "#" * 60)
    print("TEST: All different hostnames")
    print("#" * 60)

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


if __name__ == "__main__":
    test_basic()
    test_single_url()
    test_all_different_hostnames()
    test_large_graph()

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
