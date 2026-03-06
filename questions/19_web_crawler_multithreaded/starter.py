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


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    edges = {
        "http://example.com/": ["http://example.com/about", "http://other.com/"],
        "http://example.com/about": ["http://example.com/"],
    }
    parser = HtmlParser(edges, delay=0.1)
    result = crawl("http://example.com/", parser)
    print(f"Crawled: {result}")
