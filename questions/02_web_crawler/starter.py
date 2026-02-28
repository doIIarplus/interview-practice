"""
Web Crawler

Implement a web crawler that discovers all pages on a single domain.
See QUESTION.md for full problem description.
"""

from urllib.parse import urlparse, urldefrag


def get_links(url: str) -> list[str]:
    """Fetch a URL and return all links found on that page.

    THIS IS A STUB. In a real interview environment, this function
    would be provided and would make actual HTTP requests.

    For local testing, you can replace this with a mock implementation.

    Args:
        url: The URL to fetch.

    Returns:
        A list of absolute URLs found on the page.
        Returns an empty list if the page cannot be fetched.
    """
    # Mock link graph for testing
    link_graph: dict[str, list[str]] = {
        "https://example.com/": [
            "https://example.com/about",
            "https://example.com/blog",
            "https://other.com/external",
        ],
        "https://example.com/about": [
            "https://example.com/",
            "https://example.com/about#team",
        ],
        "https://example.com/blog": [
            "https://example.com/blog/post1",
            "https://example.com/blog/post2",
        ],
        "https://example.com/blog/post1": [
            "https://example.com/blog",
        ],
        "https://example.com/blog/post2": [
            "https://example.com/",
        ],
    }
    return link_graph.get(url, [])


def crawl(seed_url: str) -> set[str]:
    """Crawl all pages on the same domain as seed_url.

    Starting from the seed URL, discover all pages on the same domain
    by following links. Strip URL fragments, avoid revisiting pages,
    and only follow links that share the same domain as the seed.

    Args:
        seed_url: The starting URL to crawl from.

    Returns:
        A set of all discovered URLs on the same domain.
    """
    pass


# =============================================================================
# Quick Smoke Test
# =============================================================================
if __name__ == "__main__":
    result = crawl("https://example.com/")
    expected = {
        "https://example.com/",
        "https://example.com/about",
        "https://example.com/blog",
        "https://example.com/blog/post1",
        "https://example.com/blog/post2",
    }
    print(f"Crawled URLs: {result}")
    print(f"Expected:     {expected}")
    print(f"Match: {result == expected}")
