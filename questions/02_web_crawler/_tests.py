"""Hidden tests for Question 02: Web Crawler
Run: python questions/02_web_crawler/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import crawl, get_links


def test_basic_crawl():
    """Test basic crawling of a simple link graph."""
    result = crawl("https://example.com/")
    expected = {
        "https://example.com/",
        "https://example.com/about",
        "https://example.com/blog",
        "https://example.com/blog/post1",
        "https://example.com/blog/post2",
    }
    assert result == expected, f"Expected {expected}, got {result}"
    print("[PASS] test_basic_crawl")


def test_empty_seed():
    """Test crawling from a URL with no outgoing links."""
    result = crawl("https://example.com/nonexistent")
    assert result == {"https://example.com/nonexistent"}, f"Expected singleton set, got {result}"
    print("[PASS] test_empty_seed")


def run_tests():
    print("=" * 60)
    print("Web Crawler — Hidden Tests")
    print("=" * 60 + "\n")

    test_basic_crawl()
    test_empty_seed()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
