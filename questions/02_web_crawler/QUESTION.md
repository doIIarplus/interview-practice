# Question 02: Web Crawler

## Overview

Implement a web crawler that discovers all pages on a single domain starting from a seed URL.

---

## Setup

You are given a helper function with the following signature:

```python
def get_links(url: str) -> list[str]:
    """Fetch a URL and return all links found on that page.

    Args:
        url: The URL to fetch.

    Returns:
        A list of absolute URLs found on the page.
        Returns an empty list if the page cannot be fetched.
    """
    ...
```

You do **not** need to implement `get_links` -- it is provided for you. Assume it handles HTTP requests, HTML parsing, and converting relative URLs to absolute URLs. It may return URLs from any domain.

---

## Task

Implement the following function:

```python
def crawl(seed_url: str) -> set[str]:
    """Crawl all pages on the same domain as seed_url.

    Args:
        seed_url: The starting URL to crawl from.

    Returns:
        A set of all discovered URLs on the same domain.
    """
```

### Requirements

1. **Start from the seed URL** and use BFS or DFS to discover new pages by calling `get_links()` on each visited page.
2. **Do not visit the same URL twice.** Track visited URLs and skip duplicates.
3. **Strip URL fragments.** Treat `https://example.com/page#section1` and `https://example.com/page` as the same URL. Remove the fragment before processing.
4. **Stay on the same domain.** Only crawl URLs whose domain (hostname) matches the seed URL's domain. For example, if the seed is `https://example.com/`, do not crawl `https://other.com/page`.
5. **Return all discovered URLs** on the same domain, including the seed URL itself.

### Example

Given the following link structure:

```
https://example.com/
  -> https://example.com/about
  -> https://example.com/blog
  -> https://other.com/external

https://example.com/about
  -> https://example.com/
  -> https://example.com/about#team

https://example.com/blog
  -> https://example.com/blog/post1
  -> https://example.com/blog/post2

https://example.com/blog/post1
  -> https://example.com/blog

https://example.com/blog/post2
  -> https://example.com/
```

Calling `crawl("https://example.com/")` should return:

```python
{
    "https://example.com/",
    "https://example.com/about",
    "https://example.com/blog",
    "https://example.com/blog/post1",
    "https://example.com/blog/post2",
}
```

Note:
- `https://other.com/external` is excluded (different domain).
- `https://example.com/about#team` is treated as `https://example.com/about` (fragment stripped) and is not visited again.

---

## Hints

- The `urllib.parse` module has useful functions: `urlparse`, `urldefrag`, `urljoin`.
- Think about what data structure is best for tracking visited URLs.
- Consider edge cases: What if `get_links()` returns a URL you have already visited? What if it returns the seed URL?

---

## Constraints

- All URLs are valid, absolute HTTP/HTTPS URLs.
- `get_links()` always returns absolute URLs.
- The number of pages on the domain is finite (no infinite loops if you handle visited tracking correctly).
- `get_links()` may be slow (network I/O) -- this matters for the follow-up discussion.
