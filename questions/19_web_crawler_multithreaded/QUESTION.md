# Question 19: Web Crawler Multithreaded

*(Based on LeetCode 1242)*

## Problem

Given a URL `start_url` and an interface `HtmlParser`, implement a **multithreaded** web crawler that crawls all links under the same hostname as `start_url`.

The `HtmlParser` interface has one method:

```python
class HtmlParser:
    def get_urls(self, url: str) -> list[str]:
        """Returns all URLs found on the page at the given URL.
        This method simulates a network call and takes ~100ms to complete."""
```

Write a function:

```python
def crawl(start_url: str, parser: HtmlParser) -> list[str]:
```

that returns all URLs reachable from `start_url` that share the same hostname.

---

## Requirements

- Return all URLs under the **same hostname** as `start_url`.
- **Do not crawl the same URL twice.**
- The `get_urls` call is **I/O-bound** (simulated network call) -- a single-threaded solution will be too slow.
- You **must use multithreading** to achieve acceptable performance.
- The solution **must be thread-safe** (no duplicate visits, no race conditions).

---

## Hostname Definition

The hostname is defined as everything between `http://` and the first `/` (or end of string if there is no `/` after the protocol).

| URL | Hostname |
|-----|----------|
| `http://news.yahoo.com/news/topics/` | `news.yahoo.com` |
| `http://news.yahoo.com/news` | `news.yahoo.com` |
| `http://news.google.com` | `news.google.com` |

---

## Constraints

- Up to **1000 URLs** in the graph.
- Each `get_urls` call takes **~100ms** (simulated network latency).
- A single-threaded solution would take ~100 seconds for 1000 URLs; your solution should complete in **~5-10 seconds** with adequate parallelism.

---

## Example

```
URL graph (page -> links found on that page):
  http://news.yahoo.com/news        -> [http://news.yahoo.com/news/topics/, http://news.google.com]
  http://news.yahoo.com/news/topics/ -> [http://news.yahoo.com/news, http://news.yahoo.com]
  http://news.google.com             -> [http://news.yahoo.com, http://news.yahoo.com/news]
  http://news.yahoo.com              -> []

start_url = "http://news.yahoo.com/news"

Result: ["http://news.yahoo.com/news",
         "http://news.yahoo.com/news/topics/",
         "http://news.yahoo.com"]
```

Note: `http://news.google.com` is **not** included because it has a different hostname (`news.google.com` vs `news.yahoo.com`).

---

## Starter Code

See `starter.py` for:
- A `HtmlParser` simulator with configurable delay
- A reference **single-threaded** solution (to see how slow it is)
- A `crawl()` function stub for you to implement
- Test cases with timing comparisons

Run the starter code to see the single-threaded performance, then implement the multithreaded version and compare.

---

## Hints

- Consider `concurrent.futures.ThreadPoolExecutor` for managing threads.
- Think about how to safely share the "visited" set across threads.
- BFS naturally maps to a producer-consumer pattern with threads.
- You need to know when all work is done -- think about how to detect termination.
