# Rubric: Web Crawler

> **This file is hidden from the candidate.**

## Scoring Breakdown

### Correct BFS/DFS Implementation (25%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Uses a queue (BFS) or stack (DFS) | 10 | Clear traversal structure; `collections.deque` for BFS is ideal |
| Visits each URL exactly once | 10 | Uses a `set` for visited tracking; checks before enqueuing/visiting |
| Returns complete set of discovered URLs | 5 | All reachable same-domain pages are found |

**Green flags:**
- Uses `collections.deque` with `popleft()` for BFS
- Adds to visited set *before* or *when* enqueuing (not after dequeueing) to avoid duplicates in queue
- Clean loop structure: dequeue, get links, filter, enqueue

**Red flags:**
- Uses a list as a queue with `pop(0)` (O(n) per operation)
- Recursion without depth limit for DFS (stack overflow risk)
- Checks visited only after dequeueing (allows duplicates in queue, wasting memory)

### URL Normalization (25%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Fragment stripping | 10 | Uses `urldefrag()` or manual splitting on `#` |
| Domain comparison | 10 | Uses `urlparse().netloc` or `.hostname` to compare domains |
| Normalizes before any comparison | 5 | Strips fragment before checking visited set and domain |

**Green flags:**
- Uses `urllib.parse.urldefrag()` for fragment removal
- Uses `urlparse(url).netloc` for domain extraction
- Normalizes URLs in a single helper function

**Red flags:**
- String matching for domain (e.g., `url.startswith("https://example.com")`) -- breaks for `example.com.evil.com`
- Forgets to normalize the seed URL itself
- Strips fragments but not from the seed URL

### Code Quality (15%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Readable structure | 5 | Clear variable names, logical flow |
| Use of standard library | 5 | Leverages `urllib.parse`, `collections.deque` |
| No unnecessary complexity | 5 | Solution is not over-engineered for the synchronous version |

### Edge Case Handling (20%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Circular links | 5 | Handles pages that link back to each other |
| Self-links | 5 | Handles pages that link to themselves |
| URLs with query parameters | 5 | `page?a=1` and `page?a=2` are treated as different URLs |
| Fragment-only differences | 5 | `page#a` and `page#b` are treated as the same URL |

**Additional edge cases to probe:**
- What if `get_links()` returns an empty list?
- What if the seed URL is not fetchable?
- What about trailing slashes: is `example.com/about` the same as `example.com/about/`?

### Concurrency Follow-Up (15%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Identifies I/O as the bottleneck | 5 | Recognizes that `get_links()` is the slow part |
| Proposes threading or asyncio | 5 | Concrete suggestion with awareness of tradeoffs |
| Thread-safe visited set | 5 | Understands need for synchronization when using threads |

---

## Overall Assessment

| Rating | Description |
|--------|-------------|
| **Strong hire** | Clean BFS with proper URL normalization, handles all edge cases, articulate about concurrency tradeoffs |
| **Hire** | Working solution with fragment stripping and domain filtering, minor issues with edge cases |
| **Borderline** | Basic BFS works but misses fragment stripping or domain filtering |
| **No hire** | Cannot implement basic BFS traversal, or solution has infinite loops |

## Time Expectations

- Basic implementation: 10-15 minutes
- With all edge cases: 15-20 minutes
- Concurrency follow-up discussion: 10-15 minutes
- Total: 25-35 minutes

## Sample Solution Sketch

```python
from collections import deque
from urllib.parse import urlparse, urldefrag

def crawl(seed_url: str) -> set[str]:
    seed_url, _ = urldefrag(seed_url)
    seed_domain = urlparse(seed_url).netloc

    visited: set[str] = set()
    queue: deque[str] = deque([seed_url])
    visited.add(seed_url)

    while queue:
        url = queue.popleft()
        for link in get_links(url):
            normalized, _ = urldefrag(link)
            if normalized not in visited and urlparse(normalized).netloc == seed_domain:
                visited.add(normalized)
                queue.append(normalized)

    return visited
```
