# Follow-Up Questions: Web Crawler

> **This file is hidden from the candidate.**

## Follow-Up 1: Making It Concurrent

**Question:** This is currently synchronous and slow. How would you make it concurrent?

**What to look for:**
- Identifies that `get_links()` is I/O-bound, so concurrency helps despite the GIL
- Suggests `concurrent.futures.ThreadPoolExecutor` or `asyncio` with `aiohttp`
- Understands that the visited set needs thread-safe access (lock, or use the thread pool's future results to update serially)
- Discusses the tradeoff: too many concurrent requests can overwhelm the server or get rate-limited

**Red flags:**
- Suggests multiprocessing for an I/O-bound problem (overkill, IPC overhead)
- No awareness of thread safety for the shared visited set
- Cannot articulate why threading helps for I/O despite the GIL

---

## Follow-Up 2: ThreadPoolExecutor Implementation

**Question:** Implement a version using `concurrent.futures.ThreadPoolExecutor`. How many workers would you use and why?

**What to look for:**
- Working implementation that submits `get_links` calls to the pool
- Collects results and feeds newly discovered URLs back into the pool
- Uses a lock or serial aggregation for the visited set
- Reasonable worker count discussion: 10-50 for network I/O, depends on server capacity and network latency
- Mentions that the optimal number depends on the target server's capacity and the network round-trip time

**Sample structure they should arrive at:**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def crawl_concurrent(seed_url: str, max_workers: int = 20) -> set[str]:
    seed_url, _ = urldefrag(seed_url)
    seed_domain = urlparse(seed_url).netloc
    visited = {seed_url}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_links, seed_url): seed_url}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                url = futures.pop(future)
                for link in future.result():
                    normalized, _ = urldefrag(link)
                    if normalized not in visited and urlparse(normalized).netloc == seed_domain:
                        visited.add(normalized)
                        futures[executor.submit(get_links, normalized)] = normalized
    return visited
```

**Red flags:**
- Submits all URLs at once without limiting concurrency
- No mechanism to wait for results and process them
- Race conditions on the visited set with no mitigation

---

## Follow-Up 3: Politeness Policies

**Question:** How would you implement politeness policies (rate limiting per domain)?

**What to look for:**
- Mentions `robots.txt` parsing and respecting `Crawl-delay`
- Suggests per-domain rate limiting (e.g., max N requests per second per domain)
- Implementation ideas: token bucket, leaky bucket, or simple `time.sleep()` between requests to same domain
- Mentions `User-Agent` header setting
- Discusses that in a single-domain crawler this is simpler (global rate limit), but for multi-domain it requires per-domain tracking

**Bonus:**
- Mentions exponential backoff for failed requests
- Discusses respecting `robots.txt` `Disallow` directives
- Mentions `Retry-After` HTTP header

---

## Follow-Up 4: Distributed Crawler

**Question:** What if you needed to crawl billions of pages across many domains? How would you design a distributed crawler?

**What to look for:**
- URL frontier: a distributed queue of URLs to crawl, partitioned by domain for politeness
- Workers: multiple crawler processes/machines pulling from the frontier
- Deduplication: a distributed set (e.g., Bloom filter, Redis set, or a distributed hash table) for visited URLs
- Storage: crawled content stored in a distributed file system (S3, HDFS) or database
- Coordination: discusses how to partition work (by domain hash, consistent hashing)
- Mentions real systems: Scrapy, Apache Nutch, or custom architectures

**Bonus:**
- Discusses URL prioritization (important pages first)
- Mentions DNS resolution caching
- Discusses content deduplication (same content at different URLs)
- Mentions incremental re-crawling strategies

**Red flags:**
- No concept of distributed systems challenges (network partitions, worker failures)
- Suggests a single machine with more threads
- No awareness of the URL frontier concept

---

## Follow-Up 5: GIL and asyncio

**Question:** How does Python's GIL affect your threading approach here? Would asyncio be better? Why or why not?

**What to look for:**
- Understands that the GIL prevents true parallel CPU execution but releases during I/O
- For I/O-bound work like web crawling, threading works well because the GIL is released during network calls
- `asyncio` avoids thread overhead and context-switching costs; better for very high concurrency (thousands of concurrent requests)
- `asyncio` requires an async HTTP library (e.g., `aiohttp`), not `requests`
- Threading is simpler to implement and debug; asyncio has a steeper learning curve and "infects" the codebase (async/await everywhere)
- For moderate concurrency (10-100 workers), threading and asyncio perform similarly
- For very high concurrency (1000+ concurrent connections), asyncio is more efficient (lower memory per connection)

**Red flags:**
- Believes the GIL makes threading completely useless
- Cannot explain when asyncio would be preferred over threading
- Confuses asyncio with multiprocessing
