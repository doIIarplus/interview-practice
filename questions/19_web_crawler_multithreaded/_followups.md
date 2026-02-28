# Follow-Up Questions: Web Crawler Multithreaded (Question 19)

## 1. How would you limit the number of concurrent requests per hostname (politeness)?

**Expected discussion:**
- Use a `Semaphore` to limit concurrent requests to a given hostname.
- In production crawlers, this is called a "politeness policy" -- you don't want to overwhelm a single server.
- Could use a per-hostname rate limiter (token bucket or leaky bucket algorithm).
- Alternatively, limit the ThreadPoolExecutor's `max_workers` to control overall concurrency.
- More sophisticated: use a priority queue with per-domain delays between requests.

---

## 2. What happens if `get_urls` raises an exception? How would you handle errors?

**Expected discussion:**
- Without handling, a single exception in a thread can crash the crawler or cause it to hang.
- With `ThreadPoolExecutor`, calling `future.result()` re-raises the exception -- need try/except.
- Options:
  - **Skip and log:** Catch the exception, log it, continue crawling other URLs.
  - **Retry with backoff:** Retry failed URLs a few times with exponential backoff.
  - **Circuit breaker:** If too many failures, stop crawling that hostname.
- In production: distinguish between transient errors (timeout, 503) and permanent errors (404, invalid URL).

---

## 3. How does Python's GIL affect this? Why does threading still help here?

**Expected discussion:**
- The GIL (Global Interpreter Lock) prevents multiple threads from executing Python bytecode simultaneously.
- **However**, the GIL is released during I/O operations (network calls, file reads, `time.sleep`).
- Since `get_urls` is I/O-bound (simulated with `time.sleep`), threads can truly run in parallel -- while one thread is waiting for I/O, another thread can proceed.
- If `get_urls` were CPU-bound (e.g., parsing HTML with pure Python), threading would NOT help -- you'd need `multiprocessing` instead.
- This is the classic use case where threading shines in Python: I/O-bound workloads.

---

## 4. How would you implement this with `asyncio` instead of threads? What are the trade-offs?

**Expected discussion:**
- Replace `get_urls` with an async version: `async def get_urls(url) -> list[str]`
- Use `asyncio.gather()` or `asyncio.create_task()` to run multiple requests concurrently.
- Use an `asyncio.Queue` for BFS.

```python
async def crawl(start_url, parser):
    hostname = get_hostname(start_url)
    visited = {start_url}
    queue = asyncio.Queue()
    await queue.put(start_url)

    async def worker():
        while True:
            url = await queue.get()
            urls = await parser.async_get_urls(url)
            for next_url in urls:
                if next_url not in visited and get_hostname(next_url) == hostname:
                    visited.add(next_url)
                    await queue.put(next_url)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(16)]
    await queue.join()
    for w in workers:
        w.cancel()
    return list(visited)
```

**Trade-offs:**
- **asyncio advantages:** Lower overhead per "thread" (coroutines are lightweight), no GIL concerns, no locks needed (single-threaded event loop).
- **asyncio disadvantages:** Requires async/await throughout the call chain ("function coloring" problem), harder to debug, requires async-compatible libraries (e.g., `aiohttp` instead of `requests`).
- **Threading advantages:** Works with existing synchronous libraries, simpler mental model for some developers.
- **Threading disadvantages:** Higher memory overhead per thread (~8MB stack), potential for race conditions, GIL limits CPU-bound work.

---

## 5. What if the graph has millions of URLs? How would you limit memory?

**Expected discussion:**
- The visited set grows linearly with URLs discovered. For millions of URLs, this could use significant memory.
- **Bloom filter:** Use a probabilistic data structure that trades a small false-positive rate for much less memory. A Bloom filter with 1% FPR uses ~10 bits per element vs ~50+ bytes for a hash set entry.
- **Disk-backed set:** Store visited URLs on disk (e.g., SQLite, LevelDB) instead of in memory.
- **URL compression:** Store hashes of URLs instead of full strings.
- **Frontier management:** Don't keep all discovered URLs in memory -- use a disk-backed priority queue.
- **Crawl budget:** Set a maximum number of URLs to crawl and stop when reached.

---

## 6. How would you add a maximum crawl depth?

**Expected discussion:**
- Track the depth of each URL (distance from start_url in the BFS).
- Store `(url, depth)` pairs instead of just URLs.
- When discovering new URLs, their depth is `current_depth + 1`.
- Skip URLs that exceed the maximum depth.
- This requires passing depth information along with URLs to worker threads.

```python
# In the queue/futures, track depth:
queue.put((start_url, 0))
# When processing:
if depth < max_depth:
    for next_url in parser.get_urls(url):
        queue.put((next_url, depth + 1))
```

---

## 7. How would you distribute this across multiple machines?

**Expected discussion:**
- **URL partitioning:** Assign URLs to machines based on hostname hash (consistent hashing). Each machine is responsible for a set of hostnames.
- **Message queue:** Use a distributed message queue (Kafka, RabbitMQ, SQS) for URL frontier. Machines pull URLs from the queue, crawl them, and push discovered URLs back.
- **Distributed visited set:** Use a distributed key-value store (Redis, Memcached) or a distributed Bloom filter to track visited URLs.
- **Coordination:** Need a coordinator to manage the overall crawl state, detect termination, and handle machine failures.
- **Challenges:** Network partitions, duplicate work, consistent termination detection, politeness across machines.
- **Real-world examples:** Google's original MapReduce paper, Apache Nutch, Scrapy with distributed backends.
