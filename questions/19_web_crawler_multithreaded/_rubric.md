# Rubric: Web Crawler Multithreaded (Question 19)

## Grading Criteria

### 1. Correct Hostname Extraction and Filtering (10%)

- Correctly extracts hostname from URLs (strip `http://`, take text before first `/`)
- Filters out URLs with different hostnames
- Handles URLs with and without trailing paths

**Full marks:** Correct hostname logic, properly filters every discovered URL.
**Partial:** Minor edge case issues (e.g., doesn't handle URLs without path).
**Minimal:** Incorrect hostname extraction or no filtering.

---

### 2. Thread-Safe Visited Set (20%)

- Uses a `Lock` to protect the shared visited set, OR
- Uses a thread-safe data structure, OR
- Uses a single-threaded coordinator pattern that avoids races

**Full marks:** No race conditions possible. Only one thread can check-and-add to visited at a time.
**Partial:** Uses a lock but has a TOCTOU gap (check `if url not in visited` and `visited.add(url)` not atomic).
**Minimal:** No synchronization at all -- will produce duplicate visits or crashes.

Common correct pattern:
```python
with lock:
    if url in visited:
        return  # or continue
    visited.add(url)
# Now safe to crawl url
```

---

### 3. Proper Use of ThreadPoolExecutor or Threading (20%)

- Uses `ThreadPoolExecutor` with reasonable max_workers, OR
- Uses `threading.Thread` with proper lifecycle management
- Does not create an unbounded number of threads

**Full marks:** Clean use of `ThreadPoolExecutor` or well-managed thread pool. Reasonable worker count.
**Partial:** Works but creates too many threads, or uses threads without a pool.
**Minimal:** No multithreading, or threading code that doesn't actually parallelize.

---

### 4. Correct BFS/DFS Traversal in Concurrent Context (20%)

- All reachable same-hostname URLs are discovered
- The crawler terminates correctly (doesn't hang, doesn't exit early)
- Handles the "wave" pattern: discover URLs -> submit new tasks -> wait for completion

**Full marks:** Correctly discovers all URLs, terminates cleanly.
**Partial:** Discovers most URLs but may miss some due to early termination, or hangs occasionally.
**Minimal:** Doesn't traverse properly, or deadlocks.

Key termination patterns:
1. **Futures-based:** Submit initial URL, collect futures, submit new URLs from results, repeat until no new futures.
2. **Queue-based:** Use a counter/condition variable. Increment when submitting work, decrement when completing. Done when counter reaches 0.
3. **Recursive futures:** Each task returns new URLs, which spawn new tasks.

---

### 5. Significant Speedup Over Single-Threaded (15%)

- On the 50-node test, should achieve at least 3-5x speedup
- Ideally 10-20x+ speedup (50 nodes with 0.1s delay = 5s serial, should be <0.5s with enough parallelism)

**Full marks:** >5x speedup on the large test.
**Partial:** 2-5x speedup (some parallelism but not enough).
**Minimal:** <2x speedup or slower than single-threaded.

---

### 6. Clean Code, No Deadlocks, Proper Shutdown (15%)

- ThreadPoolExecutor used with `with` statement or properly shut down
- No potential deadlocks
- No resource leaks (threads properly joined)
- Clear, readable code

**Full marks:** Clean, production-quality code with proper resource management.
**Partial:** Works but messy, potential resource leaks.
**Minimal:** Deadlocks possible, resources not cleaned up.

---

## Common Approaches

### Approach 1: ThreadPoolExecutor with Futures (Recommended)

```python
def crawl(start_url, parser):
    hostname = get_hostname(start_url)
    visited = set([start_url])
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(parser.get_urls, start_url)}
        while futures:
            done = set()
            for future in list(futures):
                if future.done():
                    done.add(future)
            if not done:
                time.sleep(0.001)
                continue
            futures -= done
            for future in done:
                for url in future.result():
                    with lock:
                        if url not in visited and get_hostname(url) == hostname:
                            visited.add(url)
                            futures.add(executor.submit(parser.get_urls, url))
    return list(visited)
```

### Approach 2: ThreadPoolExecutor with as_completed (Cleaner)

```python
def crawl(start_url, parser):
    hostname = get_hostname(start_url)
    visited = {start_url}

    with ThreadPoolExecutor(max_workers=16) as executor:
        # Map future -> url so we can track
        futures = [executor.submit(parser.get_urls, start_url)]
        while futures:
            next_futures = []
            for future in as_completed(futures):
                for url in future.result():
                    if url not in visited and get_hostname(url) == hostname:
                        visited.add(url)
                        next_futures.append(executor.submit(parser.get_urls, url))
            futures = next_futures
    return list(visited)
```

Note: This approach is safe without a lock because `visited` is only accessed from the main thread (the `as_completed` loop runs in the main thread). The key insight is that submission of new futures and checking visited both happen in the main thread.

### Approach 3: Shared Queue with Worker Threads

```python
def crawl(start_url, parser):
    hostname = get_hostname(start_url)
    visited = set([start_url])
    lock = threading.Lock()
    queue = Queue()
    queue.put(start_url)
    active = threading.Semaphore(0)
    active.release()  # 1 active task

    def worker():
        while True:
            url = queue.get()
            if url is None:
                break
            for next_url in parser.get_urls(url):
                with lock:
                    if next_url not in visited and get_hostname(next_url) == hostname:
                        visited.add(next_url)
                        active.release()
                        queue.put(next_url)
            # Signal this task is done
            # ... (termination detection is the hard part)

    # Termination is tricky with this approach
```

---

## Red Flags

- No locking on shared visited set (race condition)
- Creating one thread per URL (resource exhaustion for large graphs)
- Busy-waiting without sleep (CPU waste)
- Not handling exceptions from get_urls (one error kills everything)
- Deadlock from incorrect lock usage
- Using `threading.Lock` recursively without `RLock`

## Green Flags

- Uses `with` statement for ThreadPoolExecutor
- Reasonable thread pool size (8-32 workers)
- Clean termination logic
- Mentions that GIL doesn't hurt here because work is I/O-bound
- Considers error handling
