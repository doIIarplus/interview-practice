# Python Concurrency — From Fundamentals to Interview Ready

This guide covers everything you need to know about Python concurrency for the
Anthropic Performance Engineer interview. We start from the basics and build up
to patterns you will use directly in the interview questions.

---

## Table of Contents

1. [The GIL Explained Simply](#the-gil-explained-simply)
2. [Three Concurrency Models](#three-concurrency-models)
3. [Threading: The Basics](#threading-the-basics)
4. [ThreadPoolExecutor](#threadpoolexecutor)
5. [Synchronization Primitives](#synchronization-primitives)
6. [Common Pitfalls](#common-pitfalls)
7. [Asyncio Basics](#asyncio-basics)
8. [Multiprocessing](#multiprocessing)
9. [Worked Example: Concurrent URL Fetcher](#worked-example-concurrent-url-fetcher)
10. [Interview Patterns](#interview-patterns)

---

## The GIL Explained Simply

The **Global Interpreter Lock (GIL)** is a mutex in CPython that allows only one
thread to execute Python bytecode at a time. This means:

```
Thread 1:  [===RUN===]...........[===RUN===]...........
Thread 2:  ...........[===RUN===]...........[===RUN===]
                    ^                     ^
              GIL switches          GIL switches
```

### What the GIL means in practice

- **CPU-bound work:** Threading does NOT speed things up. Two threads doing math
  will take the same wall-clock time as one thread (or longer, due to context
  switching overhead).
- **I/O-bound work:** Threading DOES help. When a thread is waiting for I/O (network,
  disk, sleep), it releases the GIL, letting other threads run.

```
Thread 1:  [compute][---waiting for network---][compute]
Thread 2:  ..........[compute][---waiting for network---][compute]
                     ^
              GIL is free while Thread 1 waits for I/O,
              so Thread 2 can run
```

### Key insight for interviews

When the interviewer asks "how would you make this concurrent?" your first
question should be: **is the bottleneck CPU or I/O?**

- I/O-bound (web requests, file reads, database queries) -> `threading` or `asyncio`
- CPU-bound (number crunching, compression, hashing) -> `multiprocessing`

---

## Three Concurrency Models

| Model | Best For | GIL Impact | Overhead |
|-------|----------|------------|----------|
| `threading` | I/O-bound tasks, shared state | Limited by GIL for CPU work | Low (shared memory) |
| `asyncio` | Many I/O-bound tasks (thousands) | Single-threaded, no GIL issue | Very low (coroutines) |
| `multiprocessing` | CPU-bound tasks | Each process has its own GIL | High (separate memory) |

### Decision flowchart

```
Is the task CPU-bound or I/O-bound?
├── CPU-bound
│   └── Use multiprocessing (or C extension that releases GIL)
└── I/O-bound
    ├── Few tasks (< 100) with shared state?
    │   └── Use threading
    └── Many tasks (hundreds/thousands)?
        └── Use asyncio
```

---

## Threading: The Basics

### Creating and starting threads

```python
import threading
import time

def worker(name, delay):
    """Simulate some I/O-bound work."""
    print(f"[{name}] Starting")
    time.sleep(delay)  # Simulates I/O — releases the GIL
    print(f"[{name}] Done after {delay}s")

# Create threads
t1 = threading.Thread(target=worker, args=("A", 2))
t2 = threading.Thread(target=worker, args=("B", 1))

# Start them
t1.start()
t2.start()

# Wait for both to finish
t1.join()
t2.join()

print("All done!")
# Total time: ~2 seconds (not 3), because they ran concurrently
```

### daemon threads

A daemon thread is killed when the main thread exits. Use for background tasks
you do not need to wait for.

```python
t = threading.Thread(target=worker, args=("bg", 10), daemon=True)
t.start()
# If main thread exits here, the daemon thread is killed immediately
```

---

## ThreadPoolExecutor

This is the **most common pattern** in interview code. It manages a pool of
threads and provides a clean API.

### Basic usage

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def fetch_url(url):
    """Simulate fetching a URL."""
    time.sleep(1)  # Simulate network I/O
    return f"Content of {url}"

urls = [f"https://example.com/page/{i}" for i in range(10)]

# Method 1: map() — preserves order, blocks until all done
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch_url, urls))
    for r in results:
        print(r)

# Method 2: submit() + as_completed() — results as they finish
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(fetch_url, url): url for url in urls}
    for future in as_completed(futures):
        url = futures[future]
        try:
            result = future.result()
            print(f"{url} -> {result}")
        except Exception as e:
            print(f"{url} raised {e}")
```

### When to use which method

- `executor.map(fn, iterable)` — simple, order-preserving. Good when you want
  all results in order.
- `executor.submit(fn, *args)` + `as_completed()` — more flexible. Good when
  you want to process results as they arrive or need individual error handling.

### Choosing `max_workers`

- For I/O-bound: 10-100 workers is common. Each thread mostly sleeps.
- For CPU-bound: use `multiprocessing` instead, but if you must thread, use
  `os.cpu_count()` workers.
- Default (Python 3.8+): `min(32, os.cpu_count() + 4)`

---

## Synchronization Primitives

When threads share mutable state, you need synchronization to avoid race
conditions.

### Lock — Mutual Exclusion

A `Lock` ensures only one thread accesses a critical section at a time.

```python
import threading

counter = 0
lock = threading.Lock()

def increment(n):
    global counter
    for _ in range(n):
        with lock:           # Acquire lock, auto-releases when block exits
            counter += 1     # Only one thread at a time here

threads = [threading.Thread(target=increment, args=(100000,)) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter: {counter}")  # Always 400000 with lock; may be less without
```

**Without the lock**, `counter += 1` is not atomic in Python. It involves:
1. Read `counter` (LOAD)
2. Add 1 (ADD)
3. Write back (STORE)

Two threads can read the same value and both write back value+1, losing an
increment.

### RLock — Reentrant Lock

An `RLock` can be acquired multiple times by the **same** thread. Useful when a
locked function calls another locked function.

```python
rlock = threading.RLock()

def outer():
    with rlock:
        inner()  # This would deadlock with a regular Lock!

def inner():
    with rlock:
        print("Inner acquired the lock")
```

### Event — One-Time Signal

An `Event` lets one thread signal others that something has happened.

```python
import threading
import time

ready = threading.Event()

def producer():
    print("Producer: preparing data...")
    time.sleep(2)
    print("Producer: data ready!")
    ready.set()  # Signal all waiting threads

def consumer(name):
    print(f"Consumer {name}: waiting for data...")
    ready.wait()  # Blocks until ready.set() is called
    print(f"Consumer {name}: processing data!")

threads = [
    threading.Thread(target=producer),
    threading.Thread(target=consumer, args=("A",)),
    threading.Thread(target=consumer, args=("B",)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Condition — Wait/Notify

A `Condition` lets threads wait for a specific condition to become true. This is
the classic producer-consumer pattern.

```python
import threading
import time
import random

buffer = []
MAX_SIZE = 5
condition = threading.Condition()

def producer():
    for i in range(20):
        with condition:
            while len(buffer) >= MAX_SIZE:
                condition.wait()  # Release lock and wait
            buffer.append(i)
            print(f"Produced {i}, buffer size: {len(buffer)}")
            condition.notify_all()  # Wake up consumers
        time.sleep(random.uniform(0, 0.1))

def consumer(name):
    consumed = 0
    while consumed < 10:
        with condition:
            while len(buffer) == 0:
                condition.wait()  # Release lock and wait
            item = buffer.pop(0)
            print(f"Consumer {name} consumed {item}, buffer size: {len(buffer)}")
            consumed += 1
            condition.notify_all()  # Wake up producer

threads = [
    threading.Thread(target=producer),
    threading.Thread(target=consumer, args=("A",)),
    threading.Thread(target=consumer, args=("B",)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Semaphore — Bounded Concurrency

A `Semaphore` limits the number of threads that can access a resource
simultaneously.

```python
import threading
import time

# Allow at most 3 concurrent database connections
db_semaphore = threading.Semaphore(3)

def query_database(query_id):
    with db_semaphore:
        print(f"Query {query_id}: connected (max 3 at a time)")
        time.sleep(1)  # Simulate query
        print(f"Query {query_id}: done")

threads = [threading.Thread(target=query_database, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## Common Pitfalls

### 1. Deadlock

**Deadlock** occurs when two threads each hold a lock the other needs.

```python
# BAD: Classic deadlock
lock_a = threading.Lock()
lock_b = threading.Lock()

def thread_1():
    with lock_a:
        time.sleep(0.1)  # Give thread_2 time to acquire lock_b
        with lock_b:      # DEADLOCK: thread_2 holds lock_b
            print("Thread 1")

def thread_2():
    with lock_b:
        time.sleep(0.1)  # Give thread_1 time to acquire lock_a
        with lock_a:      # DEADLOCK: thread_1 holds lock_a
            print("Thread 2")
```

**Fix:** Always acquire locks in the same order.

```python
# GOOD: Consistent lock ordering
def thread_1():
    with lock_a:       # Always acquire lock_a first
        with lock_b:
            print("Thread 1")

def thread_2():
    with lock_a:       # Always acquire lock_a first
        with lock_b:
            print("Thread 2")
```

### 2. Race Condition

**Race condition:** result depends on the timing of thread execution.

```python
# BAD: Check-then-act race condition
if key not in cache:         # Thread 1 checks
    # Thread 2 might insert key here!
    cache[key] = compute()   # Thread 1 inserts (may overwrite Thread 2's value)

# GOOD: Atomic check-and-insert with a lock
with lock:
    if key not in cache:
        cache[key] = compute()
```

### 3. Thread-Safe Data Structures

Python's `queue.Queue` is thread-safe. Python's `list` and `dict` are NOT fully
thread-safe for compound operations (though individual operations like
`list.append` happen to be atomic in CPython due to the GIL — do not rely on
this).

```python
import queue

# Thread-safe queue — use this for producer-consumer patterns
q = queue.Queue(maxsize=100)

def producer():
    for i in range(50):
        q.put(i)  # Blocks if full

def consumer():
    while True:
        item = q.get()  # Blocks if empty
        process(item)
        q.task_done()
```

---

## Asyncio Basics

`asyncio` is Python's single-threaded concurrency model. Instead of OS threads,
it uses **coroutines** that voluntarily yield control at `await` points.

### Core concepts

```python
import asyncio

async def fetch(url):
    """A coroutine — note the 'async def'."""
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulate network I/O; yields control
    return f"Content of {url}"

async def main():
    # Run multiple coroutines concurrently
    urls = [f"https://example.com/{i}" for i in range(5)]
    results = await asyncio.gather(*[fetch(url) for url in urls])
    for r in results:
        print(r)

# Run the event loop
asyncio.run(main())
# Total time: ~1 second (all 5 fetches run concurrently)
```

### asyncio vs threading

| Aspect | threading | asyncio |
|--------|-----------|---------|
| Concurrency model | OS threads, preemptive | Coroutines, cooperative |
| Context switching | OS decides when to switch | You decide (at `await`) |
| Shared state | Need locks | Usually no locks needed (single-threaded) |
| Number of tasks | Hundreds (thread overhead) | Thousands (coroutines are cheap) |
| Code style | Regular functions | `async def` / `await` |
| Libraries | Standard `requests`, etc. | Need async-compatible libs (`aiohttp`) |

### When to use asyncio in interviews

- When you need to manage **many** concurrent I/O operations
- When the problem naturally fits an event-driven model
- When you want to avoid the complexity of thread synchronization

---

## Multiprocessing

For CPU-bound work, use `multiprocessing` to bypass the GIL entirely.

```python
from multiprocessing import Pool
import math

def is_prime(n):
    """CPU-bound: check if n is prime."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

numbers = list(range(1_000_000, 1_001_000))

# Sequential
# results = [is_prime(n) for n in numbers]  # Slow on multi-core

# Parallel across CPU cores
with Pool(processes=4) as pool:
    results = pool.map(is_prime, numbers)

primes = [n for n, is_p in zip(numbers, results) if is_p]
print(f"Found {len(primes)} primes")
```

### multiprocessing caveats

- **Serialization overhead:** Arguments and results are pickled (serialized) and
  sent between processes. Large objects are slow to transfer.
- **No shared memory by default:** Each process has its own memory space. Use
  `multiprocessing.Value`, `multiprocessing.Array`, or `multiprocessing.Manager`
  for shared state.
- **Startup cost:** Creating processes is slower than creating threads.

---

## Worked Example: Concurrent URL Fetcher

This is the kind of thing you might build in Q02 (Web Crawler) or Q19
(Multithreaded Web Crawler).

```python
"""
Concurrent URL fetcher with:
- Thread pool for parallel downloads
- Rate limiting (max N concurrent requests)
- Timeout handling
- Error handling per URL
- Results collection
"""

import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Simulated network — replace with real `requests` in production
# ---------------------------------------------------------------------------

def simulate_fetch(url: str, timeout: float = 5.0) -> str:
    """Simulate fetching a URL with random latency and occasional failures."""
    latency = random.uniform(0.1, 2.0)
    if latency > timeout:
        raise TimeoutError(f"Timed out fetching {url}")
    time.sleep(latency)
    if random.random() < 0.1:  # 10% failure rate
        raise ConnectionError(f"Failed to fetch {url}")
    return f"<html>Content of {url} (fetched in {latency:.2f}s)</html>"

# ---------------------------------------------------------------------------
# Concurrent fetcher
# ---------------------------------------------------------------------------

class ConcurrentFetcher:
    def __init__(self, max_workers: int = 10, timeout: float = 5.0,
                 max_per_domain: int = 3):
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_per_domain = max_per_domain

        # Per-domain semaphores for rate limiting
        self._domain_semaphores: dict[str, threading.Semaphore] = defaultdict(
            lambda: threading.Semaphore(self.max_per_domain)
        )
        self._lock = threading.Lock()

    def _get_domain_semaphore(self, url: str) -> threading.Semaphore:
        domain = urlparse(url).netloc
        with self._lock:
            return self._domain_semaphores[domain]

    def _fetch_one(self, url: str) -> tuple[str, str | None, str | None]:
        """Fetch a single URL with per-domain rate limiting.

        Returns (url, content_or_None, error_or_None).
        """
        sem = self._get_domain_semaphore(url)
        with sem:  # At most max_per_domain concurrent requests per domain
            try:
                content = simulate_fetch(url, timeout=self.timeout)
                return (url, content, None)
            except Exception as e:
                return (url, None, str(e))

    def fetch_all(self, urls: list[str]) -> dict[str, str | None]:
        """Fetch all URLs concurrently. Returns {url: content_or_None}."""
        results = {}
        errors = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self._fetch_one, url): url
                for url in urls
            }
            for future in as_completed(future_to_url):
                url, content, error = future.result()
                if error:
                    errors[url] = error
                    print(f"  ERROR {url}: {error}")
                else:
                    results[url] = content
                    print(f"  OK    {url}")

        print(f"\nFetched {len(results)}/{len(urls)} URLs "
              f"({len(errors)} errors)")
        return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    urls = [f"https://example.com/page/{i}" for i in range(20)]
    urls += [f"https://other-site.org/article/{i}" for i in range(10)]

    fetcher = ConcurrentFetcher(max_workers=8, timeout=3.0, max_per_domain=3)

    start = time.time()
    results = fetcher.fetch_all(urls)
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Sequential would have taken: ~{len(urls) * 1.0:.0f}s")
```

### Key patterns to note in the example above

1. **ThreadPoolExecutor** manages the thread pool — no manual thread creation.
2. **Per-domain semaphore** limits concurrency to avoid overwhelming a single
   server — a real-world concern.
3. **`defaultdict` with Lock** for thread-safe lazy initialization.
4. **`as_completed`** processes results as they arrive rather than waiting for all.
5. **Tuple return** `(url, content, error)` avoids raising exceptions across
   thread boundaries.

---

## Interview Patterns

### Pattern 1: Producer-Consumer with Queue

Used in Q09 (Concurrent Task Scheduler).

```python
import queue
import threading

task_queue = queue.Queue()
results = {}
results_lock = threading.Lock()

def worker():
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill — shutdown signal
            break
        task_id, work_fn = task
        result = work_fn()
        with results_lock:
            results[task_id] = result
        task_queue.task_done()

# Start workers
workers = [threading.Thread(target=worker, daemon=True) for _ in range(4)]
for w in workers:
    w.start()

# Submit tasks
for i in range(20):
    task_queue.put((i, lambda i=i: i * i))

# Wait for all tasks to complete
task_queue.join()

# Shutdown workers
for _ in workers:
    task_queue.put(None)
```

### Pattern 2: Parallel BFS with Visited Set

Used in Q02 (Web Crawler) and Q19 (Multithreaded Web Crawler).

```python
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

visited = set()
visited_lock = threading.Lock()

def crawl_bfs(start_url, max_pages=100):
    """BFS web crawl with thread pool."""
    frontier = deque([start_url])

    with visited_lock:
        visited.add(start_url)

    with ThreadPoolExecutor(max_workers=8) as executor:
        while frontier and len(visited) < max_pages:
            # Submit a batch of URLs
            batch = []
            while frontier and len(batch) < 8:
                batch.append(frontier.popleft())

            futures = {executor.submit(fetch_and_parse, url): url
                       for url in batch}

            for future in as_completed(futures):
                url = futures[future]
                try:
                    child_urls = future.result()
                    with visited_lock:
                        for child in child_urls:
                            if child not in visited and len(visited) < max_pages:
                                visited.add(child)
                                frontier.append(child)
                except Exception as e:
                    print(f"Error crawling {url}: {e}")

    return visited
```

### Pattern 3: Barrier Synchronization

Used when all threads must reach a point before any can proceed (relevant to
Q15 collective communication).

```python
import threading

barrier = threading.Barrier(4)  # Wait for 4 threads

def phase_worker(worker_id):
    # Phase 1
    print(f"Worker {worker_id}: phase 1 complete")
    barrier.wait()  # All threads must reach here before any continues

    # Phase 2 — all threads guaranteed to have finished phase 1
    print(f"Worker {worker_id}: phase 2 complete")

threads = [threading.Thread(target=phase_worker, args=(i,)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## Quick Reference

| What you want | Use this |
|---------------|----------|
| Run N I/O tasks in parallel | `ThreadPoolExecutor(max_workers=N)` |
| Protect shared mutable state | `threading.Lock` |
| Limit concurrent access to resource | `threading.Semaphore(N)` |
| Signal that something happened | `threading.Event` |
| Producer-consumer queue | `queue.Queue` |
| Wait for all threads to reach a point | `threading.Barrier(N)` |
| Run N CPU tasks in parallel | `multiprocessing.Pool(N)` |
| Many concurrent I/O operations | `asyncio.gather(...)` |
| Process results as they complete | `as_completed(futures)` |
| Gracefully shut down workers | Poison pill (`None` in queue) |
