# Rubric: Question 08 — Load Balancer for Inference

**Total: 100 points**

---

## 1. Correct Least-Connections Routing (20 points)

### Full Credit (20 pts)
- Routes to the server with the lowest `current_load / capacity` ratio.
- Correctly breaks ties by choosing the server with the most GPU memory.
- Routing is efficient (not brute-forcing through all requests to count load).

**Reference implementation for server selection:**
```python
eligible = [s for s in self.servers.values()
            if not s.is_draining and s.has_capacity]

if is_large_request:
    eligible = [s for s in eligible
                if s.gpu_memory_gb >= LARGE_REQUEST_MIN_GPU_MEMORY_GB]

if not eligible:
    raise RuntimeError("No server available")

# Sort by (load_ratio ascending, gpu_memory_gb descending)
best = min(eligible, key=lambda s: (s.load_ratio, -s.gpu_memory_gb))
```

### Partial Credit (10-15 pts)
- Routes to least loaded server but doesn't handle ties correctly.
- Uses absolute load instead of load ratio (ignoring capacity differences).

### No Credit (0 pts)
- Round-robin or random routing.
- Does not consider server load at all.

---

## 2. Proper Capacity Enforcement (15 points)

### Full Credit (15 pts)
- Servers at maximum capacity are excluded from routing.
- Raises `RuntimeError` with a descriptive message when all servers are full.
- Correctly tracks load increments on `route_request` and decrements on `complete_request`.

### Partial Credit (7-10 pts)
- Tracks load correctly but allows over-capacity routing.
- Raises wrong exception type.

### No Credit (0 pts)
- No capacity tracking.

---

## 3. Server Removal / Graceful Drain (15 points)

### Full Credit (15 pts)
- `remove_server` immediately stops new requests from being routed to the server.
- In-flight requests on the server continue until `complete_request` is called.
- The server is fully removed from all data structures when its last in-flight request
  completes.
- The `is_draining` flag is visible in `get_server_stats()`.

### Partial Credit (7-10 pts)
- Server stops receiving new requests but is never fully removed.
- Or: server is removed immediately, losing track of in-flight requests.

### No Credit (0 pts)
- `remove_server` just deletes the server without draining.
- No handling of in-flight requests.

---

## 4. Large Request Routing to High-Memory Servers (15 points)

### Full Credit (15 pts)
- Requests with `estimated_tokens > 4096` are only routed to servers with
  `gpu_memory_gb >= 80`.
- Raises `RuntimeError` if no qualifying high-memory server has capacity.
- The threshold values are configurable (or at minimum, defined as constants).

### Partial Credit (7-10 pts)
- Filters by GPU memory but uses wrong threshold or comparison operator.
- Hard-codes threshold without making it clear/configurable.

### No Credit (0 pts)
- No special handling for large requests.

---

## 5. Clean OOP Design and Error Handling (15 points)

### Full Credit (15 pts)
- Uses appropriate data structures (e.g., `ServerInfo` dataclass or similar).
- Maintains a `request_id -> server_id` mapping for O(1) request lookup.
- Raises correct exception types with descriptive messages:
  - `ValueError` for duplicate server/request IDs.
  - `KeyError` for unknown server/request IDs.
  - `RuntimeError` for no-capacity situations.
- Clean separation of concerns; methods are focused and readable.
- No code duplication.

### Partial Credit (7-10 pts)
- Works but uses overly complex data structures or has code duplication.
- Missing some error checks.

### No Credit (0 pts)
- Spaghetti code, no encapsulation, missing most error handling.

---

## 6. Efficient Data Structures (10 points)

### Full Credit (10 pts)
- `complete_request` is O(1) via a `request_id -> server_id` mapping (no scanning all servers).
- `route_request` is O(S) where S is the number of servers (acceptable; servers are few).
- Avoids unnecessary iteration over all requests.
- Could mention that a heap/priority queue could optimize server selection if the server
  pool were very large.

### Partial Credit (5-7 pts)
- `complete_request` scans all servers to find the request (O(S * R) where R is requests per server).

### No Credit (0 pts)
- Gross inefficiencies, e.g., O(N^2) operations for every route/complete call.

---

## 7. Thread Safety Considerations (10 points)

### Full Credit (10 pts)
- Discusses that the base implementation is not thread-safe.
- Identifies specific race conditions:
  - Two concurrent `route_request` calls could both read the same load ratio and
    route to the same server, exceeding capacity.
  - A `complete_request` racing with a `remove_server` drain check.
- Proposes solutions: `threading.Lock`, `asyncio`, or atomic operations.
- Bonus: mentions read-write locks for better concurrency (reads are more frequent).

### Partial Credit (5-7 pts)
- Mentions thread safety but can't identify specific race conditions.
- Or: adds a lock but doesn't explain why it's needed.

### No Credit (0 pts)
- No awareness of thread safety issues.

---

## Red Flags
- Mutable default arguments in method signatures.
- Not tracking which server a request is on (scanning all servers on complete).
- Silently ignoring errors instead of raising exceptions.
- Memory leaks — never cleaning up completed request records.
- Using global state instead of instance attributes.

## Green Flags
- Clean use of dataclasses for server state.
- Property methods for derived values (load_ratio, has_capacity).
- Comprehensive docstrings and type hints.
- Mentions production considerations: health checks, circuit breakers, request queuing.
- Discusses how this compares to real load balancers (Envoy, HAProxy, Kubernetes).
- Considers request queuing when all servers are at capacity (rather than just rejecting).
