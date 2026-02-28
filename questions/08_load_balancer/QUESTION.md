# Question 08: Load Balancer for Inference

## Difficulty: Medium
## Topics: System Design, Load Balancing, OOP, Resource Management
## Estimated Time: 45-60 minutes

---

## Background

When serving LLM inference at scale, you need to route incoming requests to a pool of GPU
servers efficiently. Unlike traditional web load balancing, LLM inference has unique
characteristics:

- Requests have **highly variable latencies** — a 10-token completion is much faster than a
  4,000-token completion.
- GPU servers have **limited concurrency** — each server can only handle a fixed number of
  simultaneous requests before running out of GPU memory.
- Some requests are **memory-intensive** — long-context or high-token requests need servers
  with more GPU memory.
- Servers may need to be **drained gracefully** for maintenance or rolling deployments.

Your task is to build a load balancer that handles these concerns.

---

## Task

Implement a `LoadBalancer` class with the following interface:

### Methods

#### `add_server(server_id: str, capacity: int, gpu_memory_gb: int) -> None`
Register a new server with the load balancer.
- `server_id`: Unique identifier for the server (e.g., `"gpu-server-01"`).
- `capacity`: Maximum number of concurrent requests this server can handle.
- `gpu_memory_gb`: Amount of GPU memory in gigabytes (e.g., 40, 80).
- Raise `ValueError` if a server with this ID already exists.

#### `remove_server(server_id: str) -> None`
Remove a server from the pool.
- The server should stop receiving **new** requests immediately.
- Any in-flight requests on this server should be allowed to complete (graceful drain).
- Once all in-flight requests complete, the server is fully removed.
- Raise `KeyError` if the server does not exist.

#### `route_request(request_id: str, estimated_tokens: int) -> str`
Route an incoming request to the best available server.
- Returns the `server_id` of the chosen server.
- Raise `ValueError` if the request ID is already in use.
- Raise `RuntimeError` if no server has available capacity.

**Routing strategy — Weighted Least Connections:**
1. Calculate the load ratio for each server: `current_load / capacity`.
2. Route to the server with the **lowest** load ratio.
3. If there is a tie, choose the server with the **most available GPU memory**.
4. For requests with `estimated_tokens > 4096`, only consider servers with
   `gpu_memory_gb >= 80`. Raise `RuntimeError` if no qualifying server is available.
5. Servers that are being drained (removed but have in-flight requests) must **not**
   receive new requests.

#### `complete_request(request_id: str) -> None`
Mark a request as completed, freeing capacity on its server.
- Raise `KeyError` if the request ID is not found.
- If the server was marked for removal and this was its last in-flight request,
  fully remove the server now.

#### `get_server_stats() -> dict[str, dict]`
Return current load statistics for all active servers. Each entry should contain:
```python
{
    "server_id": {
        "capacity": int,
        "current_load": int,
        "gpu_memory_gb": int,
        "load_ratio": float,       # current_load / capacity
        "is_draining": bool,       # True if marked for removal
        "active_requests": list[str]  # list of active request IDs
    }
}
```

---

## Example

```python
lb = LoadBalancer()

# Add servers
lb.add_server("gpu-01", capacity=4, gpu_memory_gb=40)
lb.add_server("gpu-02", capacity=4, gpu_memory_gb=80)
lb.add_server("gpu-03", capacity=2, gpu_memory_gb=80)

# Route some requests
server = lb.route_request("req-1", estimated_tokens=100)
# Both gpu-01 and gpu-02 have load_ratio 0/4 = 0.0, gpu-03 has 0/2 = 0.0
# Three-way tie: gpu-02 or gpu-03 wins (80GB > 40GB), then gpu-02 wins
# over gpu-03 by capacity tie-breaking or either is acceptable
print(server)  # "gpu-02" or "gpu-03" (both have 80GB)

server = lb.route_request("req-2", estimated_tokens=5000)
# estimated_tokens > 4096, so only gpu-02 and gpu-03 qualify (>= 80GB)
print(server)  # "gpu-02" or "gpu-03"

# Complete a request
lb.complete_request("req-1")

# Check stats
stats = lb.get_server_stats()
print(stats["gpu-01"]["current_load"])  # 0
print(stats["gpu-01"]["load_ratio"])    # 0.0

# Drain a server
lb.remove_server("gpu-02")
# gpu-02 still has in-flight requests, so it stays until they complete
stats = lb.get_server_stats()
print(stats["gpu-02"]["is_draining"])   # True

# New requests won't go to gpu-02
server = lb.route_request("req-3", estimated_tokens=100)
print(server)  # "gpu-01" or "gpu-03" (not gpu-02, it's draining)

# When the last request on gpu-02 completes, it's fully removed
lb.complete_request("req-2")
assert "gpu-02" not in lb.get_server_stats()
```

---

## Constraints

- Server IDs and request IDs are unique strings.
- `capacity` is a positive integer (1 to 1000).
- `gpu_memory_gb` is a positive integer (typically 16, 24, 40, 48, 80).
- `estimated_tokens` is a positive integer.
- You may assume single-threaded execution for the base implementation, but be prepared
  to discuss thread safety.

---

## What We're Looking For

1. Clean, well-structured OOP design
2. Correct implementation of weighted least-connections routing
3. Proper handling of server draining with in-flight requests
4. Robust error handling with meaningful error messages
5. Efficient data structures — avoid O(n) scans of all requests for every operation
6. Ability to discuss production concerns: thread safety, health checks, queuing
