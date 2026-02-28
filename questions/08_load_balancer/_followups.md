# Follow-Up Questions: Question 08 — Load Balancer for Inference

---

## 1. How would you handle server health checks? What if a server becomes unresponsive?

**Expected Answer:**
- Implement periodic health checks (heartbeat/ping) to each server on a configurable
  interval (e.g., every 5-10 seconds).
- Use a "circuit breaker" pattern: after N consecutive failed health checks, mark the
  server as unhealthy and stop routing to it.
- Distinguish between "degraded" (slow responses, partial failures) and "down" (completely
  unresponsive).
- For in-flight requests on a server that goes down, implement a timeout after which the
  request is re-routed to another server (with idempotency guarantees).
- Implement a "recovery" state: after a server comes back online, gradually ramp up traffic
  (don't send full load immediately) — this is sometimes called "slow start."
- Health check types: passive (observe failures from real traffic) vs. active (send
  dedicated probe requests).

---

## 2. How would you implement request queuing when all servers are at capacity?

**Expected Answer:**
- Instead of immediately rejecting with `RuntimeError`, add the request to a priority queue.
- When a request completes and frees capacity, dequeue the next waiting request.
- Implement a maximum queue depth and a maximum wait time (timeout).
- Queue ordering options:
  - FIFO (fairness)
  - Priority-based (premium users first)
  - Shortest-job-first (estimated_tokens ascending, to maximize throughput)
- Backpressure: if the queue is full, return HTTP 429 (Too Many Requests) with a
  `Retry-After` header.
- Monitor queue depth as a key health metric — sustained high queue depth indicates
  you need to scale up.

---

## 3. What routing strategy would you use if requests have very different latencies?

**Expected Answer:**
- **Least-connections** (what we implemented) is already a good starting point because
  it accounts for in-flight request count.
- **Weighted response time**: track the average (or P95) response time per server and
  factor it into routing. Prefer servers with lower latency.
- **Join-Shortest-Queue (JSQ)**: similar to least-connections but considers estimated
  completion time, not just count.
- **Power of Two Choices**: randomly pick two servers and route to the less loaded one.
  Simpler to implement, nearly as good as full least-connections, and avoids the
  "thundering herd" problem.
- **Predictive routing**: use the `estimated_tokens` to predict how long a request will
  take and route to balance expected completion times, not just request counts.

---

## 4. How would you handle "sticky sessions" for multi-turn conversations?

**Expected Answer:**
- Multi-turn conversations benefit from being routed to the same server because the
  KV cache from previous turns may still be in GPU memory, avoiding recomputation.
- Implement session affinity: maintain a `session_id -> server_id` mapping.
- If the preferred server is at capacity or unhealthy, fall back to normal routing
  (the KV cache will need to be recomputed, but correctness is maintained).
- Set a TTL on sticky sessions so stale mappings are cleaned up.
- Consider "soft affinity" — prefer the sticky server but allow overflow to others,
  versus "hard affinity" where the session must go to the same server.
- KV cache migration: some advanced systems can transfer the KV cache between servers,
  making session affinity less critical.

---

## 5. What if you needed to support request priorities (premium vs. free tier)?

**Expected Answer:**
- Reserve a portion of each server's capacity for premium requests (e.g., 75% available
  to all, 25% reserved for premium).
- Implement priority queuing: when all servers are at capacity, premium requests get
  dequeued first.
- Rate limiting per tier: free tier has lower requests-per-minute limits.
- Separate server pools: dedicate specific servers to premium traffic for isolation.
- Preemption (advanced): for batch/offline requests, allow a premium real-time request
  to preempt and pause a running batch request.
- SLA-based routing: premium requests get routed to servers with the lowest current
  latency, not just lowest load.

---

## 6. How would you implement graceful server drain for rolling deployments?

**Expected Answer:**
- Our `remove_server` already implements the basic pattern: stop new traffic, wait for
  in-flight requests to complete.
- For rolling deployments, you'd drain servers one at a time (or in small batches),
  waiting for each to fully drain before proceeding.
- Set a drain timeout: if in-flight requests don't complete within N minutes, force-kill
  them (with appropriate error handling/retry on the client side).
- Pre-warm the replacement server: before draining the old server, add the new server to
  the pool and let it take traffic to warm its caches.
- Coordinate with a deployment orchestrator (Kubernetes, Nomad) that manages the
  rolling update sequence.
- Canary deployment: route a small percentage of traffic to the new version first, monitor
  for errors, then proceed with full rollout.

---

## 7. How does this compare to real systems like Kubernetes service mesh or Envoy?

**Expected Answer:**
- **Envoy proxy** implements many of the same concepts: weighted least-connections, health
  checking, circuit breaking, outlier detection, and request draining.
- **Kubernetes Services** use kube-proxy for basic load balancing (round-robin or
  iptables-based), but service meshes (Istio/Envoy) add more sophisticated routing.
- Key differences from our implementation:
  - Real systems handle connection pooling and HTTP/2 multiplexing.
  - They support retries with exponential backoff and jitter.
  - They have observability built in (metrics, tracing, logging).
  - They handle TLS termination and authentication.
- LLM-specific systems (e.g., vLLM, TensorRT-LLM serving, Triton Inference Server) add:
  - Continuous batching (dynamically adding requests to a running batch).
  - KV cache management.
  - Token-level streaming.
  - Model-parallel routing (a single request spans multiple GPUs).
- Our load balancer is a simplified version of what systems like Anthropic's internal
  serving infrastructure must handle at a much larger scale.
