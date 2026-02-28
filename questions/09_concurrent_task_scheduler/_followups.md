# Follow-Up Questions: Question 09 — Concurrent Task Scheduler

---

## 1. How would you detect circular dependencies? At what point should you check?

**Expected Answer:**
- Circular dependencies make the DAG unsolvable — tasks in a cycle can never become ready
  because they are all waiting on each other.
- **Detection methods:**
  - **Kahn's algorithm (BFS-based topological sort):** Compute in-degrees. Repeatedly
    remove nodes with in-degree 0 and decrement neighbors' in-degrees. If nodes remain
    after the process, there is a cycle.
  - **DFS with coloring:** Mark nodes as WHITE (unvisited), GRAY (in progress), BLACK
    (done). If you encounter a GRAY node during DFS, there is a cycle.
- **When to check:**
  - Option 1: At `run()` time, before submitting any tasks. This is simpler and catches
    all cycles at once.
  - Option 2: At `add_task()` time, incrementally. This gives earlier feedback but is
    more complex (must check if adding this edge creates a cycle).
  - Option 1 is preferred for simplicity unless the DAG is built interactively.
- Another signal: if `run()` has submitted all currently ready tasks and some tasks remain
  without results but none are in-flight, there must be a cycle or a dependency on a
  non-existent task.

---

## 2. What if tasks can have timeouts? How would you implement per-task timeouts?

**Expected Answer:**
- `concurrent.futures.Future` supports `result(timeout=N)` which raises `TimeoutError`
  after N seconds.
- For per-task timeouts, add a `timeout` parameter to `add_task` and wrap the execution:
  ```python
  future = executor.submit(task.execute_fn)
  try:
      result = future.result(timeout=task.timeout)
  except TimeoutError:
      future.cancel()  # Best-effort cancellation
      # Mark as failed
  ```
- Caveat: `future.cancel()` only prevents the task from starting if it hasn't been picked
  up yet. If it's already running, Python threads cannot be forcibly interrupted.
- For hard timeouts on running tasks, you'd need multiprocessing (can kill processes) or
  asyncio with `asyncio.wait_for`.
- Alternative: have the task function itself check a cancellation token periodically.

---

## 3. How would you add task cancellation support?

**Expected Answer:**
- Define a `CancellationToken` (or `threading.Event`) that tasks can check:
  ```python
  class CancellationToken:
      def __init__(self):
          self._cancelled = threading.Event()
      def cancel(self):
          self._cancelled.set()
      def is_cancelled(self) -> bool:
          return self._cancelled.is_set()
  ```
- Task functions would accept a token and periodically check `token.is_cancelled()`.
- The scheduler's `cancel()` method would:
  1. Set the cancellation token for the target task.
  2. Cancel any pending futures (`future.cancel()`).
  3. Mark the task and its dependents as "cancelled."
- Cooperative cancellation is the standard pattern in Python because threads cannot be
  forcibly killed.
- For the scheduler itself, add a `cancel_all()` that stops submitting new tasks and
  cancels all pending futures.

---

## 4. What if you needed to limit the number of concurrent tasks (e.g., max 4 GPU tasks)?

**Expected Answer:**
- The `ThreadPoolExecutor(max_workers=N)` already limits overall concurrency.
- For **resource-specific limits** (e.g., "at most 4 GPU tasks, unlimited CPU tasks"),
  use a `threading.Semaphore`:
  ```python
  gpu_semaphore = threading.Semaphore(4)

  def execute_gpu_task(fn):
      with gpu_semaphore:
          return fn()
  ```
- Tag tasks with resource requirements at `add_task` time.
- More sophisticated: implement a resource manager that tracks available resources
  (GPU slots, memory) and only makes a task "ready" when both its dependencies are met
  AND its required resources are available.
- This is how systems like Kubernetes (resource requests/limits) and Airflow (pools)
  handle it.

---

## 5. How would you persist the DAG state for crash recovery?

**Expected Answer:**
- Serialize the DAG and task states to durable storage (database, file, or distributed
  key-value store).
- Store for each task: task_id, status, dependencies, result, error, attempts.
- On startup, load the persisted state and resume:
  - Tasks already marked "success" are skipped.
  - Tasks marked "in_progress" at crash time are re-executed (must ensure task
    idempotency).
  - Tasks still "pending" follow normal scheduling.
- Use write-ahead logging (WAL): persist state changes before executing them.
- This is essentially what Airflow does with its metadata database.
- For distributed systems, use a consensus protocol (Raft/Paxos) or a distributed
  database to ensure consistency across scheduler replicas.
- Idempotency is critical: if a task partially completed before a crash, re-executing it
  must not cause incorrect results (e.g., double-writing, duplicate side effects).

---

## 6. How does this relate to systems like Airflow, Celery, or Dask?

**Expected Answer:**
- **Apache Airflow:**
  - DAG-based workflow orchestrator, very similar to our scheduler.
  - DAGs defined in Python, tasks can be Python functions, Bash commands, SQL queries, etc.
  - Has scheduling (cron-based triggers), retry policies, alerting, and a web UI.
  - Uses a metadata database for state persistence and crash recovery.
  - Workers execute tasks; the scheduler just orchestrates.
- **Celery:**
  - Distributed task queue, not inherently DAG-based.
  - Tasks are submitted to a broker (RabbitMQ, Redis) and picked up by workers.
  - Supports chaining, groups, and chords for dependency patterns.
  - Better for "fire-and-forget" tasks than complex DAGs.
- **Dask:**
  - Parallel computing library for Python with a DAG-based task scheduler.
  - Optimized for data-parallel workloads (like pandas/numpy operations).
  - Dynamic task graph (tasks can spawn new tasks).
  - Scheduler uses sophisticated heuristics for data locality and memory management.
- Our implementation is closest to a simplified Airflow scheduler, but without
  persistence, scheduling triggers, or distributed workers.

---

## 7. What would change if tasks were distributed across multiple machines?

**Expected Answer:**
- **Serialization:** Task functions and results must be serializable (pickle, JSON, or
  protocol buffers) to send across the network.
- **Task distribution:** Replace the local thread pool with a distributed task queue
  (Celery, RabbitMQ, gRPC calls to worker nodes).
- **State management:** Centralized state store (database, etcd, ZooKeeper) instead of
  in-memory dicts.
- **Fault tolerance:** Workers can crash. Need heartbeats, task timeouts, and re-assignment
  of tasks from dead workers.
- **Data locality:** If tasks produce large outputs, prefer scheduling dependent tasks on
  the same machine to avoid data transfer. This is what Dask and Spark optimize for.
- **Network partitions:** Must handle split-brain scenarios where the scheduler can't
  reach some workers. Use consensus protocols or "at-least-once" semantics with
  idempotent tasks.
- **Scheduler HA:** The scheduler itself becomes a single point of failure. Run multiple
  scheduler replicas with leader election (Raft, ZooKeeper).
- This is essentially the architecture of systems like Ray, Dask Distributed, or
  Kubernetes Jobs.
