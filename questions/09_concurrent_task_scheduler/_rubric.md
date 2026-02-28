# Rubric: Question 09 — Concurrent Task Scheduler

**Total: 100 points**

---

## 1. Correct Dependency Resolution and Topological Execution (25 points)

### Full Credit (25 pts)
- Tasks never start before all their dependencies have completed successfully.
- The execution order respects the DAG structure for all test cases.
- Works correctly for chains, diamonds, wide fan-out/fan-in, and disconnected components.
- Handles tasks with no dependencies (start immediately).
- Handles tasks with multiple dependencies (wait for all).

**Key implementation detail:**
The candidate should track remaining (unmet) dependencies for each task. When a task
completes, decrement the dependency count for all tasks that depend on it. When a task's
remaining count reaches zero, it becomes ready to execute.

```python
# Example data structures:
remaining_deps = {task_id: len(task.dependencies) for ...}
dependents = defaultdict(set)  # task_id -> set of tasks that depend on it
for task in tasks:
    for dep in task.dependencies:
        dependents[dep].add(task.task_id)
```

### Partial Credit (12-18 pts)
- Works for simple cases but fails for complex DAGs.
- Topological sort is correct but executes tasks sequentially (no parallelism).

### No Credit (0 pts)
- Tasks execute in random/insertion order without dependency checking.

---

## 2. Proper Parallelism (20 points)

### Full Credit (20 pts)
- Uses `concurrent.futures.ThreadPoolExecutor` correctly.
- Independent tasks (no dependency relationship) run concurrently.
- The parallelism test passes: 4 independent 0.5s tasks complete in ~0.5s, not ~2s.
- Newly unblocked tasks are submitted to the pool immediately when their last dependency
  completes (not batched into "rounds").

**Correct approach:**
```python
with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    # Submit all initially ready tasks
    for task in ready_tasks:
        future = executor.submit(self._execute_task, task)
        future.add_done_callback(...)
    # Wait for all tasks to complete
```

### Partial Credit (10-15 pts)
- Uses thread pool but executes in "rounds" (all tasks at depth 0, then all at depth 1, etc.),
  missing some parallelism opportunities.
- Or: uses threads manually (threading.Thread) instead of ThreadPoolExecutor.

### Minimal Credit (5 pts)
- All tasks execute sequentially on one thread.

### No Credit (0 pts)
- No concurrency at all.

---

## 3. Correct Failure Propagation (20 points)

### Full Credit (20 pts)
- When a task fails after all retries, it is marked as `"failed"`.
- All tasks that depend on it — directly or transitively — are marked as `"skipped"`
  with `attempts=0`.
- Independent branches (tasks not downstream of the failed task) are unaffected and
  complete normally.
- A task is only skipped if at least one of its dependencies failed. If some dependencies
  succeeded and others failed, it is still skipped.

**Transitive skip propagation:**
```
A -> B (fails) -> D -> F
A -> C (succeeds) -> E
```
B fails: D and F are skipped. C and E succeed. A succeeded.

### Partial Credit (10-15 pts)
- Skips direct dependents but not transitive dependents.
- Or: marks all remaining tasks as skipped when any task fails (too aggressive).

### No Credit (0 pts)
- No failure handling — exceptions crash the scheduler.
- Or: all tasks after a failure are skipped regardless of dependencies.

---

## 4. Retry Logic (10 points)

### Full Credit (10 pts)
- A task with `retries=N` is attempted at most `N + 1` times.
- If the task succeeds on any attempt, it is marked as `"success"` with `attempts`
  equal to the number of attempts made (including failed ones).
- If the task fails on all attempts, it is marked as `"failed"` with `attempts = retries + 1`,
  and `error` is the exception from the last attempt.
- Retry happens synchronously within the task execution (not re-submitted to the pool).

### Partial Credit (5-7 pts)
- Retries work but `attempts` count is off by one.
- Or: retries are implemented by re-submitting to the pool (works but adds complexity).

### No Credit (0 pts)
- No retry support.

---

## 5. Clean Design and Threading Primitives (15 points)

### Full Credit (15 pts)
- Clean separation between task registration (`add_task`) and execution (`run`).
- Proper use of synchronization primitives:
  - A lock protects shared mutable state (remaining dependency counts, results dict).
  - Or uses `concurrent.futures.as_completed()` or callbacks.
  - A countdown mechanism (Event, Condition, or manual counter) to know when all tasks
    are done.
- No busy-waiting or polling loops.
- The scheduler is reusable or clearly documents that `run()` can only be called once.

**Two common correct approaches:**

1. **Callback-based:**
   Submit ready tasks to pool. On completion callback (thread-safe), update remaining
   deps and submit newly ready tasks. Use a threading.Event or counter to detect completion.

2. **as_completed loop:**
   Submit ready tasks, use `as_completed()` to process results, submit newly ready tasks.
   Loop until all tasks have results.

### Partial Credit (7-10 pts)
- Works but has race conditions that happen to not trigger in tests.
- Uses sleep/polling instead of proper synchronization.
- Excessive locking (one giant lock around everything).

### No Credit (0 pts)
- Race conditions that cause incorrect results.
- Deadlocks.

---

## 6. Edge Cases (10 points)

### Full Credit (10 pts)
- Empty DAG: `run()` returns `{}`.
- Single task: works correctly.
- All tasks have no dependencies: all run in parallel.
- All tasks fail: each failed, dependents skipped, scheduler completes.
- Duplicate task ID: raises `ValueError`.
- Bonus (not required): circular dependency detection at `add_task` or `run` time.

### Partial Credit (5-7 pts)
- Handles most edge cases but crashes on empty DAG or single task.

### No Credit (0 pts)
- Crashes on trivial inputs.

---

## Red Flags
- Using `time.sleep()` as a synchronization mechanism.
- Global mutable state instead of instance attributes.
- Not using `with` statement for the ThreadPoolExecutor (resource leak).
- Ignoring thread safety entirely (unprotected shared state mutations in callbacks).
- Re-implementing threading primitives instead of using standard library.
- The task execution function takes arguments (should be zero-argument callable).

## Green Flags
- Uses `concurrent.futures.ThreadPoolExecutor` idiomatically.
- Clean callback-based or `as_completed`-based event loop.
- Implements cycle detection (BFS/DFS at `run` time).
- Discusses the tradeoff between callback-based and polling-based approaches.
- Mentions that in production you'd use something like Celery, Airflow, or Dask.
- Considers task cancellation and timeout as extensions.
- Notes that Python's GIL means true CPU parallelism requires multiprocessing, but
  for I/O-bound tasks (like RPC calls in ML pipelines), threading is appropriate.
- Uses `threading.Event` or `threading.Condition` for clean completion signaling.
