# Question 09: Concurrent Task Scheduler

## Difficulty: Hard
## Topics: Concurrency, DAG Scheduling, Fault Tolerance, Threading
## Estimated Time: 50-70 minutes

---

## Background

Distributed ML systems frequently execute complex workflows composed of interdependent tasks:
training data preprocessing, model training steps, evaluation, checkpointing, and deployment.
These workflows form a DAG (Directed Acyclic Graph) where some tasks depend on others and
independent tasks should run in parallel for maximum throughput.

A robust task scheduler must:
- Respect dependencies (never start a task before its prerequisites complete).
- Maximize parallelism (run all ready tasks concurrently).
- Handle failures gracefully (retry failed tasks, skip downstream tasks if retries are
  exhausted, but still complete independent branches).

---

## Task

Build a concurrent task scheduler that executes a DAG of tasks with dependencies.

### TaskResult

Each task produces a `TaskResult` with:
- `status`: One of `"success"`, `"failed"`, or `"skipped"`.
  - `"success"`: The task executed and returned a value.
  - `"failed"`: The task raised an exception after exhausting all retries.
  - `"skipped"`: The task was not executed because one of its dependencies (direct or
    transitive) failed.
- `result`: The return value of the task function if successful, `None` otherwise.
- `error`: The exception object if the task failed, `None` otherwise.
- `attempts`: The number of times the task was attempted (1 for immediate success,
  `retries + 1` for exhausting all retries, 0 for skipped tasks).

### TaskScheduler

Implement a `TaskScheduler` class with two methods:

#### `add_task(task_id, execute_fn, dependencies=None, retries=0)`

Register a task with the scheduler.
- `task_id` (str): Unique identifier for this task.
- `execute_fn` (Callable[[], Any]): A zero-argument callable that performs the task's work.
  It may return a value (captured in `TaskResult.result`) or raise an exception.
- `dependencies` (set[str] | None): Set of task IDs that must complete successfully before
  this task can start. Default: no dependencies.
- `retries` (int): Number of additional attempts if the task fails. Default: 0 (no retries;
  one attempt total).
- Raise `ValueError` if a task with this ID already exists.

#### `run() -> dict[str, TaskResult]`

Execute all registered tasks respecting dependencies and maximizing parallelism. Returns
a dict mapping each task_id to its `TaskResult`.

**Execution rules:**
1. Tasks with no unmet dependencies should start immediately, in parallel.
2. When a task completes successfully, check if any blocked tasks now have all dependencies
   met and start them.
3. If a task fails (raises an exception), retry it up to `retries` times. If it still fails:
   - Mark it as `"failed"`.
   - Mark all tasks that depend on it (directly or transitively) as `"skipped"`.
4. Independent branches of the DAG should complete even if another branch fails.
5. Use `concurrent.futures.ThreadPoolExecutor` for parallelism.

---

## Example

Consider this DAG:

```
    A
   / \
  B   C
   \ /
    D

    E  (independent)
```

- A has no dependencies.
- B depends on A.
- C depends on A.
- E has no dependencies.
- D depends on both B and C.

```python
scheduler = TaskScheduler(max_workers=4)

scheduler.add_task("A", lambda: "result_A")
scheduler.add_task("B", lambda: "result_B", dependencies={"A"})
scheduler.add_task("C", lambda: "result_C", dependencies={"A"})
scheduler.add_task("D", lambda: "result_D", dependencies={"B", "C"})
scheduler.add_task("E", lambda: "result_E")

results = scheduler.run()
```

**Expected execution order:**
1. A and E start immediately (no dependencies), running in parallel.
2. When A completes, B and C start in parallel.
3. When both B and C complete, D starts.
4. All tasks succeed.

```python
results["A"].status   # "success"
results["A"].result   # "result_A"
results["A"].attempts # 1
results["E"].status   # "success"  (ran in parallel with A)
results["D"].status   # "success"  (ran after B and C)
```

### Failure Example

If B fails after all retries:

```python
def failing_B():
    raise ValueError("B broke")

scheduler = TaskScheduler(max_workers=4)
scheduler.add_task("A", lambda: "result_A")
scheduler.add_task("B", failing_B, dependencies={"A"}, retries=2)
scheduler.add_task("C", lambda: "result_C", dependencies={"A"})
scheduler.add_task("D", lambda: "result_D", dependencies={"B", "C"})
scheduler.add_task("E", lambda: "result_E")

results = scheduler.run()
```

**Expected results:**
```python
results["A"].status    # "success"
results["A"].attempts  # 1

results["B"].status    # "failed"
results["B"].attempts  # 3 (1 initial + 2 retries)
results["B"].error     # ValueError("B broke")

results["C"].status    # "success"
results["C"].attempts  # 1

results["D"].status    # "skipped"  (depends on B which failed)
results["D"].attempts  # 0

results["E"].status    # "success"  (independent, unaffected by B's failure)
results["E"].attempts  # 1
```

Note that C still succeeds (it only depends on A), and E still succeeds (it is independent).
Only D is skipped because it depends on B, which failed.

---

## Constraints

- Task IDs are unique strings.
- The DAG may have 0 to 10,000 tasks.
- The DAG is guaranteed to be acyclic for the base implementation. (Bonus: detect cycles.)
- `execute_fn` is a zero-argument callable. It may take variable time to complete.
- `retries` is a non-negative integer (0 means one attempt total).
- The scheduler should use `concurrent.futures.ThreadPoolExecutor` for parallelism.
- `max_workers` controls the thread pool size (default: 4).

---

## What We're Looking For

1. Correct dependency resolution — tasks never start before their dependencies complete
2. Maximum parallelism — all ready tasks run concurrently, not sequentially
3. Proper failure handling — failed tasks skip dependents, independent branches unaffected
4. Correct retry logic — attempts the right number of times, captures the final exception
5. Clean use of threading primitives — thread pool, futures, synchronization
6. Awareness of edge cases — empty DAG, single task, all tasks fail, circular dependencies
