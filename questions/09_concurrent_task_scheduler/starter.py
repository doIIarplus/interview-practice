"""
Question 09: Concurrent Task Scheduler

Build a concurrent task scheduler that executes a DAG of tasks with dependencies,
maximizing parallelism and handling failures gracefully.

Uses concurrent.futures.ThreadPoolExecutor for parallel execution.
"""

import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of a task execution.

    Attributes:
        status: One of "success", "failed", or "skipped".
        result: The return value of the task if successful, None otherwise.
        error: The exception if the task failed, None otherwise.
        attempts: Number of times the task was attempted (0 if skipped).
    """
    status: str  # "success", "failed", or "skipped"
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0


@dataclass
class Task:
    """A registered task in the scheduler.

    Attributes:
        task_id: Unique identifier for this task.
        execute_fn: Zero-argument callable that performs the task's work.
        dependencies: Set of task IDs that must complete before this task starts.
        retries: Number of additional attempts allowed on failure.
    """
    task_id: str
    execute_fn: Callable[[], Any]
    dependencies: set = field(default_factory=set)
    retries: int = 0


# ---------------------------------------------------------------------------
# TaskScheduler — implement this
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Concurrent DAG task scheduler with retry and failure propagation.

    Executes a directed acyclic graph of tasks using a thread pool. Tasks with
    no unmet dependencies run in parallel. Failed tasks cause their dependents
    to be skipped. Independent branches are unaffected by failures.

    Usage:
        scheduler = TaskScheduler(max_workers=4)
        scheduler.add_task("A", lambda: "result_A")
        scheduler.add_task("B", lambda: "result_B", dependencies={"A"})
        results = scheduler.run()
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize the scheduler.

        Args:
            max_workers: Maximum number of concurrent threads in the pool.
        """
        self.max_workers = max_workers
        # TODO: Initialize your data structures
        pass

    def add_task(
        self,
        task_id: str,
        execute_fn: Callable[[], Any],
        dependencies: Optional[set[str]] = None,
        retries: int = 0,
    ) -> None:
        """Register a task with the scheduler.

        Args:
            task_id: Unique identifier for this task.
            execute_fn: Zero-argument callable that performs the task's work.
            dependencies: Set of task IDs that must complete successfully first.
            retries: Number of additional attempts on failure (0 = one attempt).

        Raises:
            ValueError: If a task with this ID already exists.
        """
        pass  # TODO

    def run(self) -> dict[str, TaskResult]:
        """Execute all tasks respecting dependencies and maximizing parallelism.

        Returns:
            A dict mapping each task_id to its TaskResult.

        Algorithm outline:
            1. Find all tasks with no dependencies (ready to run).
            2. Submit them to the thread pool.
            3. When a task completes:
               a. If success: check if any blocked tasks are now ready.
               b. If failed (after retries): mark dependent tasks as skipped.
            4. Repeat until all tasks have a result.
        """
        pass  # TODO


# ---------------------------------------------------------------------------
# Example DAGs for testing
# ---------------------------------------------------------------------------

def build_diamond_dag(scheduler: TaskScheduler, fail_b: bool = False) -> None:
    """Build a diamond-shaped DAG: A -> {B, C} -> D, plus independent E.

        A
       / \\
      B   C
       \\ /
        D

        E  (independent)
    """
    scheduler.add_task("A", lambda: "result_A")

    if fail_b:
        attempt_count = 0

        def failing_b():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError(f"B failed on attempt {attempt_count}")

        scheduler.add_task("B", failing_b, dependencies={"A"}, retries=2)
    else:
        scheduler.add_task("B", lambda: "result_B", dependencies={"A"})

    scheduler.add_task("C", lambda: "result_C", dependencies={"A"})
    scheduler.add_task("D", lambda: "result_D", dependencies={"B", "C"})
    scheduler.add_task("E", lambda: "result_E")


def build_chain_dag(scheduler: TaskScheduler, length: int = 5) -> None:
    """Build a linear chain: T0 -> T1 -> T2 -> ... -> T(n-1)."""
    for i in range(length):
        task_id = f"T{i}"
        deps = {f"T{i-1}"} if i > 0 else None
        scheduler.add_task(task_id, lambda idx=i: f"result_{idx}", dependencies=deps)


def build_wide_dag(scheduler: TaskScheduler, width: int = 10) -> None:
    """Build a wide DAG: root -> {W0, W1, ..., W(n-1)} -> sink."""
    scheduler.add_task("root", lambda: "root_done")
    for i in range(width):
        scheduler.add_task(
            f"W{i}",
            lambda idx=i: f"worker_{idx}_done",
            dependencies={"root"},
        )
    scheduler.add_task(
        "sink",
        lambda: "sink_done",
        dependencies={f"W{i}" for i in range(width)},
    )


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    scheduler = TaskScheduler(max_workers=4)
    scheduler.add_task("A", lambda: "result_A")
    scheduler.add_task("B", lambda: "result_B", dependencies={"A"})
    results = scheduler.run()
    for tid, result in results.items():
        print(f"  {tid}: {result.status} -> {result.result}")
