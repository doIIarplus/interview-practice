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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_dag() -> None:
    """Test the diamond DAG with all tasks succeeding."""
    print("Test: Basic diamond DAG (all succeed)")
    scheduler = TaskScheduler(max_workers=4)
    build_diamond_dag(scheduler, fail_b=False)
    results = scheduler.run()

    assert results["A"].status == "success", f"A: {results['A']}"
    assert results["A"].result == "result_A"
    assert results["A"].attempts == 1

    assert results["B"].status == "success", f"B: {results['B']}"
    assert results["C"].status == "success", f"C: {results['C']}"
    assert results["D"].status == "success", f"D: {results['D']}"
    assert results["E"].status == "success", f"E: {results['E']}"

    print("  PASSED\n")


def test_failure_propagation() -> None:
    """Test that B's failure skips D but not C or E."""
    print("Test: Failure propagation (B fails -> D skipped)")
    scheduler = TaskScheduler(max_workers=4)
    build_diamond_dag(scheduler, fail_b=True)
    results = scheduler.run()

    assert results["A"].status == "success", f"A: {results['A']}"
    assert results["A"].attempts == 1

    assert results["B"].status == "failed", f"B: {results['B']}"
    assert results["B"].attempts == 3  # 1 initial + 2 retries
    assert isinstance(results["B"].error, ValueError)

    assert results["C"].status == "success", f"C: {results['C']}"
    assert results["C"].attempts == 1

    assert results["D"].status == "skipped", f"D: {results['D']}"
    assert results["D"].attempts == 0

    assert results["E"].status == "success", f"E: {results['E']}"
    assert results["E"].attempts == 1

    print("  PASSED\n")


def test_empty_dag() -> None:
    """Test that an empty DAG returns an empty result dict."""
    print("Test: Empty DAG")
    scheduler = TaskScheduler(max_workers=4)
    results = scheduler.run()
    assert results == {}, f"Expected empty dict, got {results}"
    print("  PASSED\n")


def test_single_task() -> None:
    """Test a DAG with a single task."""
    print("Test: Single task")
    scheduler = TaskScheduler(max_workers=4)
    scheduler.add_task("only", lambda: 42)
    results = scheduler.run()
    assert results["only"].status == "success"
    assert results["only"].result == 42
    assert results["only"].attempts == 1
    print("  PASSED\n")


def test_chain_dag() -> None:
    """Test a linear chain of tasks."""
    print("Test: Chain DAG (T0 -> T1 -> T2 -> T3 -> T4)")
    scheduler = TaskScheduler(max_workers=4)
    build_chain_dag(scheduler, length=5)
    results = scheduler.run()

    for i in range(5):
        tid = f"T{i}"
        assert results[tid].status == "success", f"{tid}: {results[tid]}"
        assert results[tid].result == f"result_{i}"
    print("  PASSED\n")


def test_wide_dag() -> None:
    """Test a wide fan-out/fan-in DAG."""
    print("Test: Wide DAG (root -> 10 workers -> sink)")
    scheduler = TaskScheduler(max_workers=4)
    build_wide_dag(scheduler, width=10)
    results = scheduler.run()

    assert results["root"].status == "success"
    for i in range(10):
        assert results[f"W{i}"].status == "success"
    assert results["sink"].status == "success"
    print("  PASSED\n")


def test_parallelism() -> None:
    """Verify that independent tasks actually run in parallel."""
    print("Test: Parallelism (independent tasks run concurrently)")
    scheduler = TaskScheduler(max_workers=4)

    # Four independent tasks, each sleeping 0.5s
    # If run sequentially: ~2.0s. If parallel: ~0.5s.
    for i in range(4):
        scheduler.add_task(
            f"P{i}",
            lambda: time.sleep(0.5) or "done",
        )

    start = time.perf_counter()
    results = scheduler.run()
    elapsed = time.perf_counter() - start

    for i in range(4):
        assert results[f"P{i}"].status == "success"

    # Should complete in roughly 0.5s, not 2.0s
    assert elapsed < 1.5, f"Tasks took {elapsed:.2f}s — not parallel enough!"
    print(f"  Completed 4 x 0.5s tasks in {elapsed:.2f}s")
    print("  PASSED\n")


def test_retry_then_succeed() -> None:
    """Test a task that fails once then succeeds on retry."""
    print("Test: Retry then succeed")
    call_count = 0

    def flaky_task():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError(f"Flaky failure #{call_count}")
        return "finally_worked"

    scheduler = TaskScheduler(max_workers=4)
    scheduler.add_task("flaky", flaky_task, retries=3)
    results = scheduler.run()

    assert results["flaky"].status == "success", f"flaky: {results['flaky']}"
    assert results["flaky"].result == "finally_worked"
    assert results["flaky"].attempts == 3  # failed twice, succeeded on third
    print(f"  Task succeeded on attempt {results['flaky'].attempts}")
    print("  PASSED\n")


def test_duplicate_task_id() -> None:
    """Test that adding a duplicate task ID raises ValueError."""
    print("Test: Duplicate task ID")
    scheduler = TaskScheduler(max_workers=4)
    scheduler.add_task("dup", lambda: None)
    try:
        scheduler.add_task("dup", lambda: None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASSED\n")


def run_all_tests() -> None:
    """Run all test cases."""
    print("=" * 60)
    print("Concurrent Task Scheduler — Test Suite")
    print("=" * 60 + "\n")

    test_basic_dag()
    test_failure_propagation()
    test_empty_dag()
    test_single_task()
    test_chain_dag()
    test_wide_dag()
    test_parallelism()
    test_retry_then_succeed()
    test_duplicate_task_id()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
