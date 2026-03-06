"""Hidden tests for Question 09: Concurrent Task Scheduler
Run: python questions/09_concurrent_task_scheduler/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
from starter import TaskScheduler, TaskResult, build_diamond_dag, build_chain_dag, build_wide_dag


def test_basic_dag():
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


def test_failure_propagation():
    """Test that B's failure skips D but not C or E."""
    print("Test: Failure propagation (B fails -> D skipped)")
    scheduler = TaskScheduler(max_workers=4)
    build_diamond_dag(scheduler, fail_b=True)
    results = scheduler.run()

    assert results["A"].status == "success", f"A: {results['A']}"
    assert results["A"].attempts == 1
    assert results["B"].status == "failed", f"B: {results['B']}"
    assert results["B"].attempts == 3
    assert isinstance(results["B"].error, ValueError)
    assert results["C"].status == "success", f"C: {results['C']}"
    assert results["C"].attempts == 1
    assert results["D"].status == "skipped", f"D: {results['D']}"
    assert results["D"].attempts == 0
    assert results["E"].status == "success", f"E: {results['E']}"
    assert results["E"].attempts == 1
    print("  PASSED\n")


def test_empty_dag():
    """Test that an empty DAG returns an empty result dict."""
    print("Test: Empty DAG")
    scheduler = TaskScheduler(max_workers=4)
    results = scheduler.run()
    assert results == {}, f"Expected empty dict, got {results}"
    print("  PASSED\n")


def test_single_task():
    """Test a DAG with a single task."""
    print("Test: Single task")
    scheduler = TaskScheduler(max_workers=4)
    scheduler.add_task("only", lambda: 42)
    results = scheduler.run()
    assert results["only"].status == "success"
    assert results["only"].result == 42
    assert results["only"].attempts == 1
    print("  PASSED\n")


def test_chain_dag():
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


def test_wide_dag():
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


def test_parallelism():
    """Verify that independent tasks actually run in parallel."""
    print("Test: Parallelism (independent tasks run concurrently)")
    scheduler = TaskScheduler(max_workers=4)

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

    assert elapsed < 1.5, f"Tasks took {elapsed:.2f}s — not parallel enough!"
    print(f"  Completed 4 x 0.5s tasks in {elapsed:.2f}s")
    print("  PASSED\n")


def test_retry_then_succeed():
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
    assert results["flaky"].attempts == 3
    print(f"  Task succeeded on attempt {results['flaky'].attempts}")
    print("  PASSED\n")


def test_duplicate_task_id():
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


def run_tests():
    print("=" * 60)
    print("Concurrent Task Scheduler — Hidden Tests")
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
    run_tests()
