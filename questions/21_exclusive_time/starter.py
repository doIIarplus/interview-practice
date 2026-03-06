"""
Question 21: Exclusive Time of Functions (LeetCode 636)

Given n functions and a list of call logs, compute the exclusive time of
each function. Exclusive time is the time a function spent executing,
NOT including time spent in functions it called.

Key detail: "end" timestamps are inclusive. A function starting at 0 and
ending at 0 has an exclusive time of 1 unit.
"""


def exclusive_time(n: int, logs: list[str]) -> list[int]:
    """
    Compute the exclusive time of each function.

    Args:
        n: Number of functions (IDs from 0 to n-1).
        logs: List of log strings in format "{func_id}:{start|end}:{timestamp}".
              Logs are in chronological order.

    Returns:
        List of length n where result[i] is the exclusive time of function i.

    Approach hint:
        Use a stack to track the currently executing function.
        - On "start": credit the current top-of-stack function for elapsed time,
          then push the new function.
        - On "end": credit the ending function for elapsed time (+1 because
          end is inclusive), pop it, and set prev_timestamp = timestamp + 1.
    """
    pass


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    n = 2
    logs = ["0:start:0", "1:start:2", "1:end:5", "0:end:6"]
    result = exclusive_time(n, logs)
    print(f"Exclusive time: {result}")
