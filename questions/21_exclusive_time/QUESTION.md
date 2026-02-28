# Question 21: Exclusive Time of Functions

*(Based on LeetCode 636)*

## Problem

On a **single-threaded** CPU, we execute a program with `n` functions. Each function has a unique ID from `0` to `n - 1`.

Function calls are managed with a **call stack**: when a function starts, it is pushed onto the stack; when it ends, it is popped. A function may call other functions before it ends (nesting). A function may also call itself recursively.

You are given a list of logs. Each log has the format:

```
"{function_id}:{start|end}:{timestamp}"
```

- `"0:start:3"` means function 0 **started** at the **beginning** of timestamp 3.
- `"1:end:5"` means function 1 **ended** at the **end** of timestamp 5.

**Important:** "end" timestamps are **inclusive**. A function that starts at timestamp 0 and ends at timestamp 0 has an exclusive time of **1 unit** (it occupied the entire unit of time from 0 to 1).

Return an array `result` where `result[i]` is the **exclusive time** of function `i`. Exclusive time is the time spent executing that function only -- it does **not** include time spent in functions it called.

Implement:

```python
def exclusive_time(n: int, logs: list[str]) -> list[int]:
```

---

## Examples

### Example 1: Basic Nesting

```
n = 2
logs = ["0:start:0", "1:start:2", "1:end:5", "0:end:6"]

Timeline (each unit is one timestamp):
  Time:  0  1  2  3  4  5  6
         [  0  [     1     ]  0]

Function 0: runs during [0,1] and [6,6] = 2 + 1 = 3
Function 1: runs during [2,5] = 4

Output: [3, 4]
```

### Example 2: Recursive Call

```
n = 1
logs = ["0:start:0", "0:start:2", "0:end:5", "0:end:6"]

Function 0 calls itself:
  Time:  0  1  2  3  4  5  6
         [ 0a  [    0b     ] 0a]

  Outer call (0a): runs during [0,1] and [6,6] = 2 + 1 = 3
  Inner call (0b): runs during [2,5] = 4
  Total for function 0: 3 + 4 = 7

Output: [7]
```

### Example 3: Sequential Calls

```
n = 2
logs = ["0:start:0", "0:end:0", "1:start:1", "1:end:1", "0:start:2", "0:end:2"]

Timeline:
  Time:  0  1  2
         [0] [1] [0]

Function 0: runs during [0,0] and [2,2] = 1 + 1 = 2
Function 1: runs during [1,1] = 1

Output: [2, 1]
```

### Example 4: Deeply Nested

```
n = 3
logs = ["0:start:0", "1:start:1", "2:start:2", "2:end:3", "1:end:4", "0:end:5"]

Timeline:
  Time:  0  1  2  3  4  5
         [0 [1 [ 2  ] 1] 0]

Function 0: runs during [0,0] and [5,5] = 1 + 1 = 2
Function 1: runs during [1,1] and [4,4] = 1 + 1 = 2
Function 2: runs during [2,3] = 2

Output: [2, 2, 2]
```

---

## Constraints

- `1 <= n <= 100`
- `1 <= logs.length <= 500`
- `0 <= function_id < n`
- `0 <= timestamp <= 10^9`
- Logs are in chronological order
- Each "start" has a corresponding "end"
- A function that starts first will end last (proper nesting, like valid parentheses)

---

## Key Insight

Think about how the stack changes at each log entry:

- **When a function starts:** The previously running function (top of stack) gets credit for the time since the last event. Then push the new function.
- **When a function ends:** It gets credit for the time since the last event (plus 1, because "end" is inclusive). Then pop it and set the "previous timestamp" to `timestamp + 1`.

---

## Starter Code

See `starter.py` for the function stub and test cases.
