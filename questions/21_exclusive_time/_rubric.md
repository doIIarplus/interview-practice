# Rubric: Exclusive Time of Functions (Question 21)

## Grading Criteria

### 1. Correct Log Parsing (10%)

- Correctly splits each log string into `function_id`, `type` (start/end), and `timestamp`
- Converts function_id and timestamp to integers
- Handles the format cleanly

**Full marks:** Clean parsing, proper type conversion.
**Partial:** Works but messy (e.g., hardcoded indices instead of split).
**Minimal:** Incorrect parsing.

```python
func_id, event_type, timestamp = log.split(":")
func_id = int(func_id)
timestamp = int(timestamp)
```

---

### 2. Stack-Based Approach Tracking Current Function (25%)

- Uses a stack to track which function is currently executing
- Pushes on "start", pops on "end"
- Uses the stack to correctly attribute time to the right function

**Full marks:** Clear stack usage, correct push/pop logic. Understands that the top of the stack is the currently executing function.
**Partial:** Uses a stack but has logic errors in attribution.
**Minimal:** No stack, tries to use other approaches that don't handle nesting.

---

### 3. Correct Handling of "end" Being Inclusive (20%)

- When processing an "end" event at timestamp T, the function occupied time through the **end** of T (i.e., through T+1 in exclusive terms).
- Must add `timestamp - prev + 1` for end events.
- Must set `prev_timestamp = timestamp + 1` after an end event.

This is the trickiest part of the problem. Many candidates get the off-by-one wrong.

**Full marks:** Correctly handles the inclusive end with `+1`. Sets prev correctly after end.
**Partial:** Gets the formula mostly right but has an off-by-one in some cases.
**Minimal:** Doesn't account for inclusive end at all.

Key formulas:
```python
if event_type == "start":
    # Credit current top-of-stack for time elapsed
    if stack:
        result[stack[-1]] += timestamp - prev_timestamp
    prev_timestamp = timestamp

elif event_type == "end":
    # Credit ending function (inclusive end = +1)
    result[stack[-1]] += timestamp - prev_timestamp + 1
    prev_timestamp = timestamp + 1
```

---

### 4. Correct Exclusive Time: Subtracting Child Time from Parent (20%)

- When a child function starts, the parent stops accumulating time
- When the child ends, the parent resumes accumulating time
- The stack naturally handles this: we add time to `stack[-1]` between events

**Full marks:** Correctly handles parent-child time attribution through the stack. No double-counting.
**Partial:** Generally correct but may double-count or miss time in edge cases.
**Minimal:** Doesn't handle nesting, calculates total time instead of exclusive.

---

### 5. Handling Recursive Calls (15%)

- Same function ID can appear multiple times on the stack
- Each invocation is treated independently (its own start and end)
- Time is correctly accumulated -- all invocations add to the same `result[func_id]`

**Full marks:** Recursive calls work correctly, accumulates time across all invocations.
**Partial:** Works for simple cases but fails on deep recursion.
**Minimal:** Crashes or gives wrong results with recursive calls.

---

### 6. Edge Cases and Code Clarity (10%)

- Single function (n=1)
- Function that starts and ends at the same timestamp (exclusive time = 1)
- Large timestamps
- Multiple sequential calls at the top level
- Clean, readable code

**Full marks:** All edge cases pass, code is clean and well-organized.
**Partial:** Most edge cases work, code is acceptable.
**Minimal:** Multiple edge case failures, messy code.

---

## Reference Solution

```python
def exclusive_time(n: int, logs: list[str]) -> list[int]:
    result = [0] * n
    stack = []
    prev_timestamp = 0

    for log in logs:
        parts = log.split(":")
        func_id = int(parts[0])
        event_type = parts[1]
        timestamp = int(parts[2])

        if event_type == "start":
            # Credit time to current top-of-stack function
            if stack:
                result[stack[-1]] += timestamp - prev_timestamp
            stack.append(func_id)
            prev_timestamp = timestamp
        else:  # "end"
            # Credit time to the ending function (inclusive, so +1)
            result[stack[-1]] += timestamp - prev_timestamp + 1
            stack.pop()
            prev_timestamp = timestamp + 1

    return result
```

### Walkthrough of Example 1:

```
n=2, logs=["0:start:0", "1:start:2", "1:end:5", "0:end:6"]

Initial: result=[0,0], stack=[], prev=0

Log "0:start:0":
  stack empty, no credit to give
  push 0 -> stack=[0], prev=0

Log "1:start:2":
  Credit stack[-1]=0: result[0] += 2-0 = 2 -> result=[2,0]
  push 1 -> stack=[0,1], prev=2

Log "1:end:5":
  Credit stack[-1]=1: result[1] += 5-2+1 = 4 -> result=[2,4]
  pop -> stack=[0], prev=5+1=6

Log "0:end:6":
  Credit stack[-1]=0: result[0] += 6-6+1 = 1 -> result=[3,4]
  pop -> stack=[], prev=7

Result: [3, 4] ✓
```

---

## Common Mistakes

1. **Off-by-one on "end":** Forgetting that `end:5` means "through the end of timestamp 5", so it includes one more unit than you might think.
2. **Not updating prev_timestamp correctly after "end":** Must set to `timestamp + 1`, not `timestamp`.
3. **Confusing exclusive vs. inclusive time:** Adding child time to parent instead of only attributing time when the function is on top of the stack.
4. **Not handling recursion:** Assuming each function ID appears at most once on the stack.

## Red Flags

- Tries to compute time as `end - start` for each function (ignores nesting)
- No stack usage
- Modifying the wrong index in the result array
- Not handling the inclusive end semantics

## Green Flags

- Immediately recognizes this as a stack problem
- Correctly identifies the "end is inclusive" subtlety
- Traces through an example to verify the +1 logic
- Clean parsing with split(":")
- Mentions the connection to profiling/tracing tools
