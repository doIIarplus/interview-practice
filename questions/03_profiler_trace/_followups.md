# Follow-Up Questions: Profiler Trace Conversion

> **This file is hidden from the candidate.**

## Follow-Up 1: Denoising Profiler Data

**Question:** How would you handle "noisy" profiler data where a function might briefly disappear from a sample due to sampling artifacts? For example, if you see `["main", "foo"]`, then `["main"]`, then `["main", "foo"]` -- should that generate end/start events for `foo`, or should it be smoothed out?

**What to look for:**
- Recognizes that sampling profilers have inherent noise (the function was likely still running, the profiler just missed it)
- Suggests a threshold: only emit end/start events if the function is absent for N consecutive samples
- Discusses lookahead: before emitting an end event, check if the function reappears within a window
- Mentions the tradeoff between accuracy and latency (lookahead requires buffering)
- Bonus: discusses that this is effectively a low-pass filter on the binary signal "function present/absent"

**Red flags:**
- No awareness that sampling profilers are inherently imprecise
- Suggests that the raw data is always correct and should never be smoothed

---

## Follow-Up 2: Filtering by Sample Count

**Question:** Can you filter the trace to only show functions that appeared in at least N samples?

**What to look for:**
- First pass: count how many samples each (function, position) or just each function appears in
- Second pass: generate trace events only for functions meeting the threshold
- Discusses whether to count by unique function name or by (function, stack position)
- Considers that filtering might create "gaps" in the trace that need handling

**Bonus:**
- Suggests doing this as a post-processing step on the trace events rather than modifying the core algorithm
- Mentions that this is useful for filtering out very short-lived functions (noise) vs. long-running functions (signal)

---

## Follow-Up 3: Self Time vs Total Time

**Question:** How would you compute the "self time" vs "total time" for each function from the trace events?

**What to look for:**
- **Total time:** The number of samples between a function's start and end events (proportional to wall-clock time if sampling rate is constant)
- **Self time:** The number of samples where the function is at the *top* of the stack (no child function is executing above it)
- Understands that self time = total time - time spent in child calls
- Implementation: iterate through samples, for each sample the top-of-stack function gets +1 self time, all functions in the stack get +1 total time
- Mentions that self time is often more useful for optimization (tells you where CPU time is actually spent)

**Bonus:**
- Discusses that with the trace events, total time can be computed from start/end pairs
- Self time requires going back to the original samples (or tracking the "topmost" state)
- Mentions the relationship to flame graph widths

---

## Follow-Up 4: Estimating Wall-Clock Durations

**Question:** If the profiler sampled at a known frequency (e.g., 100Hz), how would you estimate wall-clock durations?

**What to look for:**
- Each sample represents `1/frequency` seconds (10ms at 100Hz)
- Total time for a function = (number of samples it appears in) * (1/frequency)
- Self time = (number of samples at top of stack) * (1/frequency)
- Discusses statistical uncertainty: with N samples, the error is roughly proportional to `1/sqrt(N)`
- Understands that sampling profilers give statistical estimates, not exact measurements
- Mentions that low-frequency events (functions called rarely or briefly) may not appear in any sample

**Bonus:**
- Discusses Nyquist-like considerations: the profiler can't detect events shorter than 2 * sample period
- Mentions that higher sampling rates give better accuracy but more overhead
- Discusses how to present confidence intervals on duration estimates

---

## Follow-Up 5: Visualization

**Question:** How would you visualize this data? (flame graphs, etc.)

**What to look for:**
- **Flame graphs:** x-axis is time (sample index), y-axis is stack depth, width of each box is the number of consecutive samples a function appears at that depth
- Understands the difference between:
  - **Flame chart** (timeline view): x-axis is time, shows exactly when functions ran
  - **Flame graph** (aggregated view): x-axis is not time but width represents total samples, stacks are merged
- Discusses how to generate SVG or use existing tools (Brendan Gregg's flamegraph.pl, speedscope, py-spy's built-in visualization)
- Mentions that the trace events we generate are essentially the data needed for a flame chart

**Bonus:**
- Mentions interactive visualization features: zoom, search, highlight
- Discusses icicle graphs (inverted flame graphs) as an alternative
- Mentions integration with Chrome's trace viewer (`chrome://tracing`) which accepts JSON trace event format
- Discusses how to convert our trace events to Chrome Trace Event Format:
  ```json
  {"name": "foo", "ph": "B", "ts": 0, "pid": 1, "tid": 1}
  {"name": "foo", "ph": "E", "ts": 100, "pid": 1, "tid": 1}
  ```

**Red flags:**
- Has never heard of flame graphs
- Suggests only text-based output (print statements) with no awareness of visualization tools
