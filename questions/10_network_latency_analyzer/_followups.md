# Follow-up Questions: Network Latency Analyzer

---

## 1. How would you handle spans from different machines with unsynchronized clocks?

**What we're looking for:**
- Awareness that clock skew is a real problem in distributed systems
- Knowledge of NTP and its limitations (~1ms accuracy at best)
- Mentions that some tracing systems use relative timestamps or causal ordering
- Discusses techniques: logical clocks (Lamport timestamps), hybrid logical clocks
- May mention that Google's Spanner uses TrueTime with GPS and atomic clocks
- Practical approach: use the parent-child relationship (causal ordering) rather
  than relying on absolute timestamps for ordering; use clock skew correction
  heuristics (e.g., if a child starts before its parent, adjust)

**Strong answer includes:**
- Understanding that self-time calculation becomes unreliable with skew
- Suggestion to detect and flag inconsistent timestamps
- Reference to OpenTelemetry's clock correction mechanisms

---

## 2. What if traces are incomplete (some spans are missing)?

**What we're looking for:**
- Recognition that this is common in production (sampling, failures, timeouts)
- For tree building: orphan spans (parent ID exists but parent span is missing)
- Options: attach orphans to a synthetic root, skip them, or flag as incomplete
- For analysis: missing spans mean self-time is overestimated for parents
  (because the child's time isn't subtracted)
- Statistical analysis should be robust to partial data

**Strong answer includes:**
- Mentions tail-based sampling vs head-based sampling
- Discusses how partial traces affect critical path analysis
- Proposes confidence metrics ("this trace is 80% complete")

---

## 3. How would you build a real-time anomaly detection system for this data?

**What we're looking for:**
- Streaming statistics: Welford's online algorithm for mean and variance
- Sliding window approach (e.g., last 5 minutes of data)
- Exponentially weighted moving average (EWMA) for adapting to drift
- Alerting thresholds and hysteresis to avoid alert fatigue
- Architecture: stream processor (Kafka Streams, Flink) consuming trace data

**Strong answer includes:**
- Discusses the difference between detecting individual anomalous spans vs
  detecting a system-wide latency shift
- Mentions that you need both: per-span anomaly detection AND aggregate
  percentile monitoring (p50, p99, p999)
- Discusses seasonality (latency patterns vary by time of day)

---

## 4. How would you aggregate this data across millions of traces efficiently?

**What we're looking for:**
- Sampling strategies: head-based (random), tail-based (keep interesting traces)
- Sketching algorithms: t-digest or HDR histogram for percentile estimation
- Pre-aggregation: aggregate per (service, operation) at the collection point
  rather than storing all individual spans
- Storage: columnar stores for analytical queries (ClickHouse, etc.)
- Approximate algorithms: HyperLogLog for unique counts, Count-Min Sketch

**Strong answer includes:**
- Specific discussion of t-digest for latency percentiles
- Understanding of the storage vs accuracy trade-off
- Mentions that 1% sampling gives you good p50/p99 but bad p99.9 estimates

---

## 5. What existing tools solve this problem?

**Expected knowledge:**
- **Jaeger**: Open-source distributed tracing, originally from Uber
- **Zipkin**: Open-source tracing from Twitter
- **OpenTelemetry**: Vendor-neutral observability framework (merges OpenTracing + OpenCensus)
- **Datadog APM**, **New Relic**, **Honeycomb**: Commercial solutions
- **AWS X-Ray**: Cloud-native tracing

**Strong answer includes:**
- Understanding of the OpenTelemetry data model (traces, spans, attributes)
- Knowledge of the collector architecture (agent -> collector -> backend)
- Discusses trade-offs between self-hosted (Jaeger) and SaaS (Datadog)
- Mentions that OpenTelemetry is becoming the industry standard

---

## 6. How would you correlate latency spikes with resource metrics?

**What we're looking for:**
- Join trace data with infrastructure metrics (CPU, memory, GPU utilization, network)
- Time-based correlation: when latency spikes, what metrics also spiked?
- Tags/labels: add resource utilization as span attributes
- Statistical correlation: Pearson/Spearman correlation between latency and metrics
- Causation vs correlation: a GPU utilization spike might cause latency, or both
  might be caused by a traffic spike

**Strong answer includes:**
- Mentions exemplars: linking specific metric data points to traces
- Discusses the USE method (Utilization, Saturation, Errors) for resource analysis
- For ML workloads: GPU memory pressure, CUDA stream contention, batch size effects
- Proposes automated root cause analysis: when latency spikes, automatically
  check correlated metrics and present likely causes
