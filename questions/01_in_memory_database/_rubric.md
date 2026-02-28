# Rubric: In-Memory Database

> **This file is hidden from the candidate.**

## Scoring Breakdown

### Level 1: Basic Operations (25%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Correct SET/GET/DELETE | 10 | All three operations work correctly for basic cases |
| Missing key/field handling | 5 | GET returns "" for nonexistent keys/fields; DELETE returns "false" |
| Key cleanup on last field delete | 5 | When the last field in a key is deleted, the key itself is removed |
| Clean data structure choice | 5 | Uses nested dict (`dict[str, dict[str, str]]`) or equivalent |

**Red flags:**
- Using a flat dict with composite keys (e.g., `"key:field"`) -- works but fragile for scan operations later
- Not handling the "delete last field removes key" requirement
- Returning `None` instead of `""`

### Level 2: Scan Operations (25%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Correct SCAN implementation | 8 | Returns all fields sorted alphabetically |
| Correct SCAN_BY_PREFIX | 8 | Prefix matching works correctly |
| Output formatting | 5 | Correct `"field(value)"` format with comma-space separation |
| Edge cases | 4 | Empty results return "", nonexistent keys return "" |

**Green flags:**
- Uses `sorted()` on dict keys
- Clean helper function for formatting field-value pairs
- Considers that prefix="" should match everything

**Red flags:**
- Sorting by value instead of field name
- Using regex for simple prefix matching (`str.startswith()` is preferred)
- Over-engineering with a trie for prefix matching at this stage

### Level 3: Timestamped Operations (30%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Correct data structure for history | 8 | Uses sorted list of (timestamp, value) or similar |
| GET_AT with binary search or linear scan | 6 | Correctly finds most recent value at or before timestamp |
| DELETE_AT correctness | 5 | Properly records deletion in history |
| TTL implementation | 6 | SET_AT_WITH_TTL correctly expires values |
| SCAN_AT and SCAN_BY_PREFIX_AT | 3 | Correctly compose timestamp lookup with scan logic |
| Backward compatibility | 2 | Level 1/2 operations still work |

**Green flags:**
- Uses `bisect` module for efficient timestamp lookup
- Stores history as sorted list of `(timestamp, value, expiry)` tuples
- Clean separation between "find value at time T" and the operations that use it
- Uses a sentinel value (e.g., `None`) for deletions in the history

**Red flags:**
- Storing separate histories for each operation type instead of a unified timeline
- Not handling the TTL boundary correctly (value at `t + ttl - 1` should exist, at `t + ttl` should not)
- Rebuilding entire state for each GET_AT call instead of doing a lookup
- Breaking Level 1/2 operations

### Level 4: Compaction (20%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Correct identification of removable entries | 8 | Superseded entries before compact timestamp are removed |
| Correct removal of expired/deleted entries | 6 | TTL-expired and deleted entries are cleaned up |
| Accurate count | 4 | Returns the correct number of removed entries |
| Does not break future queries | 2 | Queries after compaction timestamp still work correctly |

**Green flags:**
- Iterates through all keys and fields, filters history lists in place
- Clear logic: "for entries at or before T, keep only the most recent; if that most recent is deleted/expired, remove it too"
- Good test coverage of edge cases

**Red flags:**
- Removes entries that are still needed for future timestamp queries
- Off-by-one errors in what counts as "at or before"
- Modifying data structures while iterating over them unsafely

---

## Overall Assessment Criteria

| Category | Look For |
|----------|----------|
| **Incremental design** | Does the candidate build clean abstractions that extend naturally from Level 1 to Level 4? Or do they have to refactor heavily? |
| **Not over-engineering** | At Level 1, a simple nested dict is ideal. Using a database engine or complex class hierarchy is a red flag. |
| **Edge case awareness** | Do they ask about or handle: empty strings, double deletes, GET on deleted fields, TTL of 0, etc.? |
| **Code quality** | Meaningful variable names, type hints, no dead code, DRY principles. |
| **Testing mindset** | Do they run the smoke tests? Do they add their own test cases? |

## Time Expectations

- Level 1: 5-10 minutes
- Level 2: 5-10 minutes
- Level 3: 15-20 minutes
- Level 4: 10-15 minutes
- Total: 35-55 minutes

Candidates who reach Level 4 with clean code and correct behavior are strong hires. Completing Levels 1-3 cleanly is a passing bar.
