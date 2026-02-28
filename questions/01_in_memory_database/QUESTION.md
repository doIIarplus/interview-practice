# Question 01: In-Memory Database

## Overview

Implement an in-memory database that supports basic key-field-value operations, with progressive levels of complexity. Each level builds on the previous one.

---

## Level 1: Basic Operations

Implement a class `InMemoryDB` that supports the following operations:

- **`SET(key, field, value)`** -- Sets the value of `field` within the record identified by `key`. If the key or field does not exist, create it. Returns nothing (`""`).
- **`GET(key, field)`** -- Returns the value of `field` within the record identified by `key`. If the key or field does not exist, return `""`.
- **`DELETE(key, field)`** -- Deletes `field` from the record identified by `key`. Returns `"true"` if the field existed and was deleted, `"false"` otherwise. If deleting the last field in a key, the key should also be removed.

### Example (Level 1)

```python
db = InMemoryDB()

db.set("user1", "name", "Alice")        # => ""
db.set("user1", "age", "30")            # => ""
db.get("user1", "name")                 # => "Alice"
db.get("user1", "email")                # => ""
db.get("user2", "name")                 # => ""
db.delete("user1", "name")              # => "true"
db.get("user1", "name")                 # => ""
db.delete("user1", "name")              # => "false"
```

---

## Level 2: Scan Operations

Add the following operations to your `InMemoryDB` class:

- **`SCAN(key)`** -- Returns all field-value pairs for the given `key`, sorted alphabetically by field name. Each pair should be formatted as `"field(value)"`. Returns an empty string if the key does not exist.
- **`SCAN_BY_PREFIX(key, prefix)`** -- Returns all field-value pairs for the given `key` where the field name starts with `prefix`, sorted alphabetically by field name. Each pair should be formatted as `"field(value)"`. Returns an empty string if no matching fields exist.

### Example (Level 2)

```python
db = InMemoryDB()

db.set("user1", "name", "Alice")
db.set("user1", "age", "30")
db.set("user1", "address", "123 Main St")

db.scan("user1")
# => "address(123 Main St), age(30), name(Alice)"

db.scan_by_prefix("user1", "a")
# => "address(123 Main St), age(30)"

db.scan_by_prefix("user1", "name")
# => "name(Alice)"

db.scan_by_prefix("user1", "z")
# => ""

db.scan("nonexistent")
# => ""
```

---

## Level 3: Timestamped Operations

Add support for timestamped operations. All timestamps are positive integers representing seconds. Operations at earlier timestamps always happen before operations at later timestamps. Timestamps are guaranteed to be provided in non-decreasing order across calls.

- **`SET_AT(key, field, value, timestamp)`** -- Sets the value at the given timestamp. Returns `""`.
- **`GET_AT(key, field, timestamp)`** -- Returns the value of the field at the given timestamp. If the field did not exist at that time, return `""`. The value returned should be the most recent value set at or before the given timestamp.
- **`DELETE_AT(key, field, timestamp)`** -- Deletes the field at the given timestamp. Returns `"true"` if the field had a value at that timestamp, `"false"` otherwise.
- **`SCAN_AT(key, timestamp)`** -- Like `SCAN`, but returns the state of the record as of the given timestamp.
- **`SCAN_BY_PREFIX_AT(key, prefix, timestamp)`** -- Like `SCAN_BY_PREFIX`, but returns state as of the given timestamp.
- **`SET_AT_WITH_TTL(key, field, value, timestamp, ttl)`** -- Sets the value at the given timestamp, but the value expires `ttl` seconds after the timestamp. A value set at timestamp `t` with TTL `ttl` is valid from `t` to `t + ttl - 1` (inclusive). Returns `""`.

The original Level 1 and Level 2 operations should still work. You may treat them as having an implicit "latest" timestamp.

### Example (Level 3)

```python
db = InMemoryDB()

db.set_at("user1", "name", "Alice", 1)          # => ""
db.set_at("user1", "name", "Bob", 5)            # => ""
db.get_at("user1", "name", 3)                   # => "Alice"
db.get_at("user1", "name", 5)                   # => "Bob"
db.get_at("user1", "name", 0)                   # => ""

db.delete_at("user1", "name", 10)               # => "true"
db.get_at("user1", "name", 9)                   # => "Bob"
db.get_at("user1", "name", 10)                  # => ""

db.set_at_with_ttl("user1", "token", "abc", 20, 10)  # => ""
db.get_at("user1", "token", 20)                 # => "abc"
db.get_at("user1", "token", 29)                 # => "abc"
db.get_at("user1", "token", 30)                 # => ""  (expired)

db.set_at("user1", "age", "30", 1)
db.set_at("user1", "email", "a@b.com", 2)
db.scan_at("user1", 1)
# => "age(30), name(Alice)"

db.scan_at("user1", 2)
# => "age(30), email(a@b.com), name(Alice)"
```

---

## Level 4: Compaction

Add a compaction operation that cleans up expired and deleted entries:

- **`COMPACT(timestamp)`** -- Removes all entries from the database's history that are no longer needed as of the given timestamp. Specifically, for each field, if there are multiple historical values where the timestamp is at or before the compact timestamp, keep only the most recent one. Remove any entries that have expired (TTL) or been deleted at or before the compact timestamp. Returns a string representation of the number of entries removed.

### Example (Level 4)

```python
db = InMemoryDB()

db.set_at("user1", "name", "Alice", 1)
db.set_at("user1", "name", "Bob", 5)
db.set_at("user1", "name", "Charlie", 10)

# Compacting at timestamp 7: the entries at t=1 ("Alice") is superseded
# by t=5 ("Bob"), which is the most recent at or before t=7.
# The entry at t=10 ("Charlie") is in the future, so it stays.
# Removed: 1 entry (Alice at t=1)
db.compact(7)                                    # => "1"

# After compaction, historical queries at t>=5 still work
db.get_at("user1", "name", 5)                   # => "Bob"
db.get_at("user1", "name", 10)                  # => "Charlie"

db.set_at_with_ttl("user1", "token", "xyz", 20, 5)  # expires at t=25
db.compact(30)
# The token expired at t=25, which is before t=30, so it is removed.
# "Bob" at t=5 is superseded by "Charlie" at t=10, and t=10 <= 30,
# so "Bob" is removed. "Charlie" is the most recent at or before t=30, kept.
# Removed: 2 entries (Bob, token)
# => "2"
```

---

## Constraints

- Keys, fields, and values are all non-empty strings containing no spaces.
- Timestamps are positive integers.
- Operations are called in non-decreasing timestamp order.
- All return values are strings.
