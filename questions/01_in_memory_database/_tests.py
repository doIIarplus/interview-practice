"""Hidden tests for Question 01: In-Memory Database
Run: python questions/01_in_memory_database/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import InMemoryDB


def test_level1_basic():
    """Test Level 1: basic set/get/delete operations."""
    db = InMemoryDB()
    assert db.set("user1", "name", "Alice") == ""
    assert db.set("user1", "age", "30") == ""
    assert db.get("user1", "name") == "Alice"
    assert db.get("user1", "email") == ""
    assert db.delete("user1", "name") == "true"
    assert db.get("user1", "name") == ""
    assert db.delete("user1", "name") == "false"
    print("[PASS] test_level1_basic")


def test_level2_scan():
    """Test Level 2: scan and scan_by_prefix."""
    db = InMemoryDB()
    db.set("user1", "name", "Alice")
    db.set("user1", "age", "30")
    db.set("user1", "address", "123 Main St")
    assert db.scan("user1") == "address(123 Main St), age(30), name(Alice)"
    assert db.scan_by_prefix("user1", "a") == "address(123 Main St), age(30)"
    assert db.scan_by_prefix("user1", "z") == ""
    print("[PASS] test_level2_scan")


def test_level3_timestamps():
    """Test Level 3: timestamped operations and TTL."""
    db = InMemoryDB()
    db.set_at("user1", "name", "Alice", 1)
    db.set_at("user1", "name", "Bob", 5)
    assert db.get_at("user1", "name", 3) == "Alice"
    assert db.get_at("user1", "name", 5) == "Bob"
    db.set_at_with_ttl("user1", "token", "abc", 20, 10)
    assert db.get_at("user1", "token", 29) == "abc"
    assert db.get_at("user1", "token", 30) == ""
    print("[PASS] test_level3_timestamps")


def test_level4_compaction():
    """Test Level 4: compaction."""
    db = InMemoryDB()
    db.set_at("user1", "name", "Alice", 1)
    db.set_at("user1", "name", "Bob", 5)
    db.set_at("user1", "name", "Charlie", 10)
    assert db.compact(7) == "1"
    print("[PASS] test_level4_compaction")


def run_tests():
    print("=" * 60)
    print("In-Memory Database — Hidden Tests")
    print("=" * 60 + "\n")

    test_level1_basic()
    test_level2_scan()
    test_level3_timestamps()
    test_level4_compaction()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
