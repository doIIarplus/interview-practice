# Question 04: Distributed Mode Finding

## Problem Statement

You have a large dataset distributed across **N nodes** (default N=10). Each node holds a
portion of the data. Your task is to find the **mode** (the most frequently occurring value)
across **all** nodes combined.

You have access to three communication/computation primitives:

```python
read_local_data(node_id: int) -> list[int]
```
Reads the data stored locally on the given node. Simulated cost: **10 bytes/sec** throughput.

```python
send(from_node: int, to_node: int, data: bytes)
```
Sends a byte payload from one node to another. Simulated cost: **1 byte/sec** throughput
(10x slower than local reads).

```python
barrier()
```
Synchronizes all nodes. Blocks until every node has reached this point.

## Constraints

- **Total data**: approximately 10,000 integers spread across 10 nodes (~1,000 per node).
- **Value range**: each integer is in the range `[0, 1000)`.
- **Network is slow**: sending data between nodes is 10x slower than reading local data.
- **Goal**: minimize total wall-clock time.
- **Tie-breaking**: if multiple values share the highest frequency, return the **smallest** value.

## Your Task

Implement the function:

```python
def find_mode(cluster: Cluster) -> int:
    """
    Find the mode (most frequent value) across all nodes in the cluster.

    Args:
        cluster: A Cluster instance that provides access to N nodes,
                 each holding a portion of the distributed dataset.

    Returns:
        The most frequently occurring integer across all nodes.
        If there is a tie, return the smallest value among the tied values.
    """
    pass
```

The `Cluster` class provides:
- `cluster.num_nodes` -- number of nodes (default 10)
- `cluster.read_local_data(node_id)` -- read data on a node
- `cluster.send(from_node, to_node, data)` -- send bytes between nodes
- `cluster.barrier()` -- synchronize all nodes
- `cluster.get_received(node_id)` -- retrieve data that was sent to a node
- `cluster.total_time` -- the simulated wall-clock time consumed so far

## Example

```python
# Suppose data across 3 nodes is:
# Node 0: [1, 2, 3, 2, 2]
# Node 1: [3, 3, 3, 4, 5]
# Node 2: [2, 2, 3, 3, 3]
#
# Global frequencies: {1:1, 2:4, 3:6, 4:1, 5:1}
# Mode = 3
#
# A naive approach sends ALL raw data to one node:
#   15 integers * 4 bytes each = 60 bytes over the network
#   At 1 byte/sec, that's 60 simulated seconds of network time.
#
# A smarter approach sends local frequency COUNTS to one node:
#   Each node sends at most 1000 count entries (far fewer in practice).
#   This dramatically reduces network traffic.
```

## Hints

- Think about what you can compute locally before communicating.
- Consider what the minimum amount of data you need to send over the network is.
- The value range `[0, 1000)` is a useful constraint.

## Getting Started

See `starter.py` for the `Cluster` simulation framework and function signature.
