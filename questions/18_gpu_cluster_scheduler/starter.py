"""
Question 18: GPU Cluster Scheduler

Place model-parallel workloads onto a GPU cluster with hierarchical topology
(NVLink within nodes, InfiniBand between nodes). Handle 3D parallelism
(tensor, pipeline, data) and GPU failures.

See QUESTION.md for full problem description.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GPU:
    """Represents a single GPU in the cluster."""
    gpu_id: int
    node_id: int
    memory_gb: int = 80           # e.g., A100 80GB
    memory_used_gb: float = 0.0
    is_healthy: bool = True

    @property
    def memory_free_gb(self) -> float:
        return self.memory_gb - self.memory_used_gb

    def __repr__(self) -> str:
        status = "OK" if self.is_healthy else "FAILED"
        return f"GPU({self.gpu_id}, node={self.node_id}, free={self.memory_free_gb:.0f}GB, {status})"


@dataclass
class ClusterTopology:
    """Describes the cluster's physical layout and bandwidth."""
    num_nodes: int
    gpus_per_node: int = 8
    intra_node_bw_gbps: float = 600.0   # NVLink bandwidth (GB/s)
    inter_node_bw_gbps: float = 50.0    # InfiniBand bandwidth (GB/s)

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node


@dataclass
class Job:
    """Describes a model-parallel training job."""
    job_id: str
    tensor_parallel_size: int       # GPUs per TP group
    pipeline_parallel_size: int     # number of pipeline stages
    data_parallel_size: int         # number of DP replicas
    memory_per_gpu_gb: float        # memory required per GPU

    # Communication parameters (for cost estimation)
    hidden_dim: int = 4096
    seq_len: int = 2048
    micro_batch_size: int = 1
    num_params_billions: float = 70.0
    dtype_bytes: int = 2            # FP16

    @property
    def total_gpus(self) -> int:
        """Total GPUs needed for this job."""
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size

    def __repr__(self) -> str:
        return (
            f"Job({self.job_id!r}, TP={self.tensor_parallel_size}, "
            f"PP={self.pipeline_parallel_size}, DP={self.data_parallel_size}, "
            f"total_gpus={self.total_gpus})"
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class GPUClusterScheduler:
    """
    Schedules model-parallel jobs onto a GPU cluster with
    topology-aware placement.
    """

    def __init__(self, topology: ClusterTopology) -> None:
        """
        Initialize the scheduler.

        Creates GPU objects for the entire cluster and sets up
        tracking structures for job placements.
        """
        self.topology = topology

        # TODO: Create GPU objects
        # self.gpus: dict[int, GPU] = {}  — maps gpu_id -> GPU
        # self.nodes: dict[int, list[int]] = {}  — maps node_id -> [gpu_ids]

        # TODO: Track job placements
        # self.job_placements: dict[str, dict] = {}  — maps job_id -> placement dict
        # self.gpu_to_job: dict[int, str] = {}  — maps gpu_id -> job_id

    # -------------------------------------------------------------------
    # Part 1: Topology
    # -------------------------------------------------------------------

    def bandwidth_between(self, gpu_a: int, gpu_b: int) -> float:
        """
        Return bandwidth in GB/s between two GPUs.

        - Same GPU: float('inf')
        - Same node (NVLink): intra_node_bw_gbps
        - Different nodes (InfiniBand): inter_node_bw_gbps
        """
        # TODO: Implement
        pass

    # -------------------------------------------------------------------
    # Part 2: Job placement
    # -------------------------------------------------------------------

    def place_job(self, job: Job) -> Optional[dict[str, list[list[int]]]]:
        """
        Place a model-parallel job onto the cluster.

        Constraints:
            - TP groups MUST be within the same node.
            - PP stages SHOULD prefer same node, can span nodes.
            - DP replicas can be anywhere.
            - Each GPU has enough free memory.

        Returns:
            {
                "tp_groups": [[gpu_ids], ...],
                "pp_stages": [[gpu_ids], ...],
                "dp_replicas": [[gpu_ids], ...],
            }
            or None if placement is impossible.

        The three views describe the same set of GPUs organized differently:
            - tp_groups: groups of GPUs that communicate via allreduce.
            - pp_stages: groups of GPUs at each pipeline stage.
            - dp_replicas: groups of GPUs that are DP replicas of each other.

        Total GPUs = tp * pp * dp.
        """
        # TODO: Implement
        #
        # Suggested approach:
        # 1. Validate: tp_size <= gpus_per_node, total_gpus <= available GPUs.
        # 2. For each DP replica:
        #    a. For each PP stage:
        #       - Find a node with tp_size free GPUs.
        #       - Assign those GPUs.
        # 3. Record the placement.
        pass

    def estimate_communication_cost(self, job: Job, placement: dict) -> dict:
        """
        Estimate communication time for one training step.

        Uses simplified model: time = data_size / bandwidth.

        TP allreduce: ring allreduce within each TP group.
            time = 2 * (tp_size - 1) / tp_size * data_size / bandwidth
            data_size = seq_len * hidden_dim * dtype_bytes

        PP send/recv: between adjacent pipeline stages.
            time = data_size / bandwidth
            data_size = micro_batch_size * seq_len * hidden_dim * dtype_bytes

        DP allreduce: across all DP replicas.
            time = 2 * (dp_size - 1) / dp_size * data_size / min_bandwidth
            data_size = num_params_billions * 1e9 * dtype_bytes

        Returns:
            {
                "tp_time_ms": float,
                "pp_time_ms": float,
                "dp_time_ms": float,
                "total_ms": float,
                "bottleneck": str,    # "tp", "pp", or "dp"
            }
        """
        # TODO: Implement
        pass

    # -------------------------------------------------------------------
    # Part 3: Fault tolerance
    # -------------------------------------------------------------------

    def handle_gpu_failure(self, failed_gpu_id: int) -> list[dict]:
        """
        Handle a GPU failure.

        1. Mark the GPU as unhealthy.
        2. Identify all affected jobs.
        3. For each affected job:
           a. Release all its GPUs.
           b. Attempt to re-place the job on healthy GPUs.
           c. If re-placement fails, the job is killed.

        Returns:
            [
                {
                    "job_id": str,
                    "action": "rescheduled" | "killed",
                    "old_placement": dict,
                    "new_placement": dict | None,
                }
            ]
        """
        # TODO: Implement
        pass

    def find_checkpointable_boundary(self, job: Job) -> int:
        """
        Find the optimal pipeline stage boundary for checkpointing.

        Returns the stage index (0-indexed) at the midpoint of the pipeline.
        For a pipeline with PP stages, this is PP // 2.
        """
        # TODO: Implement
        pass

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def cluster_stats(self) -> dict:
        """
        Return cluster-wide statistics.

        Returns:
            {
                "total_gpus": int,
                "healthy_gpus": int,
                "free_gpus": int,
                "active_jobs": int,
                "nodes": [
                    {"node_id": int, "free_gpus": int, "healthy_gpus": int},
                    ...
                ],
            }
        """
        # TODO: Implement
        pass


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    topo = ClusterTopology(num_nodes=4)
    scheduler = GPUClusterScheduler(topo)
    print(f"Total GPUs: {topo.total_gpus}")
    print(f"Bandwidth (same node): {scheduler.bandwidth_between(0, 1)} GB/s")
