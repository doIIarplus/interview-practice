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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_topology_bandwidth():
    """Test bandwidth lookups based on GPU placement."""
    topo = ClusterTopology(num_nodes=4)
    scheduler = GPUClusterScheduler(topo)

    # Same GPU
    bw = scheduler.bandwidth_between(0, 0)
    assert bw == float('inf') or bw > 1e6, f"Same GPU should have ~infinite bandwidth, got {bw}"

    # Same node (GPUs 0 and 7 are both on node 0)
    bw = scheduler.bandwidth_between(0, 7)
    assert bw == 600.0, f"Same node should be 600 GB/s, got {bw}"

    # Different nodes (GPU 0 on node 0, GPU 8 on node 1)
    bw = scheduler.bandwidth_between(0, 8)
    assert bw == 50.0, f"Different nodes should be 50 GB/s, got {bw}"

    print("[PASS] test_topology_bandwidth")


def test_basic_placement():
    """Test placing a job with 3D parallelism."""
    topo = ClusterTopology(num_nodes=4)  # 32 GPUs
    scheduler = GPUClusterScheduler(topo)

    job = Job(
        job_id="llama-70b",
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        memory_per_gpu_gb=70.0,
    )
    assert job.total_gpus == 16

    placement = scheduler.place_job(job)
    assert placement is not None, "Should be able to place 16 GPUs on 32-GPU cluster"

    # Verify TP constraint: each TP group within one node
    for tp_group in placement["tp_groups"]:
        assert len(tp_group) == 4, f"TP group should have 4 GPUs, got {len(tp_group)}"
        nodes = {gpu_id // topo.gpus_per_node for gpu_id in tp_group}
        assert len(nodes) == 1, f"TP group spans multiple nodes: {tp_group} -> nodes {nodes}"

    # Verify total GPU count
    all_gpus = set()
    for tp_group in placement["tp_groups"]:
        all_gpus.update(tp_group)
    assert len(all_gpus) == 16, f"Expected 16 unique GPUs, got {len(all_gpus)}"

    print("[PASS] test_basic_placement")


def test_tp_exceeds_node():
    """TP size > GPUs per node should be impossible."""
    topo = ClusterTopology(num_nodes=4, gpus_per_node=8)
    scheduler = GPUClusterScheduler(topo)

    job = Job(
        job_id="too-large-tp",
        tensor_parallel_size=16,  # exceeds 8 GPUs per node
        pipeline_parallel_size=1,
        data_parallel_size=1,
        memory_per_gpu_gb=10.0,
    )
    placement = scheduler.place_job(job)
    assert placement is None, "TP=16 should fail on 8-GPU nodes"

    print("[PASS] test_tp_exceeds_node")


def test_insufficient_gpus():
    """Job requires more GPUs than available."""
    topo = ClusterTopology(num_nodes=2)  # 16 GPUs
    scheduler = GPUClusterScheduler(topo)

    job = Job(
        job_id="massive",
        tensor_parallel_size=8,
        pipeline_parallel_size=4,
        data_parallel_size=2,
        memory_per_gpu_gb=10.0,
    )
    assert job.total_gpus == 64
    placement = scheduler.place_job(job)
    assert placement is None, "64 GPUs cannot fit on 16-GPU cluster"

    print("[PASS] test_insufficient_gpus")


def test_multiple_jobs():
    """Place two jobs; second job should use remaining GPUs."""
    topo = ClusterTopology(num_nodes=4)  # 32 GPUs
    scheduler = GPUClusterScheduler(topo)

    job1 = Job(
        job_id="job1",
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        memory_per_gpu_gb=70.0,
    )
    placement1 = scheduler.place_job(job1)
    assert placement1 is not None

    job2 = Job(
        job_id="job2",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=4,
        memory_per_gpu_gb=30.0,
        num_params_billions=7.0,
    )
    placement2 = scheduler.place_job(job2)
    assert placement2 is not None, "Should fit in remaining 16 GPUs"

    # Verify no GPU overlap
    gpus1 = set()
    for g in placement1["tp_groups"]:
        gpus1.update(g)
    gpus2 = set()
    for g in placement2["tp_groups"]:
        gpus2.update(g)
    assert gpus1.isdisjoint(gpus2), "Jobs should not share GPUs"

    print("[PASS] test_multiple_jobs")


def test_communication_cost():
    """Test communication cost estimation."""
    topo = ClusterTopology(num_nodes=4)
    scheduler = GPUClusterScheduler(topo)

    job = Job(
        job_id="comm-test",
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        memory_per_gpu_gb=40.0,
        hidden_dim=4096,
        seq_len=2048,
        micro_batch_size=1,
        num_params_billions=70.0,
        dtype_bytes=2,
    )
    placement = scheduler.place_job(job)
    assert placement is not None

    cost = scheduler.estimate_communication_cost(job, placement)
    assert "tp_time_ms" in cost
    assert "pp_time_ms" in cost
    assert "dp_time_ms" in cost
    assert "total_ms" in cost
    assert "bottleneck" in cost
    assert cost["total_ms"] > 0
    assert cost["bottleneck"] in ("tp", "pp", "dp")

    print(f"[PASS] test_communication_cost")
    print(f"       TP: {cost['tp_time_ms']:.2f} ms")
    print(f"       PP: {cost['pp_time_ms']:.2f} ms")
    print(f"       DP: {cost['dp_time_ms']:.2f} ms")
    print(f"       Bottleneck: {cost['bottleneck']}")


def test_gpu_failure():
    """Test GPU failure handling and rescheduling."""
    topo = ClusterTopology(num_nodes=4)  # 32 GPUs
    scheduler = GPUClusterScheduler(topo)

    job = Job(
        job_id="resilient",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        memory_per_gpu_gb=40.0,
    )
    placement = scheduler.place_job(job)
    assert placement is not None

    # Find a GPU used by the job
    used_gpus = set()
    for g in placement["tp_groups"]:
        used_gpus.update(g)
    failed_gpu = list(used_gpus)[0]

    actions = scheduler.handle_gpu_failure(failed_gpu)
    assert len(actions) >= 1, "Should report at least one affected job"

    for action in actions:
        assert action["job_id"] == "resilient"
        assert action["action"] in ("rescheduled", "killed")
        if action["action"] == "rescheduled":
            # Verify new placement doesn't include failed GPU
            new_gpus = set()
            for g in action["new_placement"]["tp_groups"]:
                new_gpus.update(g)
            assert failed_gpu not in new_gpus, "Failed GPU should not be in new placement"
            # Verify TP constraint still holds
            for tp_group in action["new_placement"]["tp_groups"]:
                nodes = {gid // topo.gpus_per_node for gid in tp_group}
                assert len(nodes) == 1, "TP constraint violated after rescheduling"

    print("[PASS] test_gpu_failure")


def test_checkpoint_boundary():
    """Test checkpoint boundary selection."""
    topo = ClusterTopology(num_nodes=4)
    scheduler = GPUClusterScheduler(topo)

    job = Job(
        job_id="checkpoint-test",
        tensor_parallel_size=4,
        pipeline_parallel_size=4,
        data_parallel_size=1,
        memory_per_gpu_gb=40.0,
    )
    boundary = scheduler.find_checkpointable_boundary(job)
    assert boundary == 2, f"Midpoint of 4-stage pipeline should be 2, got {boundary}"

    job2 = Job(
        job_id="checkpoint-test-2",
        tensor_parallel_size=4,
        pipeline_parallel_size=8,
        data_parallel_size=1,
        memory_per_gpu_gb=20.0,
    )
    boundary2 = scheduler.find_checkpointable_boundary(job2)
    assert boundary2 == 4, f"Midpoint of 8-stage pipeline should be 4, got {boundary2}"

    print("[PASS] test_checkpoint_boundary")


def test_memory_constraint():
    """Test that memory constraints are enforced."""
    topo = ClusterTopology(num_nodes=1, gpus_per_node=8)  # 8 GPUs, 80GB each
    scheduler = GPUClusterScheduler(topo)

    # This job needs 85GB per GPU — exceeds A100 capacity
    job = Job(
        job_id="too-much-mem",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        memory_per_gpu_gb=85.0,
    )
    placement = scheduler.place_job(job)
    assert placement is None, "Should fail: 85GB > 80GB per GPU"

    print("[PASS] test_memory_constraint")


if __name__ == "__main__":
    test_topology_bandwidth()
    test_basic_placement()
    test_tp_exceeds_node()
    test_insufficient_gpus()
    test_multiple_jobs()
    test_communication_cost()
    test_gpu_failure()
    test_checkpoint_boundary()
    test_memory_constraint()
    print("\nAll tests passed!")
