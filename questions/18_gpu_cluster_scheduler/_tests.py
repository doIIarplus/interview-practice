"""Hidden tests for Question 18: GPU Cluster Scheduler
Run: python questions/18_gpu_cluster_scheduler/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import GPU, ClusterTopology, Job, GPUClusterScheduler


def test_topology_bandwidth():
    """Test bandwidth lookups based on GPU placement."""
    topo = ClusterTopology(num_nodes=4)
    scheduler = GPUClusterScheduler(topo)

    bw = scheduler.bandwidth_between(0, 0)
    assert bw == float('inf') or bw > 1e6, f"Same GPU should have ~infinite bandwidth, got {bw}"

    bw = scheduler.bandwidth_between(0, 7)
    assert bw == 600.0, f"Same node should be 600 GB/s, got {bw}"

    bw = scheduler.bandwidth_between(0, 8)
    assert bw == 50.0, f"Different nodes should be 50 GB/s, got {bw}"
    print("[PASS] test_topology_bandwidth")


def test_basic_placement():
    """Test placing a job with 3D parallelism."""
    topo = ClusterTopology(num_nodes=4)
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

    for tp_group in placement["tp_groups"]:
        assert len(tp_group) == 4, f"TP group should have 4 GPUs, got {len(tp_group)}"
        nodes = {gpu_id // topo.gpus_per_node for gpu_id in tp_group}
        assert len(nodes) == 1, f"TP group spans multiple nodes: {tp_group} -> nodes {nodes}"

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
        tensor_parallel_size=16,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        memory_per_gpu_gb=10.0,
    )
    placement = scheduler.place_job(job)
    assert placement is None, "TP=16 should fail on 8-GPU nodes"
    print("[PASS] test_tp_exceeds_node")


def test_insufficient_gpus():
    """Job requires more GPUs than available."""
    topo = ClusterTopology(num_nodes=2)
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
    topo = ClusterTopology(num_nodes=4)
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
    topo = ClusterTopology(num_nodes=4)
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
            new_gpus = set()
            for g in action["new_placement"]["tp_groups"]:
                new_gpus.update(g)
            assert failed_gpu not in new_gpus, "Failed GPU should not be in new placement"
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
    topo = ClusterTopology(num_nodes=1, gpus_per_node=8)
    scheduler = GPUClusterScheduler(topo)

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


def run_tests():
    print("Running GPU Cluster Scheduler tests...\n")
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


if __name__ == "__main__":
    run_tests()
