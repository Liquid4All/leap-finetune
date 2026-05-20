from leap_finetune.backends.kuberay_backend import (
    _build_submission_config,
    _generate_rayjob_manifest,
    _resolve_kuberay_cluster_spec,
)


def test_build_submission_config_injects_external_ray_cluster_settings():
    config_dict = {
        "model_name": "LFM2-1.2B",
        "training_type": "sft",
        "dataset": {"path": "HuggingFaceTB/smoltalk", "type": "sft"},
        "training_config": {"output_dir": "/tmp/original"},
        "kuberay": {"image": "example.com/leap:latest"},
    }
    kuberay_cfg = {
        "image": "example.com/leap:latest",
        "worker_replicas": 2,
        "gpus_per_worker": 4,
        "output_dir": "/outputs",
    }

    submission_config, output_dir = _build_submission_config(config_dict, kuberay_cfg)

    assert "kuberay" not in submission_config
    assert output_dir == "/outputs"
    assert submission_config["training_config"]["output_dir"] == "/outputs"
    assert submission_config["ray"]["address"] == "auto"
    assert submission_config["ray"]["num_workers"] == 8


def test_resolve_kuberay_cluster_spec_supports_legacy_gpu_count():
    spec = _resolve_kuberay_cluster_spec({"gpu_count": 4})

    assert spec["worker_replicas"] == 4
    assert spec["gpus_per_worker"] == 1
    assert spec["total_gpus"] == 4


def test_generate_rayjob_manifest_creates_worker_group_pool():
    manifest = _generate_rayjob_manifest(
        job_name="lf-test",
        configmap_name="lf-test-config",
        kuberay_cfg={
            "image": "example.com/leap:latest",
            "namespace": "default",
            "worker_replicas": 2,
            "gpus_per_worker": 4,
            "head_cpu": 2,
            "head_memory": "8Gi",
            "worker_cpu": 12,
            "worker_memory": "80Gi",
            "output_pvc": "training-outputs",
        },
        output_dir="/outputs",
    )

    cluster_spec = manifest["spec"]["rayClusterSpec"]
    head_spec = cluster_spec["headGroupSpec"]
    worker_spec = cluster_spec["workerGroupSpecs"][0]

    assert head_spec["serviceType"] == "ClusterIP"
    assert head_spec["rayStartParams"]["dashboard-host"] == "0.0.0.0"
    assert (
        head_spec["template"]["spec"]["containers"][0]["resources"]["limits"]["cpu"]
        == "2"
    )
    assert (
        "nvidia.com/gpu"
        not in head_spec["template"]["spec"]["containers"][0]["resources"]["limits"]
    )

    assert worker_spec["replicas"] == 2
    assert worker_spec["minReplicas"] == 2
    assert worker_spec["maxReplicas"] == 2
    assert worker_spec["rayStartParams"]["num-gpus"] == "4"
    assert (
        worker_spec["template"]["spec"]["containers"][0]["resources"]["limits"][
            "nvidia.com/gpu"
        ]
        == "4"
    )
    assert (
        worker_spec["template"]["spec"]["containers"][0]["resources"]["limits"][
            "memory"
        ]
        == "80Gi"
    )
