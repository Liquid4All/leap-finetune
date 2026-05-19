import yaml

from leap_finetune.backends.kuberay_backend import _generate_rayjob_manifest


def test_manifest_has_correct_structure():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={
            "image": "test-image:latest",
            "gpu_count": 4,
            "memory": "64Gi",
            "cpu": 8,
        },
        output_dir="/outputs",
    )

    assert manifest["apiVersion"] == "ray.io/v1"
    assert manifest["kind"] == "RayJob"
    assert manifest["metadata"]["name"] == "test-job"
    assert manifest["spec"]["shutdownAfterJobFinishes"] is True


def test_manifest_gpu_allocation():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={"image": "img:latest", "gpu_count": 8},
        output_dir="/outputs",
    )

    head = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]
    assert head["rayStartParams"]["num-gpus"] == "8"

    container = head["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "8"


def test_manifest_custom_gpu_type():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={
            "image": "img:latest",
            "gpu_count": 2,
            "gpu_type": "amd.com/gpu",
        },
        output_dir="/outputs",
    )

    container = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
        "containers"
    ][0]
    assert "amd.com/gpu" in container["resources"]["limits"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]


def test_manifest_env_vars():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={
            "image": "img:latest",
            "env": {"HF_TOKEN": "test123", "WANDB_API_KEY": "key456"},
        },
        output_dir="/outputs",
    )

    container = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
        "containers"
    ][0]
    env_names = {e["name"]: e["value"] for e in container["env"]}
    assert env_names["HF_TOKEN"] == "test123"
    assert env_names["WANDB_API_KEY"] == "key456"
    assert env_names["OUTPUT_DIR"] == "/outputs"


def test_manifest_output_pvc():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={"image": "img:latest", "output_pvc": "my-pvc"},
        output_dir="/outputs",
    )

    pod_spec = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    vol_names = [v["name"] for v in pod_spec["volumes"]]
    assert "output-vol" in vol_names
    assert "dshm" in vol_names

    pvc_vol = next(v for v in pod_spec["volumes"] if v["name"] == "output-vol")
    assert pvc_vol["persistentVolumeClaim"]["claimName"] == "my-pvc"


def test_manifest_no_pvc():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={"image": "img:latest"},
        output_dir="/outputs",
    )

    pod_spec = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    vol_names = [v["name"] for v in pod_spec["volumes"]]
    assert "output-vol" not in vol_names


def test_manifest_node_selector_and_tolerations():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={
            "image": "img:latest",
            "node_selector": {"gpu-type": "a100"},
            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
        },
        output_dir="/outputs",
    )

    pod_spec = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    assert pod_spec["nodeSelector"] == {"gpu-type": "a100"}
    assert pod_spec["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists"}]


def test_manifest_shared_memory():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={"image": "img:latest"},
        output_dir="/outputs",
    )

    pod_spec = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    dshm = next(v for v in pod_spec["volumes"] if v["name"] == "dshm")
    assert dshm["emptyDir"]["medium"] == "Memory"


def test_manifest_is_valid_yaml():
    manifest = _generate_rayjob_manifest(
        job_name="test-job",
        configmap_name="test-config",
        kuberay_cfg={"image": "img:latest", "gpu_count": 4},
        output_dir="/outputs",
    )

    serialized = yaml.dump(manifest)
    parsed = yaml.safe_load(serialized)
    assert parsed["kind"] == "RayJob"
