import sys
from datetime import datetime

import yaml


def check_and_handle_kuberay(config_path_arg: str) -> bool:
    if not config_path_arg:
        return False

    try:
        from leap_finetune.utils.config_resolver import resolve_config_path

        config_path = resolve_config_path(config_path_arg)
    except Exception:
        return False

    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        kuberay_cfg = config_dict.get("kuberay")
        if not kuberay_cfg:
            return False

        if not kuberay_cfg.get("image"):
            print(
                "Error: kuberay.image is required (container image with leap-finetune)"
            )
            sys.exit(1)

        print("Config contains KubeRay settings - submitting RayJob...\n")
        _print_config_summary(config_dict, kuberay_cfg)
        _submit(config_dict, kuberay_cfg)
        return True
    except SystemExit:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nError submitting KubeRay job: {e}")
        sys.exit(1)


def _print_config_summary(config_dict: dict, kuberay_cfg: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    train_cfg = config_dict.get("training_config", {})
    ds_cfg = config_dict.get("dataset", {})
    peft_cfg = config_dict.get("peft_config", {})

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold cyan", min_width=18)
    table.add_column("Value", style="green")

    table.add_row("Model", config_dict.get("model_name", "?"))
    table.add_row("Training Type", config_dict.get("training_type", "?").upper())
    table.add_row("Dataset", ds_cfg.get("path", "?"))
    if ds_cfg.get("limit"):
        table.add_row("Dataset Limit", f"{ds_cfg['limit']:,}")
    table.add_row("Image", kuberay_cfg["image"])
    table.add_row("GPUs", str(kuberay_cfg.get("gpu_count", 1)))
    table.add_row("Namespace", kuberay_cfg.get("namespace", "default"))

    if train_cfg.get("learning_rate"):
        table.add_row("Learning Rate", f"{float(train_cfg['learning_rate']):.2e}")
    if train_cfg.get("per_device_train_batch_size"):
        table.add_row("Batch Size", str(train_cfg["per_device_train_batch_size"]))
    if train_cfg.get("num_train_epochs"):
        table.add_row("Epochs", str(train_cfg["num_train_epochs"]))
    if peft_cfg.get("use_peft"):
        table.add_row("PEFT", f"Enabled ({peft_cfg.get('extends', 'custom')})")

    panel = Panel(
        table,
        title="[bold blue]KubeRay Training Configuration[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )
    Console().print(panel)


def _submit(config_dict: dict, kuberay_cfg: dict) -> None:
    try:
        from kubernetes import client, config
    except ImportError:
        print("Error: 'kubernetes' package is required for KubeRay support.")
        print("  uv add kubernetes")
        sys.exit(1)

    namespace = kuberay_cfg.get("namespace", "default")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"leap-finetune-{timestamp}"
    configmap_name = f"{job_name}-config"

    # Strip kuberay key so the pod doesn't re-dispatch
    training_config = {k: v for k, v in config_dict.items() if k != "kuberay"}

    output_dir = kuberay_cfg.get("output_dir", "/outputs")
    training_config.setdefault("training_config", {})["output_dir"] = output_dir

    config_str = yaml.dump(training_config)

    # Load kubeconfig from default location (~/.kube/config)
    try:
        config.load_kube_config()
    except Exception:
        print("Error: Could not load kubeconfig.")
        print("  Ensure ~/.kube/config is configured for your EKS cluster.")
        print("  Run: aws eks update-kubeconfig --name <cluster-name>")
        sys.exit(1)

    core_v1 = client.CoreV1Api()
    custom_api = client.CustomObjectsApi()

    # === Create ConfigMap with training config ===
    configmap = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=configmap_name, namespace=namespace),
        data={"config.yaml": config_str},
    )

    print(f"Submitting RayJob '{job_name}' to namespace '{namespace}'...")

    try:
        core_v1.create_namespaced_config_map(namespace=namespace, body=configmap)
    except client.ApiException as e:
        print(f"Failed to create ConfigMap: {e.reason}")
        sys.exit(1)

    # === Generate and submit RayJob ===
    rayjob = _generate_rayjob_manifest(
        job_name, configmap_name, kuberay_cfg, output_dir
    )

    try:
        custom_api.create_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayjobs",
            body=rayjob,
        )
    except client.ApiException as e:
        print(f"Failed to create RayJob: {e.reason}")
        sys.exit(1)

    print(f"RayJob '{job_name}' submitted successfully.")
    print("\nMonitor with:")
    print(f"  kubectl get rayjob {job_name} -n {namespace}")
    print(f"  kubectl logs -f -l ray.io/cluster={job_name} -n {namespace}")


def _generate_rayjob_manifest(
    job_name: str,
    configmap_name: str,
    kuberay_cfg: dict,
    output_dir: str,
) -> dict:
    gpu_count = kuberay_cfg.get("gpu_count", 1)
    gpu_type = kuberay_cfg.get("gpu_type", "nvidia.com/gpu")
    image = kuberay_cfg["image"]
    cpu = kuberay_cfg.get("cpu", 8)
    memory = kuberay_cfg.get("memory", "64Gi")

    # Build env vars
    env_list = [
        {"name": "OUTPUT_DIR", "value": output_dir},
        {"name": "PYTHONPATH", "value": "/app/src"},
        {"name": "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "value": "1"},
    ]
    for key, value in kuberay_cfg.get("env", {}).items():
        env_list.append({"name": key, "value": str(value)})

    # Volume mounts
    volume_mounts = [
        {"name": "config", "mountPath": "/tmp/config.yaml", "subPath": "config.yaml"},
    ]
    volumes = [
        {"name": "config", "configMap": {"name": configmap_name}},
    ]

    # Output PVC mount
    output_pvc = kuberay_cfg.get("output_pvc")
    if output_pvc:
        volume_mounts.append({"name": "output-vol", "mountPath": output_dir})
        volumes.append(
            {"name": "output-vol", "persistentVolumeClaim": {"claimName": output_pvc}}
        )

    # Shared memory for NCCL
    volume_mounts.append({"name": "dshm", "mountPath": "/dev/shm"})
    volumes.append({"name": "dshm", "emptyDir": {"medium": "Memory"}})

    # Container spec
    container = {
        "name": "ray-head",
        "image": image,
        "resources": {
            "limits": {
                gpu_type: str(gpu_count),
                "memory": memory,
                "cpu": str(cpu),
            },
            "requests": {
                gpu_type: str(gpu_count),
                "memory": memory,
                "cpu": str(cpu),
            },
        },
        "volumeMounts": volume_mounts,
        "env": env_list,
    }

    # Pod spec
    pod_spec = {"containers": [container], "volumes": volumes}

    # Node selector / tolerations
    if kuberay_cfg.get("node_selector"):
        pod_spec["nodeSelector"] = kuberay_cfg["node_selector"]
    if kuberay_cfg.get("tolerations"):
        pod_spec["tolerations"] = kuberay_cfg["tolerations"]

    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {"name": job_name},
        "spec": {
            "submissionMode": "K8sJobMode",
            "entrypoint": 'python -c "'
            "import sys; sys.argv=['leap-finetune', '/tmp/config.yaml']; "
            'from leap_finetune import main; main()"',
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 300,
            "rayClusterSpec": {
                "rayVersion": "2.48.0",
                "headGroupSpec": {
                    "rayStartParams": {"num-gpus": str(gpu_count)},
                    "template": {
                        "metadata": {
                            "labels": {"ray.io/cluster": job_name},
                        },
                        "spec": pod_spec,
                    },
                },
            },
        },
    }
