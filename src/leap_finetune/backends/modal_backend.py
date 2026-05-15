import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_HF_SECRET_NAME = "huggingface-secret"


def check_and_handle_modal(config_path_arg: str) -> bool:
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

        modal_cfg = config_dict.get("modal")
        if not modal_cfg:
            return False

        print("Config contains Modal settings - submitting Modal job...\n")
        _preflight_checks(config_dict, modal_cfg)
        _print_config_summary(config_dict, modal_cfg)
        _submit(config_dict, modal_cfg)
        return True
    except SystemExit:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nError submitting Modal job: {e}")
        sys.exit(1)


# === Preflight checks ===


def _preflight_checks(config_dict: dict, modal_cfg: dict) -> None:
    # HF token is needed for model downloads (rate limits / gated models) and trackio
    token = _get_local_hf_token()
    if not token:
        print("Error: HF token not found. Required for model downloads on Modal.")
        print("  Run: huggingface-cli login")
        sys.exit(1)

    # Auto-add and auto-create the Modal secret from local token
    secrets = modal_cfg.get("secrets") or []
    if _HF_SECRET_NAME not in secrets:
        secrets.append(_HF_SECRET_NAME)
        modal_cfg["secrets"] = secrets

    _ensure_modal_secret(token)


def _get_local_hf_token() -> str | None:
    try:
        import huggingface_hub

        return huggingface_hub.get_token()
    except Exception:
        return None


def _ensure_modal_secret(token: str) -> None:
    result = subprocess.run(
        ["modal", "secret", "list", "--json"], capture_output=True, text=True
    )
    if result.returncode == 0:
        import json

        existing = {s["Name"] for s in json.loads(result.stdout)}
        if _HF_SECRET_NAME in existing:
            return

    import modal

    print(f"Creating Modal secret '{_HF_SECRET_NAME}' from local HF token...")
    modal.Secret.create_deployed(_HF_SECRET_NAME, {"HF_TOKEN": token})
    print(f"  Created '{_HF_SECRET_NAME}' on Modal.\n")


# === Config summary ===


def _print_config_summary(config_dict: dict, modal_cfg: dict) -> None:
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
    table.add_row("GPU", modal_cfg.get("gpu", "H100"))

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
        title="[bold blue]Modal Training Configuration[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )
    Console().print(panel)


# === Submit job ===


def _submit(config_dict: dict, modal_cfg: dict) -> None:
    import modal

    detach = modal_cfg.get("detach", False)

    # Strip modal key so the container doesn't re-dispatch
    config_dict = {k: v for k, v in config_dict.items() if k != "modal"}

    # Point output_dir at the mounted volume
    output_dir = modal_cfg.get("output_dir", "/outputs")
    config_dict.setdefault("training_config", {})["output_dir"] = output_dir

    # Print trackio dashboard URL before training starts
    space_id = config_dict.get("training_config", {}).get("trackio_space_id")
    if space_id:
        print(f"\nTrackio dashboard: https://huggingface.co/spaces/{space_id}\n")

    config_str = yaml.dump(config_dict)
    app_name = modal_cfg.get("app_name", "leap-finetune")
    image = _build_image(modal_cfg)
    volume = modal.Volume.from_name(
        modal_cfg.get("output_volume", "leap-finetune"),
        create_if_missing=True,
    )
    secrets = [modal.Secret.from_name(s) for s in (modal_cfg.get("secrets") or [])]

    if detach:
        _submit_detached(app_name, image, output_dir, volume, secrets, modal_cfg, config_str)
    else:
        _submit_attached(app_name, image, output_dir, volume, secrets, modal_cfg, config_str)


def _run_train(cfg: str, output_dir: str) -> None:
    import tempfile

    for prefix in ("leap_finetune", "huggingface_hub", "datasets", "fsspec"):
        for mod_name in [m for m in sys.modules if m.startswith(prefix)]:
            del sys.modules[mod_name]

    os.environ["LEAP_FINETUNE_DIR"] = "/app"
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ.pop("HF_HUB_OFFLINE", None)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(cfg)
        tmp_path = f.name

    sys.argv = ["leap-finetune", tmp_path]
    from leap_finetune import main as leap_main

    leap_main()


def _submit_attached(app_name, image, output_dir, volume, secrets, modal_cfg, config_str):
    import modal

    app = modal.App(app_name)

    @app.function(
        image=image,
        gpu=modal_cfg.get("gpu", "H100"),
        timeout=modal_cfg.get("timeout", 86400),
        volumes={output_dir: volume},
        secrets=secrets,
        serialized=True,
    )
    def train(cfg: str) -> None:
        _run_train(cfg, output_dir)

    print("Waiting for container to start (image is cached after first build)...")
    with modal.enable_output():
        with app.run():
            try:
                train.remote(config_str)
            except modal.exception.RemoteError as e:
                print(
                    f"\nModal container exited with error after training: {e}\n"
                    "Check the logs above — training likely completed and the\n"
                    "error occurred during container teardown."
                )


def _submit_detached(app_name, image, output_dir, volume, secrets, modal_cfg, config_str):
    gpu = modal_cfg.get("gpu", "H100")
    timeout = modal_cfg.get("timeout", 86400)
    secret_names = modal_cfg.get("secrets") or []
    output_volume = modal_cfg.get("output_volume", "leap-finetune")

    base_image = modal_cfg.get("base_image", "nvidia/cuda:12.8.0-devel-ubuntu22.04")
    pyproject = str(_REPO_ROOT / "pyproject.toml")
    lockfile = str(_REPO_ROOT / "uv.lock")
    src_dir = str(_REPO_ROOT / "src" / "leap_finetune")

    script = textwrap.dedent(f"""\
        import modal

        app = modal.App("{app_name}")

        image = (
            modal.Image.from_registry("{base_image}", add_python="3.12")
            .pip_install("uv")
            .add_local_file("{pyproject}", remote_path="/app/pyproject.toml", copy=True)
            .add_local_file("{lockfile}", remote_path="/app/uv.lock", copy=True)
            .run_commands(
                "cd /app && uv export --frozen --no-dev --no-emit-project --no-hashes"
                " > requirements.txt"
                " && grep -v flash-attn requirements.txt > requirements-modal.txt"
                " && uv pip install --system -r requirements-modal.txt",
            )
            .add_local_dir("{src_dir}", remote_path="/app/src/leap_finetune", copy=True)
            .env({{
                "PYTHONPATH": "/app/src",
                "LEAP_FINETUNE_DIR": "/app",
                "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "1",
            }})
        )

        volume = modal.Volume.from_name("{output_volume}", create_if_missing=True)
        secrets = [modal.Secret.from_name(s) for s in {secret_names!r}]

        @app.function(
            image=image,
            gpu="{gpu}",
            timeout={timeout},
            volumes={{"{output_dir}": volume}},
            secrets=secrets,
            serialized=True,
        )
        def train(cfg: str) -> None:
            import os, sys, tempfile
            for prefix in ("leap_finetune", "huggingface_hub", "datasets", "fsspec"):
                for mod_name in [m for m in sys.modules if m.startswith(prefix)]:
                    del sys.modules[mod_name]
            os.environ["LEAP_FINETUNE_DIR"] = "/app"
            os.environ["OUTPUT_DIR"] = "{output_dir}"
            os.environ.pop("HF_HUB_OFFLINE", None)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(cfg)
                tmp_path = f.name
            sys.argv = ["leap-finetune", tmp_path]
            from leap_finetune import main as leap_main
            leap_main()

        @app.local_entrypoint()
        def main():
            train.remote({config_str!r})
    """)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="modal-train-"
    ) as f:
        f.write(script)
        script_path = f.name

    print(f"Submitting detached Modal job via 'modal run --detach'...")
    result = subprocess.run(
        ["modal", "run", "--detach", script_path],
    )
    os.unlink(script_path)

    if result.returncode != 0:
        print(f"modal run --detach failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\nMonitor with: modal app logs {app_name}")


# === Image build ===


def _build_image(modal_cfg: dict):
    import modal

    base_image = modal_cfg.get("base_image", "nvidia/cuda:12.8.0-devel-ubuntu22.04")
    pyproject = _REPO_ROOT / "pyproject.toml"
    lockfile = _REPO_ROOT / "uv.lock"

    image = (
        modal.Image.from_registry(base_image, add_python="3.12")
        .pip_install("uv")
        .add_local_file(str(pyproject), remote_path="/app/pyproject.toml", copy=True)
        .add_local_file(str(lockfile), remote_path="/app/uv.lock", copy=True)
        .run_commands(
            # Export exact versions from lockfile, strip flash-attn (OOMs compiling
            # from source on Modal builders), then install everything else.
            "cd /app && uv export --frozen --no-dev --no-emit-project --no-hashes"
            " > requirements.txt"
            " && grep -v flash-attn requirements.txt > requirements-modal.txt"
            " && uv pip install --system -r requirements-modal.txt",
        )
    )

    src_dir = _REPO_ROOT / "src" / "leap_finetune"
    image = image.add_local_dir(
        str(src_dir), remote_path="/app/src/leap_finetune", copy=True
    )
    image = image.env(
        {
            "PYTHONPATH": "/app/src",
            "LEAP_FINETUNE_DIR": "/app",
            "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "1",
        }
    )
    return image
