import pathlib
import sys
import tomllib

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


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

        print("Config contains Modal settings - submitting Modal job...")
        _submit(config_dict, modal_cfg)
        return True
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error submitting Modal job: {e}")
        sys.exit(1)


def _submit(config_dict: dict, modal_cfg: dict) -> None:
    import modal

    detach = modal_cfg.get("detach", False)

    # Strip modal key so the container doesn't re-dispatch
    config_dict = {k: v for k, v in config_dict.items() if k != "modal"}

    # Point output_dir at the mounted volume
    output_dir = modal_cfg.get("output_dir", "/outputs")
    config_dict.setdefault("training_config", {})["output_dir"] = output_dir

    config_str = yaml.dump(config_dict)

    app = modal.App(modal_cfg.get("app_name", "leap-finetune"))
    image = _build_image(modal_cfg)
    volume = modal.Volume.from_name(
        modal_cfg.get("output_volume", "leap-finetune-outputs"),
        create_if_missing=True,
    )
    secrets = [modal.Secret.from_name(s) for s in modal_cfg.get("secrets", [])]

    @app.function(
        image=image,
        gpu=modal_cfg.get("gpu", "H100"),
        timeout=modal_cfg.get("timeout", 86400),
        volumes={output_dir: volume},
        secrets=secrets,
    )
    def _train(cfg: str) -> None:
        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg)
            tmp_path = f.name

        sys.argv = ["leap-finetune", tmp_path]
        from leap_finetune import main as leap_main

        leap_main()

    with app.run():
        if detach:
            call = _train.spawn(config_str)
            print(f"Modal job submitted (detached). Function call ID: {call.object_id}")
            print(
                f"Monitor with: modal app logs {modal_cfg.get('app_name', 'leap-finetune')}"
            )
        else:
            _train.remote(config_str)


def _build_image(modal_cfg: dict):
    import modal

    base_image = modal_cfg.get("base_image", "nvcr.io/nvidia/pytorch:25.02-py3")
    deps = _read_deps()

    # flash-attn needs --no-build-isolation
    flash_attn_deps = [d for d in deps if "flash-attn" in d]
    regular_deps = [d for d in deps if "flash-attn" not in d]

    image = modal.Image.from_registry(base_image).pip_install(*regular_deps)

    if flash_attn_deps:
        image = image.pip_install(
            *flash_attn_deps, extra_options="--no-build-isolation"
        )

    # Copy source into the container as RUNTIME_DIR.
    src_dir = _REPO_ROOT / "src" / "leap_finetune"
    image = image.copy_local_dir(str(src_dir), remote_path="/app/src/leap_finetune")
    image = image.env({"PYTHONPATH": "/app/src", "LEAP_FINETUNE_DIR": "/app"})
    return image


def _read_deps() -> list[str]:
    pyproject_path = _REPO_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Skip modal since it's not needed inside the training container
    return [d for d in data["project"]["dependencies"] if "modal" not in d.lower()]
