import os
import warnings
import logging
import tempfile
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path
import shutil

_ENV_DONE = False


def setup_training_environment() -> None:
    global _ENV_DONE
    if _ENV_DONE:
        return

    os.environ.setdefault("DS_DISABLE_CONFIG_PRINT", "1")
    os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "ERROR")
    warnings.filterwarnings("ignore")  # keep only tracebacks

    # Use a writable cache directory to avoid permission errors
    cache = "/tmp/triton_cache"
    os.makedirs(cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache

    # Disable DeepSpeed Triton autotune to prevent /dev/shm permission errors
    os.environ.setdefault("DS_TRITON_AUTOTUNE", "0")
    os.environ.setdefault("TRITON_DISABLE_AUTOTUNE", "1")

    try:
        import deepspeed
        from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

        ds_log = deepspeed.utils.logging.logger
        ds_log.setLevel(logging.ERROR)
        for h in ds_log.handlers:
            h.setLevel(logging.ERROR)

        _orig_ds_destroy = BF16_Optimizer.destroy

        def _safe_ds_destroy(self, *a, **kw):
            try:
                _orig_ds_destroy(self, *a, **kw)
            except IndexError:
                pass

        BF16_Optimizer.destroy = _safe_ds_destroy

    except ImportError:
        print("Deepspeed not available in environment; nothing to patch")

    print("Training environment configured ✅")
    _ENV_DONE = True


def configure_wandb_logging(wandb_logging: bool) -> None:
    if wandb_logging:
        if not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_PROJECT", "leap-finetune")


def get_ray_env_vars(ray_temp_dir: str) -> dict[str, str]:
    """Environment variables passed to ray.init runtime_env.env_vars"""
    return {
        "TMPDIR": ray_temp_dir,
        "TEMP": ray_temp_dir,
        "TMP": ray_temp_dir,
        "NCCL_IB_DISABLE": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_SOCKET_IFNAME": "lo",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "NCCL_TIMEOUT": "300",
        "RAY_DISABLE_IMPORT_WARNING": "1",
        "RAY_memory_monitor_refresh_ms": "0",
        "RAY_DATA_DISABLE_PROGRESS_BARS": "1",
        "RAY_IGNORE_UNHANDLED_ERRORS": "1",
    }


def print_next_steps_panel(output_dir: str) -> None:
    """Render the Next Steps panel spanning full console width."""
    console = Console()
    quick_start_url = (
        "https://leap.liquid.ai/docs/leap-bundle/quick-start?utm_source=github"
        "&utm_medium=link&utm_campaign=LEAP&utm_content=general"
    )

    next_steps_table = Table(show_header=False, box=None, padding=(0, 2))
    next_steps_table.add_column("Property", style="bold cyan", min_width=15)
    next_steps_table.add_column("Value", style="green")

    next_steps_table.add_row("Status", "[bold green]Training complete![/bold green]")
    next_steps_table.add_row("Checkpoint Directory", f"[cyan]{output_dir}[/cyan]")
    next_steps_table.add_row(
        "Bundle",
        f"[dim]leap-bundle create {output_dir}/[CHECKPOINT_NAME][/dim]",
    )
    next_steps_table.add_row(
        "Quick Start",
        f"[link={quick_start_url}]{quick_start_url}[/link]",
    )

    console.print(
        Panel(
            next_steps_table,
            title="Next Step: Bundle for LEAP",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )


def select_ray_temp_dir(preferred: str | None = None) -> str:
    """Pick a temp directory on a filesystem with >10% free space when possible.

    Falls back to the path with the highest free ratio if none meet the threshold.
    """
    candidates: list[str] = []
    env_tmp = os.environ.get("RAY_TMPDIR")
    if env_tmp:
        candidates.append(env_tmp)
    if preferred:
        candidates.append(preferred)
    home_default = str(Path.home() / "ray_temp")
    candidates.extend(
        [
            "/tmp/ray",
            "/var/tmp/ray",
            "/dev/shm/ray",
            home_default,
        ]
    )

    best_path = home_default
    best_ratio = -1.0
    for path in candidates:
        try:
            base = Path(path)
            base.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(base))
            ratio = usage.free / usage.total if usage.total else 0.0
            if ratio > 0.10:
                return str(base)
            if ratio > best_ratio:
                best_ratio = ratio
                best_path = str(base)
        except OSError:
            continue

    return best_path


def _paths_with_free_space(
    candidates: list[str], min_free_ratio: float = 0.10
) -> list[str]:
    qualified: list[str] = []
    for path in candidates:
        try:
            base = Path(path)
            base.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(base))
            ratio = usage.free / usage.total if usage.total else 0.0
            if ratio >= min_free_ratio:
                qualified.append(str(base))
        except OSError:
            continue
    return qualified


def select_object_spilling_dir(ray_temp_dir: str | None = None) -> str:
    """Choose a directory with enough free space for Ray object spilling."""
    home = str(Path.home())
    candidates = [
        os.path.join(ray_temp_dir or home, "spill"),
        "/dev/shm/ray_spill",
        "/tmp/ray_spill",
        "/var/tmp/ray_spill",
        f"{home}/ray_spill",
    ]
    good = _paths_with_free_space(candidates, min_free_ratio=0.10)
    target = good[0] if good else candidates[-1]
    Path(target).mkdir(parents=True, exist_ok=True)
    return target


def should_connect_existing_cluster(*args, **kwargs):  # simple check
    return bool(os.environ.get("RAY_ADDRESS"))
