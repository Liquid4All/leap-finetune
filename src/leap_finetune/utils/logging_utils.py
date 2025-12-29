import os
import warnings
import logging
import tempfile

_ENV_DONE = False


def setup_training_environment() -> None:
    global _ENV_DONE
    if _ENV_DONE:
        return

    os.environ.setdefault("DS_DISABLE_CONFIG_PRINT", "1")
    os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "ERROR")
    warnings.filterwarnings("ignore")  # keep only tracebacks

    pid = os.getpid()
    existing_base = os.environ.get("TRITON_CACHE_DIR")

    if existing_base:
        cache_dir = f"{existing_base}_{pid}"
    else:
        default_base = "/dev/shm"
        cache_dir = os.path.join(default_base, f"triton_cache_{pid}")

    try:
        os.makedirs(cache_dir, exist_ok=True)
    except (OSError, PermissionError):
        fallback_base = tempfile.gettempdir()
        cache_dir = os.path.join(fallback_base, f"triton_cache_{pid}")
        os.makedirs(cache_dir, exist_ok=True)

    os.environ["TRITON_CACHE_DIR"] = cache_dir
    # DeepSpeed also respects DS_TRITON_CACHE_DIR. Set both for completeness.
    os.environ["DS_TRITON_CACHE_DIR"] = cache_dir

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
