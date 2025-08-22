import os
import warnings
import logging

_ENV_DONE = False


def setup_training_environment() -> None:
    global _ENV_DONE
    if _ENV_DONE:
        return

    os.environ.setdefault("DS_DISABLE_CONFIG_PRINT", "1")
    os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "ERROR")
    warnings.filterwarnings("ignore")  # keep only tracebacks

    cache = "/dev/shm"
    try:
        os.makedirs(cache, exist_ok=True)
    except OSError:
        cache = "/tmp/triton_cache"
        os.makedirs(cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache

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
