import os


def setup_training_environment() -> None:
    """
    Setup the training environment with proper logging configuration.
    Call this in the main trainer before ray.init().
    """

    triton_cache = os.path.expanduser("~/tmp/triton_cache")
    os.makedirs(triton_cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache

    print("Training environment configured ✅")
