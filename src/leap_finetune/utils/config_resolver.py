import pathlib

from leap_finetune.utils.constants import LEAP_FINETUNE_DIR


def resolve_config_path(config_input: str) -> pathlib.Path:
    input_path = pathlib.Path(config_input)

    # Auto-append .yaml if no extension provided
    candidates = [config_input]
    if not input_path.suffix:
        candidates.append(config_input + ".yaml")

    for candidate in candidates:
        candidate_path = pathlib.Path(candidate)

        # 1. Absolute or CWD-relative path
        if candidate_path.exists():
            return candidate_path.resolve()

        # 2. Look in ./job_configs/
        local_job_configs = pathlib.Path.cwd() / "job_configs" / candidate
        if local_job_configs.exists():
            return local_job_configs.resolve()

        # 3. Look in repo job_configs/
        repo_job_configs = LEAP_FINETUNE_DIR / "job_configs" / candidate
        if repo_job_configs.exists():
            return repo_job_configs.resolve()

    raise FileNotFoundError(f"Config file not found at: {input_path}")
