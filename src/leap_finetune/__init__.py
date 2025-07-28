import sys

from leap_finetune.trainer import ray_trainer
from leap_finetune.utils.logging import setup_training_environment
from leap_finetune.utils.constants import LEAP_FINETUNE_DIR

# Set Ray Core logging before importing Ray
setup_training_environment()


def main() -> None:
    print("Launching leap-finetune ✅")

    # Import config only in main process, not Ray workers
    root_path = str(LEAP_FINETUNE_DIR)
    sys.path.insert(0, root_path)

    try:
        from config import JOB_CONFIG

        JOB_CONFIG.print_config_summary()
        JOB_CONFIG = JOB_CONFIG.to_dict()
    except ModuleNotFoundError:
        raise FileNotFoundError(
            "config.py not found. Please create a config.py file with JOB_CONFIG defined."
        )
    except ImportError:
        raise ImportError(
            "JOB_CONFIG not found in config.py. Please define JOB_CONFIG in your config.py file."
        )
    except ValueError as e:
        raise ValueError(f"Invalid JOB_CONFIG: {e}")
    finally:
        sys.path.remove(root_path)

    ray_trainer(JOB_CONFIG)
