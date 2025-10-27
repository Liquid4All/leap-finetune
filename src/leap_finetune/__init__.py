import sys


from leap_finetune.trainer import ray_trainer  # noqa
from leap_finetune.utils.constants import LEAP_FINETUNE_DIR  # noqa


# Export new W&B and Weave integration modules
from leap_finetune.models import UnslothLoRALFM2  # noqa
from leap_finetune.callbacks import (  # noqa
    conversation_scorer,
    conversation_improvement_scorer,
    response_length_scorer,
    coherence_scorer,
    diversity_scorer,
)
from leap_finetune.utils.weave_trainer_utils import (  # noqa
    get_trainer_with_evaluation_callback,
    create_sft_config_with_weave_eval,
    DEFAULT_SCORERS,
)
from leap_finetune.data_loaders import (  # noqa
    preprocess_dataset_for_weave_evaluation,
    preprocess_dataset_for_weave_evaluation_list,
)

__all__ = [
    "ray_trainer",
    "UnslothLoRALFM2",
    "conversation_scorer",
    "conversation_improvement_scorer",
    "response_length_scorer",
    "coherence_scorer",
    "diversity_scorer",
    "get_trainer_with_evaluation_callback",
    "create_sft_config_with_weave_eval",
    "DEFAULT_SCORERS",
    "preprocess_dataset_for_weave_evaluation",
    "preprocess_dataset_for_weave_evaluation_list",
]


def main() -> None:
    # Set Ray Core logging before importing Ray

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
        raise ValueError(f"Issue with JOB_CONFIG: {e}")
    finally:
        sys.path.remove(root_path)

    ray_trainer(JOB_CONFIG)
