# === Math accuracy reward ===
#
# Re-export TRL's math_verify-backed accuracy_reward. The dataset must include
# a string `solution` column.

from trl.rewards import accuracy_reward

__all__ = ["accuracy_reward"]
