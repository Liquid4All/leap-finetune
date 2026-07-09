import json
from pathlib import Path
from typing import Any

LEAP_RAY_FINAL_METRICS_FILE = ".leap_ray_final_metrics.json"


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def hydrate_missing_ray_metrics(result, output_dir: str):
    """Fill Ray 2.51 Train V2 results from the callback's final metrics file."""
    if result is None or getattr(result, "metrics", None) is not None:
        return result

    metrics_path = Path(output_dir) / LEAP_RAY_FINAL_METRICS_FILE
    if not metrics_path.exists():
        return result

    with metrics_path.open(encoding="utf-8") as f:
        result.metrics = json.load(f)

    return result
