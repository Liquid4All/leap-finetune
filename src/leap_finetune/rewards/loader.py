"""Resolve YAML ``rewards:`` entries into Python callables.

Three YAML shapes are supported under a single ``rewards:`` key:

1. A plain list of ``<path>::<function>`` specs (no weights — GRPO
   defaults each to 1.0).
2. A dict with ``funcs:`` and optional ``weights:``.
3. A ``recipe:`` reference to a :class:`Recipe` subclass, optionally
   combined with extra ``funcs:`` and/or a ``weights:`` override that
   applies to the combined list.

Path resolution (both spec forms): absolute paths are used as-is;
relative paths are tried against the current working directory first,
then the directory containing the YAML config.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Callable
from pathlib import Path

from leap_finetune.rewards.recipe import Recipe

logger = logging.getLogger(__name__)

_SEP = "::"


def resolve_reward_specs(
    rewards_cfg: list | dict | None,
    config_dir: Path | str,
) -> tuple[list[Callable], list[float] | None]:
    """Resolve a YAML ``rewards:`` entry into ``(funcs, weights)``.

    ``weights`` is ``None`` when the user didn't specify any, letting
    ``GRPOConfig`` default every reward to 1.0.

    Raises:
        ValueError: malformed spec, missing file, missing function or
            class, weights length mismatch, or a class referenced by
            ``recipe:`` that doesn't subclass :class:`Recipe`.
    """
    if not rewards_cfg:
        return [], None

    config_dir = Path(config_dir).resolve()

    # Normalize the three YAML shapes.
    if isinstance(rewards_cfg, list):
        recipe_spec = None
        individual_specs = rewards_cfg
        weights_override = None
    elif isinstance(rewards_cfg, dict):
        recipe_spec = rewards_cfg.get("recipe")
        individual_specs = rewards_cfg.get("funcs") or rewards_cfg.get("rewards") or []
        weights_override = rewards_cfg.get("weights")
    else:
        raise ValueError(
            f"`rewards` must be a list or a dict, got {type(rewards_cfg).__name__}"
        )

    if recipe_spec is None and not individual_specs:
        return [], None

    # Load recipe pairs (if any) with their built-in default weights.
    recipe_pairs: list[tuple[Callable, float]] = []
    if recipe_spec is not None:
        if not isinstance(recipe_spec, str):
            raise ValueError(
                f"`rewards.recipe` must be a string '<path>::<ClassName>', got "
                f"{type(recipe_spec).__name__}: {recipe_spec!r}"
            )
        recipe_cls = _load_recipe_class(recipe_spec, config_dir)
        instance = recipe_cls()
        recipe_pairs = _validate_reward_pairs(instance.rewards(), recipe_spec)

    # Individual funcs each default to weight 1.0.
    individual_pairs: list[tuple[Callable, float]] = []
    for spec in individual_specs:
        if not isinstance(spec, str):
            raise ValueError(
                f"Reward spec must be a string, got {type(spec).__name__}: {spec!r}"
            )
        fn = _load_reward_spec(spec, config_dir)
        individual_pairs.append((fn, 1.0))

    all_pairs = recipe_pairs + individual_pairs
    funcs = [p[0] for p in all_pairs]
    default_weights = [p[1] for p in all_pairs]

    # An explicit `weights:` overrides both recipe defaults and 1.0s.
    if weights_override is not None:
        if not isinstance(weights_override, list) or not all(
            isinstance(w, (int, float)) for w in weights_override
        ):
            raise ValueError(
                f"`rewards.weights` must be a list of numbers, got {weights_override!r}"
            )
        if len(weights_override) != len(funcs):
            raise ValueError(
                f"`rewards.weights` has {len(weights_override)} entries but "
                f"{len(funcs)} reward function(s) were resolved "
                f"({len(recipe_pairs)} from recipe + {len(individual_pairs)} individual). "
                f"Lengths must match."
            )
        final_weights: list[float] | None = [float(w) for w in weights_override]
    elif recipe_spec is not None:
        # Ship the recipe's (possibly non-uniform) defaults through.
        final_weights = default_weights
    else:
        # Individual funcs with no explicit weights → let GRPOConfig default to 1.0.
        final_weights = None

    logger.info(
        "Loaded %d reward function(s): %s%s",
        len(funcs),
        ", ".join(f.__name__ for f in funcs),
        f" (recipe={recipe_spec})" if recipe_spec else "",
    )
    return funcs, final_weights


def load_recipe(
    spec: str,
    config_dir: Path | str = ".",
) -> type[Recipe]:
    """Load a :class:`Recipe` subclass by ``<path>::ClassName`` spec.

    Use this inside a user recipe file to subclass a shipped recipe::

        from leap_finetune.rewards import Recipe, load_recipe

        VLMGroundingRecipe = load_recipe(
            "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
        )

        class MyRecipe(VLMGroundingRecipe):
            ...
    """
    return _load_recipe_class(spec, Path(config_dir).resolve())


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_recipe_class(spec: str, config_dir: Path) -> type[Recipe]:
    """Import ``<path>::ClassName`` and verify it subclasses :class:`Recipe`."""
    if _SEP not in spec:
        raise ValueError(
            f"Recipe spec {spec!r} is malformed. Expected format: "
            f"'<path>::<ClassName>' (e.g. './rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe'). "
            f"The '::' separator is required."
        )
    path_str, _, class_name = spec.partition(_SEP)
    path_str = path_str.strip()
    class_name = class_name.strip()
    if not path_str or not class_name:
        raise ValueError(
            f"Recipe spec {spec!r} is malformed: both path and class name must be non-empty."
        )

    module = _import_reward_file(path_str, config_dir, original_spec=spec)

    if not hasattr(module, class_name):
        available = sorted(
            n
            for n in dir(module)
            if not n.startswith("_") and isinstance(getattr(module, n), type)
        )
        raise ValueError(
            f"Class {class_name!r} not found in {path_str}. Available classes: {available}"
        )

    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise ValueError(
            f"{class_name!r} in {path_str} is not a class (got {type(cls).__name__})."
        )
    if not issubclass(cls, Recipe):
        raise ValueError(
            f"{class_name!r} in {path_str} must subclass leap_finetune.rewards.Recipe. "
            f"Got a {cls.__bases__} subclass instead."
        )
    return cls


def _validate_reward_pairs(
    pairs, recipe_spec: str
) -> list[tuple[Callable, float]]:
    """Check that ``Recipe.rewards()`` returned a non-empty ``[(callable, float)]``."""
    if not isinstance(pairs, list):
        raise ValueError(
            f"Recipe {recipe_spec!r}: rewards() must return a list of "
            f"(callable, float) tuples, got {type(pairs).__name__}."
        )
    if not pairs:
        raise ValueError(
            f"Recipe {recipe_spec!r}: rewards() returned an empty list. "
            f"A recipe must declare at least one reward."
        )
    validated: list[tuple[Callable, float]] = []
    for i, item in enumerate(pairs):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise ValueError(
                f"Recipe {recipe_spec!r}: rewards()[{i}] must be a "
                f"(callable, float) tuple, got {item!r}"
            )
        fn, weight = item
        if not callable(fn):
            raise ValueError(
                f"Recipe {recipe_spec!r}: rewards()[{i}] first element must be "
                f"callable, got {type(fn).__name__}"
            )
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"Recipe {recipe_spec!r}: rewards()[{i}] weight must be a number, "
                f"got {type(weight).__name__}: {weight!r}"
            )
        validated.append((fn, float(weight)))
    return validated


def _load_reward_spec(spec: str, config_dir: Path) -> Callable:
    """Load a single ``<path>::<fn_name>`` spec into a callable."""
    if _SEP not in spec:
        raise ValueError(
            f"Reward spec {spec!r} is malformed. Expected format: '<path>::<function_name>' "
            f"(e.g. './rewards/accuracy.py::accuracy_reward'). The '::' separator is required."
        )
    path_str, _, fn_name = spec.partition(_SEP)
    path_str = path_str.strip()
    fn_name = fn_name.strip()
    if not path_str or not fn_name:
        raise ValueError(
            f"Reward spec {spec!r} is malformed: both path and function name must be non-empty."
        )

    module = _import_reward_file(path_str, config_dir, original_spec=spec)

    if not hasattr(module, fn_name):
        available = [
            n
            for n in dir(module)
            if not n.startswith("_") and callable(getattr(module, n))
        ]
        raise ValueError(
            f"Function {fn_name!r} not found in {path_str}. Available callables: {available}"
        )
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise ValueError(
            f"{fn_name!r} in {path_str} is not callable (got {type(fn).__name__})"
        )
    return fn


def _import_reward_file(path_str: str, config_dir: Path, *, original_spec: str):
    """Resolve ``path_str`` (absolute → CWD → config_dir) and import it."""
    raw = Path(path_str)
    if raw.is_absolute():
        candidates = [raw]
    else:
        cwd = Path(os.getcwd()).resolve()
        candidates = [(cwd / raw).resolve(), (config_dir / raw).resolve()]

    path: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            path = candidate
            break

    if path is None:
        raise ValueError(
            f"Reward file not found for spec {original_spec!r}. Tried: "
            + " and ".join(str(c) for c in candidates)
            + ". Use a path relative to the working directory you launch "
            "leap-finetune from (e.g. './rewards/tasks/vlm_grounding/recipe.py') or an "
            "absolute path."
        )

    module_name = f"_leap_reward_{path.stem}_{abs(hash(str(path)))}"
    spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
    if spec_obj is None or spec_obj.loader is None:
        raise ValueError(f"Could not load reward file as a Python module: {path}")
    module = importlib.util.module_from_spec(spec_obj)
    try:
        spec_obj.loader.exec_module(module)
    except Exception as e:
        raise ValueError(f"Failed to import reward file {path}: {e}") from e
    return module
