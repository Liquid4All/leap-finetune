from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

import yaml

STATE_VERSION = 2
LOG_HISTORY_LIMIT = 200
CHECKPOINT_HISTORY_LIMIT = 20

_PROGRESS_DEFAULT = {
    "heartbeat_at": None,
    "step": None,
    "epoch": None,
    "max_steps": None,
    "last_log": None,
    "last_eval": None,
    "last_checkpoint": None,
}
_HISTORY_DEFAULT = {
    "logs": [],
    "evals": [],
    "checkpoints": [],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str | None, fallback: str = "run") -> str:
    text = value or fallback
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._").lower()
    return slug[:48] or fallback


def get_state_dir(state_dir: str | pathlib.Path | None = None) -> pathlib.Path:
    if state_dir is not None:
        return pathlib.Path(state_dir).expanduser().resolve()
    configured = os.environ.get("LFT_STATE_DIR")
    if configured:
        return pathlib.Path(configured).expanduser().resolve()
    return (pathlib.Path.cwd() / ".lft").resolve()


def _state_path(state_dir: str | pathlib.Path | None = None) -> pathlib.Path:
    return get_state_dir(state_dir) / "state.json"


def _memory_path(state_dir: str | pathlib.Path | None = None) -> pathlib.Path:
    return get_state_dir(state_dir) / "memory.md"


def _ensure_state_files(state_dir: str | pathlib.Path | None = None) -> None:
    root = get_state_dir(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    if not state_path.exists():
        _write_state(_empty_state(), root)
    memory_path = root / "memory.md"
    if not memory_path.exists():
        memory_path.write_text("# LFT Memory\n\n")


def _empty_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "updated_at": _now(), "runs": []}


def _load_state(state_dir: str | pathlib.Path | None = None) -> dict[str, Any]:
    _ensure_state_files(state_dir)
    with open(_state_path(state_dir)) as handle:
        state = json.load(handle)
    state.setdefault("version", STATE_VERSION)
    state.setdefault("runs", [])
    state["runs"] = [_normalize_run(run) for run in state["runs"]]
    return state


def _write_state(
    state: dict[str, Any], state_dir: str | pathlib.Path | None = None
) -> None:
    root = get_state_dir(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    state["version"] = STATE_VERSION
    state["updated_at"] = _now()
    state["runs"] = [_normalize_run(run) for run in state.get("runs", [])]
    path = root / "state.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, default=_json_default, indent=2) + "\n")
    tmp_path.replace(path)


@contextmanager
def _state_lock(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "state.lock"
    with open(lock_path, "a+") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except Exception:
            fcntl = None
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _mutate_state(
    mutator: Callable[[dict[str, Any]], Any],
    state_dir: str | pathlib.Path | None = None,
) -> Any:
    root = get_state_dir(state_dir)
    with _state_lock(root):
        _ensure_state_files(root)
        state = _load_state(root)
        result = mutator(state)
        _write_state(state, root)
        return result


def _json_default(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _scalar_json_values(values: dict[str, Any] | None) -> dict[str, Any]:
    scalars: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if isinstance(value, pathlib.Path):
            value = str(value)
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                pass
        if isinstance(value, bool | int | str):
            scalars[str(key)] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                scalars[str(key)] = value
    return scalars


def _coerce_step(value: Any) -> int | float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _phase_for_status(status: str | None) -> str:
    if status == "submitted":
        return "submitted"
    if status == "completed":
        return "completed"
    if status == "failed":
        return "failed"
    return "training"


def _normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    run = dict(run)
    run.setdefault("status", "running")
    run.setdefault("phase", _phase_for_status(run.get("status")))
    run.setdefault("metrics", {})
    run.setdefault("backend_meta", {})
    run.setdefault("log_refs", {})

    progress = {**_PROGRESS_DEFAULT, **dict(run.get("progress") or {})}
    run["progress"] = progress

    raw_history = dict(run.get("history") or {})
    run["history"] = {
        "logs": list(raw_history.get("logs") or []),
        "evals": list(raw_history.get("evals") or []),
        "checkpoints": list(raw_history.get("checkpoints") or []),
    }
    return run


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _kind(config_dict: dict[str, Any] | None) -> str:
    if not config_dict:
        return "run"
    if config_dict.get("training_type") or config_dict.get("dataset"):
        return "train"
    if config_dict.get("evals") or config_dict.get("benchmarks"):
        return "eval"
    return "run"


def _summary(config_dict: dict[str, Any] | None, kind: str) -> str:
    if not config_dict:
        return kind
    project = (
        config_dict.get("project_name")
        or config_dict.get("job_name")
        or config_dict.get("model_name")
        or "default"
    )
    if kind == "train":
        training_type = config_dict.get("training_type", "train")
        return f"{training_type} run for {project}"
    if kind == "eval":
        model = (
            config_dict.get("checkpoint") or config_dict.get("model_name") or project
        )
        return f"eval run for {model}"
    return str(project)


def _config_path(config_path: str | pathlib.Path | None) -> str | None:
    if config_path is None:
        return None
    try:
        return str(pathlib.Path(config_path).expanduser().resolve())
    except Exception:
        return str(config_path)


def _new_run_id(summary: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{_slug(summary)}-{uuid.uuid4().hex[:6]}"


def _upsert_run(
    run: dict[str, Any], state_dir: str | pathlib.Path | None = None
) -> None:
    def mutate(state: dict[str, Any]) -> None:
        normalized = _normalize_run(run)
        runs = [
            existing for existing in state["runs"] if existing.get("id") != run["id"]
        ]
        state["runs"] = [normalized, *runs]

    _mutate_state(mutate, state_dir)


def _find_run(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    if run_id == "latest":
        if not state.get("runs"):
            raise FileNotFoundError("Run not found: latest")
        return state["runs"][0]
    for run in state.get("runs", []):
        if run.get("id") == run_id:
            return run
    raise FileNotFoundError(f"Run not found: {run_id}")


def _active_run_id(run_id: str | None) -> str | None:
    return run_id or os.environ.get("LFT_RUN_ID")


def _bounded_append(items: list[Any], item: Any, limit: int | None) -> list[Any]:
    items.append(item)
    if limit is not None and limit > 0 and len(items) > limit:
        return items[-limit:]
    return items


def _record_log(
    *,
    step: int | float | None,
    epoch: float | None,
    metrics: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "at": _now(),
        "source": source,
        "metrics": _json_safe(metrics),
    }
    if step is not None:
        record["step"] = step
    if epoch is not None:
        record["epoch"] = epoch
    return record


def _record_eval(
    *,
    step: int | float | None,
    metrics: dict[str, Any] | None,
    status: str,
    source: str,
    backend: str | None = None,
    backend_id: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "at": _now(),
        "status": status,
        "source": source,
        "metrics": _json_safe(metrics or {}),
    }
    if step is not None:
        record["step"] = step
    if backend:
        record["backend"] = backend
    if backend_id:
        record["backend_id"] = backend_id
    if error:
        record["error"] = error
    if metadata:
        record["metadata"] = _json_safe(metadata)
    return record


class RunTracker:
    def __init__(self, run_id: str, state_dir: pathlib.Path) -> None:
        self.id = run_id
        self.state_dir = state_dir

    @classmethod
    def start(
        cls,
        *,
        config_path: str | pathlib.Path | None,
        config_dict: dict[str, Any] | None,
        output_path: str | pathlib.Path | None = None,
        state_dir: str | pathlib.Path | None = None,
    ) -> RunTracker:
        root = get_state_dir(state_dir)
        _ensure_state_files(root)
        kind = _kind(config_dict)
        summary = _summary(config_dict, kind)
        run_id = os.environ.get("LFT_RUN_ID") or _new_run_id(summary)
        existing = next(
            (run for run in list_runs(root) if run.get("id") == run_id),
            None,
        )
        run = {
            "id": run_id,
            "created_at": _now(),
            "updated_at": _now(),
            "status": "running",
            "phase": "setup",
            "kind": kind,
            "config": _config_path(config_path),
            "backend": None,
            "backend_id": None,
            "output_dir": (
                str(pathlib.Path(output_path).expanduser()) if output_path else None
            ),
            "git_sha": _git_sha(),
            "summary": summary,
            "metrics": {},
            "progress": {
                **_PROGRESS_DEFAULT,
                "heartbeat_at": _now(),
            },
            "history": {key: value.copy() for key, value in _HISTORY_DEFAULT.items()},
            "log_refs": {},
            "backend_meta": {},
        }
        if existing:
            run = {**existing, **run}
            run["created_at"] = existing.get("created_at") or run["created_at"]
            for key in (
                "backend",
                "backend_id",
                "output_dir",
                "git_sha",
                "history",
                "log_refs",
                "backend_meta",
            ):
                if run.get(key) is None:
                    run[key] = existing.get(key)
            run["history"] = existing.get("history") or run["history"]
            run["log_refs"] = existing.get("log_refs") or run["log_refs"]
            run["backend_meta"] = existing.get("backend_meta") or run["backend_meta"]
        _upsert_run(run, root)
        return cls(run["id"], root)

    def load(self) -> dict[str, Any]:
        return load_run(self.id, self.state_dir)

    def update(self, **fields: Any) -> None:
        update_run_fields(self.id, state_dir=self.state_dir, **fields)

    def submitted(
        self,
        backend: str,
        backend_id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        log_refs: dict[str, Any] | None = None,
    ) -> None:
        record_backend_submission(
            self.id,
            backend=backend,
            backend_id=backend_id,
            metadata=metadata,
            log_refs=log_refs,
            state_dir=self.state_dir,
        )

    def completed(self, metrics: dict[str, Any] | None = None) -> None:
        fields: dict[str, Any] = {"status": "completed"}
        if metrics is not None:
            fields["metrics"] = metrics
        self.update(**fields, phase="completed")
        if metrics is not None:
            source = (
                "standalone_eval"
                if self.load().get("kind") == "eval"
                else "run_completed"
            )
            record_eval_result(
                self.id,
                metrics=metrics,
                status="completed",
                source=source,
                state_dir=self.state_dir,
            )

    def failed(self, exc: BaseException) -> None:
        self.update(
            status="failed",
            phase="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def update_run_fields(
    run_id: str,
    *,
    state_dir: str | pathlib.Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    def mutate(state: dict[str, Any]) -> dict[str, Any]:
        run = _find_run(state, run_id)
        run.update(_json_safe(fields))
        run["updated_at"] = _now()
        if "status" in fields and "phase" not in fields:
            run["phase"] = _phase_for_status(str(fields["status"]))
        return run.copy()

    return _mutate_state(mutate, state_dir)


def update_run_progress(
    run_id: str | None = None,
    *,
    phase: str | None = None,
    status: str | None = None,
    step: Any = None,
    epoch: Any = None,
    max_steps: Any = None,
    log: dict[str, Any] | None = None,
    source: str = "trainer",
    log_refs: dict[str, Any] | None = None,
    state_dir: str | pathlib.Path | None = None,
) -> dict[str, Any] | None:
    active_id = _active_run_id(run_id)
    if not active_id:
        return None

    coerced_step = _coerce_step(step)
    try:
        coerced_epoch = float(epoch) if epoch is not None else None
    except (TypeError, ValueError):
        coerced_epoch = None
    coerced_max_steps = _coerce_step(max_steps)
    log_metrics = _scalar_json_values(log)

    def mutate(state: dict[str, Any]) -> dict[str, Any] | None:
        try:
            run = _find_run(state, active_id)
        except FileNotFoundError:
            return None
        progress = run.setdefault("progress", _PROGRESS_DEFAULT.copy())
        history = run.setdefault(
            "history", {key: value.copy() for key, value in _HISTORY_DEFAULT.items()}
        )

        progress["heartbeat_at"] = _now()
        if phase:
            run["phase"] = phase
        if status:
            run["status"] = status
            if not phase:
                run["phase"] = _phase_for_status(status)
        if coerced_step is not None:
            progress["step"] = coerced_step
        if coerced_epoch is not None and math.isfinite(coerced_epoch):
            progress["epoch"] = coerced_epoch
        if coerced_max_steps is not None:
            progress["max_steps"] = coerced_max_steps
        if log_metrics:
            record = _record_log(
                step=coerced_step,
                epoch=progress.get("epoch"),
                metrics=log_metrics,
                source=source,
            )
            progress["last_log"] = record
            history["logs"] = _bounded_append(
                list(history.get("logs") or []), record, LOG_HISTORY_LIMIT
            )
        if log_refs:
            run.setdefault("log_refs", {}).update(_json_safe(log_refs))
        run["updated_at"] = _now()
        return run.copy()

    return _mutate_state(mutate, state_dir)


def record_eval_result(
    run_id: str | None = None,
    *,
    step: Any = None,
    metrics: dict[str, Any] | None = None,
    status: str = "completed",
    source: str = "eval",
    backend: str | None = None,
    backend_id: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    log_refs: dict[str, Any] | None = None,
    state_dir: str | pathlib.Path | None = None,
) -> dict[str, Any] | None:
    active_id = _active_run_id(run_id)
    if not active_id:
        return None
    coerced_step = _coerce_step(step)
    record = _record_eval(
        step=coerced_step,
        metrics=metrics,
        status=status,
        source=source,
        backend=backend,
        backend_id=backend_id,
        error=error,
        metadata=metadata,
    )

    def mutate(state: dict[str, Any]) -> dict[str, Any] | None:
        try:
            run = _find_run(state, active_id)
        except FileNotFoundError:
            return None
        progress = run.setdefault("progress", _PROGRESS_DEFAULT.copy())
        history = run.setdefault(
            "history", {key: value.copy() for key, value in _HISTORY_DEFAULT.items()}
        )
        progress["heartbeat_at"] = _now()
        progress["last_eval"] = record
        if coerced_step is not None:
            progress["step"] = coerced_step
        if status in {"submitted", "running"}:
            run["phase"] = "evaluating"
        elif status == "failed" and run.get("status") != "failed":
            run["phase"] = "evaluating"
        elif run.get("status") not in {"completed", "failed"}:
            run["phase"] = "training"
        history["evals"] = _bounded_append(
            list(history.get("evals") or []), record, None
        )
        if metrics:
            run.setdefault("metrics", {}).update(_json_safe(metrics))
        if log_refs:
            run.setdefault("log_refs", {}).update(_json_safe(log_refs))
        run["updated_at"] = _now()
        return run.copy()

    return _mutate_state(mutate, state_dir)


def record_checkpoint(
    run_id: str | None = None,
    *,
    path: str | pathlib.Path,
    step: Any = None,
    epoch: Any = None,
    source: str = "trainer",
    state_dir: str | pathlib.Path | None = None,
) -> dict[str, Any] | None:
    active_id = _active_run_id(run_id)
    if not active_id:
        return None
    coerced_step = _coerce_step(step)
    try:
        coerced_epoch = float(epoch) if epoch is not None else None
    except (TypeError, ValueError):
        coerced_epoch = None
    checkpoint = {
        "at": _now(),
        "path": str(path),
        "source": source,
    }
    if coerced_step is not None:
        checkpoint["step"] = coerced_step
    if coerced_epoch is not None and math.isfinite(coerced_epoch):
        checkpoint["epoch"] = coerced_epoch

    def mutate(state: dict[str, Any]) -> dict[str, Any] | None:
        try:
            run = _find_run(state, active_id)
        except FileNotFoundError:
            return None
        progress = run.setdefault("progress", _PROGRESS_DEFAULT.copy())
        history = run.setdefault(
            "history", {key: value.copy() for key, value in _HISTORY_DEFAULT.items()}
        )
        progress["heartbeat_at"] = _now()
        progress["last_checkpoint"] = _json_safe(checkpoint)
        history["checkpoints"] = _bounded_append(
            list(history.get("checkpoints") or []),
            _json_safe(checkpoint),
            CHECKPOINT_HISTORY_LIMIT,
        )
        if coerced_step is not None:
            progress["step"] = coerced_step
        run["phase"] = "saving"
        run["updated_at"] = _now()
        return run.copy()

    return _mutate_state(mutate, state_dir)


def update_log_refs(
    run_id: str | None = None,
    *,
    log_refs: dict[str, Any] | None = None,
    state_dir: str | pathlib.Path | None = None,
) -> dict[str, Any] | None:
    active_id = _active_run_id(run_id)
    if not active_id or not log_refs:
        return None

    def mutate(state: dict[str, Any]) -> dict[str, Any] | None:
        try:
            run = _find_run(state, active_id)
        except FileNotFoundError:
            return None
        run.setdefault("log_refs", {}).update(_json_safe(log_refs))
        run["updated_at"] = _now()
        return run.copy()

    return _mutate_state(mutate, state_dir)


def record_backend_submission(
    run_id: str,
    *,
    backend: str,
    backend_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    log_refs: dict[str, Any] | None = None,
    state_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    def mutate(state: dict[str, Any]) -> dict[str, Any]:
        run = _find_run(state, run_id)
        run["status"] = "submitted"
        run["phase"] = "submitted"
        run["backend"] = backend
        run["backend_id"] = backend_id
        if metadata:
            run.setdefault("backend_meta", {}).update(_json_safe(metadata))
        if log_refs:
            run.setdefault("log_refs", {}).update(_json_safe(log_refs))
        run["updated_at"] = _now()
        return run.copy()

    return _mutate_state(mutate, state_dir)


def list_runs(state_dir: str | pathlib.Path | None = None) -> list[dict[str, Any]]:
    return _load_state(state_dir).get("runs", [])


def load_run(
    run_id: str, state_dir: str | pathlib.Path | None = None
) -> dict[str, Any]:
    state = _load_state(state_dir)
    return _find_run(state, run_id).copy()


def render_run_list(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "No LFT runs found."
    lines = ["id\tstatus\tphase\tkind\tbackend\tbackend_id\tstep\tupdated_at\tsummary"]
    for run in runs:
        progress = run.get("progress") or {}
        lines.append(
            "\t".join(
                [
                    str(run.get("id") or ""),
                    str(run.get("status") or ""),
                    str(run.get("phase") or ""),
                    str(run.get("kind") or ""),
                    str(run.get("backend") or ""),
                    str(run.get("backend_id") or ""),
                    str(
                        progress.get("step") if progress.get("step") is not None else ""
                    ),
                    str(run.get("updated_at") or ""),
                    str(run.get("summary") or ""),
                ]
            )
        )
    return "\n".join(lines)


def render_run(run: dict[str, Any]) -> str:
    return yaml.safe_dump(run, sort_keys=False)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age(value: str | None) -> str:
    parsed = _parse_time(value)
    if parsed is None:
        return ""
    seconds = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"


def _first_metric(metrics: dict[str, Any] | None, names: tuple[str, ...]) -> str:
    for name in names:
        value = (metrics or {}).get(name)
        if value is not None:
            return str(value)
    return ""


def _summarize_metrics(metrics: dict[str, Any] | None, *, limit: int = 3) -> str:
    if not metrics:
        return ""
    parts = []
    for key in sorted(metrics)[:limit]:
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    suffix = " ..." if len(metrics) > limit else ""
    return ", ".join(parts) + suffix


def render_runs_report(runs: list[dict[str, Any]], *, limit: int | None = None) -> str:
    selected = runs[:limit] if limit is not None else runs
    if not selected:
        return "No LFT runs found."
    lines = [
        "id\tstatus\tphase\tbackend\tstep/max\theartbeat\tloss\tlast_eval\tlogs\tsummary"
    ]
    for run in selected:
        progress = run.get("progress") or {}
        last_log = progress.get("last_log") or {}
        log_metrics = last_log.get("metrics") or {}
        last_eval = progress.get("last_eval") or {}
        eval_metrics = last_eval.get("metrics") or {}
        step = progress.get("step")
        max_steps = progress.get("max_steps")
        step_max = (
            f"{step}/{max_steps}"
            if step is not None and max_steps is not None
            else str(step if step is not None else "")
        )
        log_refs = run.get("log_refs") or {}
        lines.append(
            "\t".join(
                [
                    str(run.get("id") or ""),
                    str(run.get("status") or ""),
                    str(run.get("phase") or ""),
                    str(run.get("backend") or ""),
                    step_max,
                    _age(progress.get("heartbeat_at")),
                    _first_metric(log_metrics, ("loss", "train_loss", "eval_loss")),
                    _summarize_metrics(eval_metrics),
                    ",".join(sorted(log_refs.keys())),
                    str(run.get("summary") or ""),
                ]
            )
        )
    return "\n".join(lines)


def _slurm_status(job_id: str) -> str | None:
    if shutil.which("sacct"):
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
            check=False,
        )
        states = [line.strip().split("|")[0] for line in result.stdout.splitlines()]
        states = [state for state in states if state]
        if states:
            return states[0]
    if shutil.which("squeue"):
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            check=False,
        )
        status = result.stdout.strip()
        if status:
            return status
    return None


def sync_run(
    run_id: str, state_dir: str | pathlib.Path | None = None
) -> dict[str, Any]:
    run = load_run(run_id, state_dir)
    remote_run = _read_remote_run_state(run)
    if remote_run:
        run = _merge_remote_run(run, remote_run, state_dir)
    if run.get("backend") == "slurm" and run.get("backend_id"):
        backend_status = _slurm_status(str(run["backend_id"]))
        if backend_status:
            status = run["status"]
            if backend_status in {"COMPLETED", "COMPLETING"}:
                status = "completed"
            elif backend_status in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                status = "failed"
            run = update_run_fields(
                str(run["id"]),
                state_dir=state_dir,
                status=status,
                backend_status=backend_status,
            )
    elif run.get("backend") == "kuberay":
        backend_status = _kuberay_status(run)
        if backend_status:
            status = run["status"]
            if backend_status.lower() in {"succeeded", "completed"}:
                status = "completed"
            elif backend_status.lower() in {"failed", "fail"}:
                status = "failed"
            run = update_run_fields(
                str(run["id"]),
                state_dir=state_dir,
                status=status,
                backend_status=backend_status,
            )
    return run


def _read_remote_run_state(run: dict[str, Any]) -> dict[str, Any] | None:
    meta = run.get("backend_meta") or {}
    remote_state_dir = meta.get("remote_state_dir")
    if not remote_state_dir:
        return None
    try:
        remote_root = pathlib.Path(str(remote_state_dir)).expanduser()
        remote_state_path = remote_root / "state.json"
        if not remote_state_path.exists():
            return None
        with remote_state_path.open() as handle:
            remote_state = json.load(handle)
    except Exception:
        return None
    for candidate in remote_state.get("runs", []):
        if candidate.get("id") == run.get("id"):
            return _normalize_run(candidate)
    return None


def _merge_remote_run(
    local_run: dict[str, Any],
    remote_run: dict[str, Any],
    state_dir: str | pathlib.Path | None,
) -> dict[str, Any]:
    merged = {
        **local_run,
        **remote_run,
        "backend": local_run.get("backend") or remote_run.get("backend"),
        "backend_id": local_run.get("backend_id") or remote_run.get("backend_id"),
        "backend_meta": {
            **(remote_run.get("backend_meta") or {}),
            **(local_run.get("backend_meta") or {}),
        },
        "log_refs": {
            **(remote_run.get("log_refs") or {}),
            **(local_run.get("log_refs") or {}),
        },
    }
    _upsert_run(merged, state_dir)
    return load_run(str(local_run["id"]), state_dir)


def _kuberay_status(run: dict[str, Any]) -> str | None:
    if not shutil.which("kubectl"):
        return None
    meta = run.get("backend_meta") or {}
    job_name = meta.get("job_name") or run.get("backend_id")
    namespace = meta.get("namespace") or "default"
    if not job_name:
        return None
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "rayjob",
            str(job_name),
            "-n",
            str(namespace),
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    status = payload.get("status") or {}
    for key in ("jobStatus", "jobDeploymentStatus", "rayClusterStatus"):
        if status.get(key):
            return str(status[key])
    return None


def read_memory(state_dir: str | pathlib.Path | None = None) -> str:
    _ensure_state_files(state_dir)
    return _memory_path(state_dir).read_text()


def add_memory_entry(
    text: str,
    *,
    ref: str | None = None,
    state_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
    _ensure_state_files(state_dir)
    path = _memory_path(state_dir)
    current = path.read_text()
    entry_lines = [f"## {_now()}"]
    if ref:
        entry_lines.extend(["", f"Refs: `{ref}`"])
    entry_lines.extend(["", text.strip(), ""])
    entry = "\n".join(entry_lines)
    header = "# LFT Memory\n\n"
    body = current.removeprefix(header)
    path.write_text(f"{header}{entry}\n{body.lstrip()}")
    return path
