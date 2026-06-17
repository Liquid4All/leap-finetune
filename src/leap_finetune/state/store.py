from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

import yaml

STATE_VERSION = 1


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
        _write_state({"version": STATE_VERSION, "updated_at": _now(), "runs": []}, root)
    memory_path = root / "memory.md"
    if not memory_path.exists():
        memory_path.write_text("# LFT Memory\n\n")


def _load_state(state_dir: str | pathlib.Path | None = None) -> dict[str, Any]:
    _ensure_state_files(state_dir)
    with open(_state_path(state_dir)) as handle:
        state = json.load(handle)
    state.setdefault("version", STATE_VERSION)
    state.setdefault("runs", [])
    return state


def _write_state(state: dict[str, Any], state_dir: str | pathlib.Path | None = None) -> None:
    root = get_state_dir(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    state["version"] = STATE_VERSION
    state["updated_at"] = _now()
    path = root / "state.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, default=_json_default, indent=2) + "\n")
    tmp_path.replace(path)


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
        model = config_dict.get("checkpoint") or config_dict.get("model_name") or project
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


def _upsert_run(run: dict[str, Any], state_dir: str | pathlib.Path | None = None) -> None:
    state = _load_state(state_dir)
    runs = [existing for existing in state["runs"] if existing.get("id") != run["id"]]
    state["runs"] = [run, *runs]
    _write_state(state, state_dir)


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
        }
        if existing:
            run = {**existing, **run}
            run["created_at"] = existing.get("created_at") or run["created_at"]
            for key in ("backend", "backend_id", "output_dir", "git_sha"):
                if run.get(key) is None:
                    run[key] = existing.get(key)
        _upsert_run(run, root)
        return cls(run["id"], root)

    def load(self) -> dict[str, Any]:
        return load_run(self.id, self.state_dir)

    def update(self, **fields: Any) -> None:
        run = self.load()
        run.update(fields)
        run["updated_at"] = _now()
        _upsert_run(run, self.state_dir)

    def submitted(self, backend: str, backend_id: str | None = None) -> None:
        self.update(status="submitted", backend=backend, backend_id=backend_id)

    def completed(self, metrics: dict[str, Any] | None = None) -> None:
        fields: dict[str, Any] = {"status": "completed"}
        if metrics is not None:
            fields["metrics"] = metrics
        self.update(**fields)

    def failed(self, exc: BaseException) -> None:
        self.update(status="failed", error=f"{type(exc).__name__}: {exc}")


def list_runs(state_dir: str | pathlib.Path | None = None) -> list[dict[str, Any]]:
    return _load_state(state_dir).get("runs", [])


def load_run(run_id: str, state_dir: str | pathlib.Path | None = None) -> dict[str, Any]:
    for run in list_runs(state_dir):
        if run.get("id") == run_id:
            return run
    raise FileNotFoundError(f"Run not found: {run_id}")


def render_run_list(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "No LFT runs found."
    lines = ["id\tstatus\tkind\tbackend\tbackend_id\tupdated_at\tsummary"]
    for run in runs:
        lines.append(
            "\t".join(
                [
                    str(run.get("id") or ""),
                    str(run.get("status") or ""),
                    str(run.get("kind") or ""),
                    str(run.get("backend") or ""),
                    str(run.get("backend_id") or ""),
                    str(run.get("updated_at") or ""),
                    str(run.get("summary") or ""),
                ]
            )
        )
    return "\n".join(lines)


def render_run(run: dict[str, Any]) -> str:
    return yaml.safe_dump(run, sort_keys=False)


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


def sync_run(run_id: str, state_dir: str | pathlib.Path | None = None) -> dict[str, Any]:
    run = load_run(run_id, state_dir)
    if run.get("backend") == "slurm" and run.get("backend_id"):
        backend_status = _slurm_status(str(run["backend_id"]))
        if backend_status:
            status = run["status"]
            if backend_status in {"COMPLETED", "COMPLETING"}:
                status = "completed"
            elif backend_status in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                status = "failed"
            run.update(status=status, backend_status=backend_status, updated_at=_now())
            _upsert_run(run, state_dir)
    return run


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
