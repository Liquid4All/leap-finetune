"""TrainerCallback for ``async_eval.mode == "sidecar"``.

At each ``eval_steps`` the callback (rank 0) stages a checkpoint, renders
an sbatch script, and submits it. The sbatch job loads vLLM, runs every
configured benchmark, and back-fills the training run's wandb log at the
originating training step. Training itself never blocks on eval — except
for the optional sync step-0 short-circuit (see ``on_train_begin``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
from leap_finetune.evaluation.sbatch_template import render_sbatch_script
from leap_finetune.utils.logging_utils import is_rank_zero

logger = logging.getLogger(__name__)

_MARKER_NAME = ".in_flight"
_STAGING_KEEP = 10

# Allowlist of slurm states where the job is still running. Anything
# outside (including states we don't recognize) is treated as terminal
# so the marker doesn't get stranded forever — slurm's terminal set is
# larger than the obvious one (BOOT_FAIL, DEADLINE, REVOKED,
# SPECIAL_EXIT, ...) and may grow across versions.
_SACCT_ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "SUSPENDED",
        "COMPLETING",
        "CONFIGURING",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESV_DEL_HOLD",
    }
)
# Terminal-state set for _wait_for_job's substring exit check (sacct
# emits e.g. "CANCELLED by 12345" so we match on substring).
_SACCT_TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
        "SPECIAL_EXIT",
    }
)

# Env vars that leak from the Ray training worker and would break a fresh
# vLLM init in the eval subprocess. Stripped before sbatch / Popen so
# Slurm's per-job CUDA_VISIBLE_DEVICES allocation applies cleanly.
_LEAK_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
)


def _clean_subprocess_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _LEAK_ENV_VARS}


class SidecarEvalCallback(TrainerCallback):
    """Submit an sbatch eval job at every ``eval_steps``. Rank 0 only."""

    def __init__(
        self,
        *,
        benchmarks: list,
        cfg: AsyncEvalConfig,
        benchmark_configs: dict | None,
        output_dir: str | None,
        wandb_run_id: str | None,
        config_dir: str | None = None,
    ):
        super().__init__()
        self.benchmarks = benchmarks
        self.cfg = cfg
        self.benchmark_configs = benchmark_configs or {}
        self.output_dir = Path(output_dir) if output_dir else None
        self.wandb_run_id = wandb_run_id
        self.config_dir = config_dir
        self._consecutive_failures = 0
        self._disabled = False

    @property
    def _eval_dir(self) -> Path:
        assert self.output_dir is not None, "output_dir not set"
        d = self.output_dir / "_async_eval"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _modality(self) -> str:
        for b in self.benchmarks:
            if hasattr(b, "processor"):
                return "vlm"
        return "text"

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        # Step-0 baseline eval blocks training until the sidecar finishes.
        # Wandb's monotonic _step counter would otherwise reject a backwards
        # write at step=0 after training has logged past it.
        if not getattr(args, "eval_on_start", False):
            return
        if self._disabled or not is_rank_zero() or not self.benchmarks:
            return
        if not self.output_dir:
            return
        self._fire(model, state, args, wait_for_completion=True)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        # Fires before _maybe_log_save_evaluate's blocking val-loss compute
        # so the sidecar's vLLM startup overlaps with train-side val loss.
        if not control.should_evaluate:
            return
        if self._disabled or not is_rank_zero() or not self.benchmarks:
            return
        if not self.output_dir:
            logger.warning(
                "[async_eval/sidecar] output_dir not set; skipping step %d",
                state.global_step,
            )
            return
        self._fire(model, state, args)

    def _fire(
        self,
        model,
        state: TrainerState,
        args: TrainingArguments,
        wait_for_completion: bool = False,
    ) -> None:
        marker = self._eval_dir / _MARKER_NAME
        self._clear_marker_if_stale(marker)
        if marker.exists():
            if self.cfg.on_overlap == "skip":
                logger.warning(
                    "[async_eval/sidecar] previous eval in flight; skipping step %d",
                    state.global_step,
                )
                return
            logger.info(
                "[async_eval/sidecar] previous eval in flight; queueing step %d",
                state.global_step,
            )

        try:
            jobid = self._submit(model, state, args)
            self._consecutive_failures = 0
            if wait_for_completion and jobid:
                self._wait_for_job(jobid, state.global_step)
        except Exception:
            self._consecutive_failures += 1
            logger.exception(
                "[async_eval/sidecar] submission failed at step %d (%d consecutive)",
                state.global_step,
                self._consecutive_failures,
            )
            if self._consecutive_failures >= self.cfg.failure.max_consecutive:
                self._disabled = True
                logger.error(
                    "[async_eval/sidecar] disabling after %d consecutive failures",
                    self._consecutive_failures,
                )

    def _submit(
        self,
        model,
        state: TrainerState,
        args: TrainingArguments,
    ) -> str | None:
        eval_dir = self._eval_dir
        marker = eval_dir / _MARKER_NAME

        ckpt_root = eval_dir / "checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_root / f"step_{state.global_step}"

        # Each sbatch self-cleans its own checkpoint on EXIT. We also evict
        # stale dirs here so a crashed runner that skipped its trap can't
        # leak disk indefinitely; keep enough to support on_overlap=queue.
        if ckpt_path.exists():
            shutil.rmtree(ckpt_path, ignore_errors=True)
        existing = sorted(
            ckpt_root.glob("step_*"),
            key=lambda p: int(p.name.removeprefix("step_"))
            if p.name[5:].isdigit()
            else 0,
        )
        for stale in existing[:-_STAGING_KEEP]:
            shutil.rmtree(stale, ignore_errors=True)

        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(str(ckpt_path))
        # The sidecar needs the tokenizer/processor at load time to
        # re-construct the benchmarks; one is enough since all share it.
        for b in self.benchmarks:
            tk = getattr(b, "tokenizer", None) or getattr(b, "processor", None)
            if tk is not None:
                try:
                    tk.save_pretrained(str(ckpt_path))
                except Exception:
                    logger.debug(
                        "[async_eval/sidecar] tokenizer save_pretrained failed",
                        exc_info=True,
                    )
                break

        bench_json = ckpt_path / "_benchmark_configs.json"
        bench_json.write_text(json.dumps(self.benchmark_configs))

        wandb_project = os.environ.get("WANDB_PROJECT", "leap-finetune")
        submission = render_sbatch_script(
            output_dir=self.output_dir,
            trigger_step=state.global_step,
            checkpoint_path=ckpt_path,
            benchmark_configs_json=bench_json,
            modality=self._modality(),
            wandb_run_id=self.wandb_run_id,
            wandb_project=wandb_project,
            job_name=f"leap_eval_step_{state.global_step}",
            vllm_gpus=self.cfg.vllm_gpus,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            dtype=self.cfg.dtype,
            max_model_len=self.cfg.max_model_len,
            sbatch_partition=self.cfg.sbatch.partition,
            sbatch_account=self.cfg.sbatch.account,
            sbatch_time=self.cfg.sbatch.time,
            sbatch_extra_args=self.cfg.sbatch.extra_args,
        )

        clean_env = _clean_subprocess_env()

        max_attempts = max(1, self.cfg.failure.max_submit_attempts)
        backoff = max(0.0, self.cfg.failure.submit_retry_backoff)
        last_err: str = ""
        for attempt in range(1, max_attempts + 1):
            try:
                result = subprocess.run(
                    ["sbatch", str(submission.script_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=clean_env,
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    "`sbatch` not found on PATH; async_eval mode=sidecar "
                    "requires running under SLURM. Use mode=sync or "
                    "mode=reserved instead."
                ) from e

            if result.returncode == 0:
                logger.info(
                    "[async_eval/sidecar] submitted step %d (attempt %d/%d): %s",
                    state.global_step,
                    attempt,
                    max_attempts,
                    result.stdout.strip(),
                )
                m = re.search(r"Submitted batch job (\d+)", result.stdout)
                jobid = m.group(1) if m else ""
                # Write marker only AFTER successful submit; format is
                # "<jobid>:<step>" so _clear_marker_if_stale can sacct the job.
                marker.write_text(f"{jobid}:{state.global_step}")
                return jobid or None

            last_err = (result.stderr or result.stdout).strip()
            if attempt < max_attempts:
                sleep_s = backoff * (2 ** (attempt - 1))
                logger.warning(
                    "[async_eval/sidecar] sbatch attempt %d/%d failed at "
                    "step %d (exit %d): %s — retrying in %.1fs",
                    attempt,
                    max_attempts,
                    state.global_step,
                    result.returncode,
                    last_err[:200],
                    sleep_s,
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)

        raise RuntimeError(
            f"sbatch submission failed after {max_attempts} attempt(s): {last_err}"
        )

    def _clear_marker_if_stale(self, marker: Path) -> None:
        """Remove an orphan ``.in_flight`` marker whose job is no longer alive.

        The sidecar script's EXIT trap doesn't fire if slurm OOM-kills the
        step, NODE_FAILs, or the user ``scancel``s with ``--signal=KILL``;
        without recovery an orphan would block all future evals under
        ``on_overlap=skip``. We ask ``sacct`` whether the recorded jobid is
        terminal; missing-sacct falls back to a 6h mtime cutoff.
        """
        if not marker.exists():
            return
        try:
            content = marker.read_text().strip()
        except OSError:
            return

        jobid = content.split(":", 1)[0] if ":" in content else ""
        if jobid.isdigit():
            proc = None
            try:
                proc = subprocess.run(
                    ["sacct", "-j", jobid, "-o", "State", "-P", "-n"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass  # sacct unavailable; fall through to mtime check.
            if proc is not None and proc.returncode == 0 and proc.stdout.strip():
                # sacct gave a definitive answer. Use an allowlist of ACTIVE
                # states — anything else (recognized-terminal or unknown) is
                # safe to clear. The terminal set is larger than the obvious
                # one and grows across slurm versions; default-to-cleared
                # avoids stranding markers on BOOT_FAIL / DEADLINE / REVOKED
                # / SPECIAL_EXIT / future states we don't enumerate.
                states_seen: list[str] = []
                for line in proc.stdout.strip().splitlines():
                    state = line.strip().split(None, 1)[0].rstrip("+").upper()
                    states_seen.append(state)
                    if state in _SACCT_ACTIVE_STATES:
                        return  # at least one row alive; keep marker.
                logger.warning(
                    "[async_eval/sidecar] clearing stale marker "
                    "(job %s no longer active; sacct states: %s)",
                    jobid,
                    ",".join(states_seen),
                )
                marker.unlink(missing_ok=True)
                return

        try:
            age = time.time() - marker.stat().st_mtime
            if age > 6 * 3600:
                logger.warning(
                    "[async_eval/sidecar] clearing stale marker (mtime %.1fh old)",
                    age / 3600,
                )
                marker.unlink(missing_ok=True)
        except OSError:
            pass

    def _wait_for_job(
        self,
        jobid: str,
        trigger_step: int,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> None:
        """Block until the Slurm job leaves PENDING/RUNNING. Used for the
        sync step-0 path so wandb's _step counter stays aligned at 0."""
        logger.info(
            "[async_eval/sidecar] waiting for step-%d sidecar (job %s) "
            "before training proceeds",
            trigger_step,
            jobid,
        )
        start = time.time()
        while time.time() - start < timeout:
            result = subprocess.run(
                ["sacct", "-j", jobid, "-o", "State", "-n", "-P"],
                capture_output=True,
                text=True,
                check=False,
            )
            states = [s.strip() for s in result.stdout.splitlines() if s.strip()]
            main = states[0] if states else ""
            if any(t in main for t in _SACCT_TERMINAL_STATES):
                logger.info(
                    "[async_eval/sidecar] step-%d sidecar finished: %s (%.1fs)",
                    trigger_step,
                    main,
                    time.time() - start,
                )
                return
            time.sleep(poll_interval)
        logger.warning(
            "[async_eval/sidecar] step-%d sidecar still running after %ds; "
            "proceeding with training",
            trigger_step,
            timeout,
        )
