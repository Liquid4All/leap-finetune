"""TrainerCallback for ``async_eval.mode == "reserved"``.

Rank 0 owns a daemon helper thread that:
1. Launches a vLLM OpenAI server on the dedicated eval GPUs (carved off
   the training pool by the driver at job start).
2. On each ``on_evaluate``: respawns the server with the latest
   checkpoint, runs all benchmarks via ``VLLMServerBackend``, pushes an
   ``EvalResult`` to an output queue.
3. ``on_log`` drains the queue and logs to wandb at the originating step.

Weight reload only supports ``respawn`` today; ``in_place`` is rejected
at construction with a clear error. Helper-thread exceptions never
propagate to training; after ``failure.max_consecutive`` failures the
callback disables itself.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
from leap_finetune.utils.logging_utils import is_rank_zero

logger = logging.getLogger(__name__)


@dataclass
class _EvalRequest:
    step: int
    ckpt_path: Path


@dataclass
class _EvalResult:
    step: int
    metrics: dict


class ReservedEvalCallback(TrainerCallback):
    """Long-running vLLM server on dedicated GPUs; helper thread drives it."""

    def __init__(
        self,
        *,
        benchmarks: list,
        cfg: AsyncEvalConfig,
        server_url: str,
        output_dir: str | None,
        eval_gpu_ids: str = "",
    ):
        super().__init__()
        if cfg.reserved.weight_reload != "respawn":
            raise NotImplementedError(
                f"async_eval.reserved.weight_reload="
                f"{cfg.reserved.weight_reload!r} not supported; use 'respawn'."
            )

        self.benchmarks = benchmarks
        self.cfg = cfg
        self.server_url = server_url
        self.output_dir = Path(output_dir) if output_dir else None
        self.eval_gpu_ids = eval_gpu_ids

        self._input_q: queue.Queue[_EvalRequest | None] = queue.Queue()
        self._output_q: queue.Queue[_EvalResult] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._disabled = False

    @property
    def _eval_dir(self) -> Path:
        assert self.output_dir is not None
        d = self.output_dir / "_async_eval"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run_loop,
                name="leap-async-eval",
                daemon=True,
            )
            self._thread.start()

    def _run_loop(self) -> None:
        """Helper thread main loop. Runs eval cycles serially."""
        from leap_finetune.evaluation.backend import VLLMServerBackend

        backend = VLLMServerBackend(self.server_url)
        while True:
            req = self._input_q.get()
            if req is None:
                break
            try:
                metrics = self._run_one_cycle(backend, req)
                self._output_q.put(_EvalResult(step=req.step, metrics=metrics))
            except Exception:
                logger.exception(
                    "[async_eval/reserved] cycle failed at step %d", req.step
                )
                self._output_q.put(_EvalResult(step=req.step, metrics={}))

    def _run_one_cycle(self, backend, req: _EvalRequest) -> dict:
        """Respawn the server with the new checkpoint, wait for /health,
        then run all benchmarks through ``backend``."""
        from leap_finetune.utils.vllm_server import _wait_for_health  # noqa: PLC2701

        self._respawn_server(req.ckpt_path)
        _wait_for_health(self.server_url, timeout=600.0, process=self._server_process)

        results: dict[str, float] = {}
        for bench in self.benchmarks:
            samples = bench.get_samples()
            if not samples:
                continue
            try:
                r = bench.evaluate_with_backend(backend, samples)
            except NotImplementedError as e:
                logger.warning(
                    "[async_eval/reserved] [%s] backend doesn't support: %s",
                    bench.name,
                    e,
                )
                continue
            except Exception:
                logger.exception(
                    "[async_eval/reserved] [%s] failed; other benchmarks continue",
                    bench.name,
                )
                continue
            for metric, total in r.metrics.items():
                avg = total / r.count if r.count > 0 else 0.0
                results[f"benchmark/{bench.name}/{metric}"] = avg

        return results

    # === Subprocess management (helper thread owns it) ===

    _server_process = None  # subprocess.Popen of trl vllm-serve

    def _respawn_server(self, ckpt_path: Path) -> None:
        import shlex
        import subprocess
        import sys
        import time
        from urllib.parse import urlparse

        # Tear down existing — then SLEEP to let CUDA fully release the GPU's
        # memory. Without this pause, the next vLLM startup can race the
        # kernel's allocator cleanup and fail engine-core init.
        if self._server_process is not None and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait(timeout=5)
            time.sleep(5)
        self._server_process = None

        port = urlparse(self.server_url).port or self.cfg.reserved.server_port
        if not self.eval_gpu_ids:
            raise RuntimeError(
                "eval_gpu_ids not set on the callback; the driver must populate "
                "train_loop_config['async_eval_gpu_ids'] when mode=reserved."
            )

        # Launch via the venv's own Python to guarantee the right interpreter
        # (a system-wide ``vllm`` shim can point at a different Python). The
        # OpenAI-compatible api_server exposes /v1/chat/completions.
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(ckpt_path),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--dtype",
            self.cfg.dtype,
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            # Stable served-model-name so clients can address respawned
            # servers by the same string across cycles.
            "--served-model-name",
            "default",
        ]
        if self.cfg.max_model_len is not None:
            cmd += ["--max-model-len", str(self.cfg.max_model_len)]

        # Strip distributed env vars leaking from the parent Ray worker
        # (LOCAL_RANK, RANK, etc.) before pinning CUDA_VISIBLE_DEVICES to
        # the carved-out eval GPUs.
        from leap_finetune.evaluation.sidecar_callback import _clean_subprocess_env

        env = _clean_subprocess_env()
        env["CUDA_VISIBLE_DEVICES"] = self.eval_gpu_ids

        # Append server stdout+stderr across respawns so failures stay debuggable.
        log_dir = self._eval_dir / "vllm_server"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "server.log"
        server_log = open(log_path, "ab")

        logger.info(
            "[async_eval/reserved] launching vLLM (log=%s): %s",
            log_path,
            " ".join(shlex.quote(c) for c in cmd),
        )
        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )

    # === TrainerCallback hooks ===

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if self._disabled or not is_rank_zero() or not self.benchmarks:
            return
        if not self.output_dir:
            logger.warning(
                "[async_eval/reserved] output_dir not set; skipping step %d",
                state.global_step,
            )
            return

        # Backpressure
        if not self._input_q.empty() and self.cfg.on_overlap == "skip":
            logger.warning(
                "[async_eval/reserved] previous eval still in flight; skipping step %d",
                state.global_step,
            )
            return

        try:
            self._ensure_thread()
            ckpt_path = self._save_checkpoint(model, state)
            self._input_q.put(_EvalRequest(step=state.global_step, ckpt_path=ckpt_path))
            self._consecutive_failures = 0
        except Exception:
            self._consecutive_failures += 1
            logger.exception(
                "[async_eval/reserved] submission failed at step %d (%d consecutive)",
                state.global_step,
                self._consecutive_failures,
            )
            if self._consecutive_failures >= self.cfg.failure.max_consecutive:
                self._disabled = True
                logger.error(
                    "[async_eval/reserved] disabling after %d consecutive failures",
                    self._consecutive_failures,
                )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if not is_rank_zero():
            return
        # Drain whatever's ready and log to wandb at the originating step
        while True:
            try:
                result = self._output_q.get_nowait()
            except queue.Empty:
                break
            self._log_to_wandb(result)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if not is_rank_zero():
            return
        # First drain — pick up anything that finished while training was ending
        while True:
            try:
                result = self._output_q.get_nowait()
            except queue.Empty:
                break
            self._log_to_wandb(result)

        # Send poison pill AFTER any in-flight requests; wait long enough for
        # one full eval cycle to complete (server respawn ~60s + eval ~30s).
        # This is the user-visible contract: if your eval was in-flight at
        # train-end, we'll wait for it instead of dropping it on the floor.
        if self._thread is not None and self._thread.is_alive():
            logger.info(
                "[async_eval/reserved] draining in-flight eval cycles (up to %ds)...",
                600,
            )
            self._input_q.put(None)
            self._thread.join(timeout=600)
            if self._thread.is_alive():
                logger.warning(
                    "[async_eval/reserved] helper thread did not exit within "
                    "drain window; proceeding to teardown"
                )

        # Second drain — pick up the results that arrived while we were
        # waiting for the helper thread to finish.
        while True:
            try:
                result = self._output_q.get_nowait()
            except queue.Empty:
                break
            self._log_to_wandb(result)

        if self._server_process is not None and self._server_process.poll() is None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=10)
            except Exception:
                pass

    # === Helpers ===

    def _save_checkpoint(self, model, state: TrainerState) -> Path:
        ckpt_root = self._eval_dir / "checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_root / f"step_{state.global_step}"

        if ckpt_path.exists():
            shutil.rmtree(ckpt_path, ignore_errors=True)

        # Bound disk: keep only the latest 2 staging checkpoints
        existing = sorted(
            ckpt_root.glob("step_*"),
            key=lambda p: int(p.name.removeprefix("step_"))
            if p.name[5:].isdigit()
            else 0,
        )
        for stale in existing[:-1]:
            shutil.rmtree(stale, ignore_errors=True)

        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(str(ckpt_path))

        # Save tokenizer/processor too — vLLM needs them at load time
        for b in self.benchmarks:
            tk = getattr(b, "tokenizer", None) or getattr(b, "processor", None)
            if tk is not None:
                try:
                    tk.save_pretrained(str(ckpt_path))
                except Exception:
                    logger.debug(
                        "[async_eval/reserved] tokenizer save_pretrained failed",
                        exc_info=True,
                    )
                break

        return ckpt_path

    def _log_to_wandb(self, result: _EvalResult) -> None:
        if not result.metrics:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            # Include train/global_step alongside benchmarks so dashboards
            # configured with train/global_step as the X axis render the
            # async benchmark points at the originating training step.
            payload = {**result.metrics, "train/global_step": result.step}
            # commit=False matches the sync callback's pattern at callback.py:128;
            # the next training-side log commits the step. Wandb may emit a
            # backwards-step warning if the trainer is past result.step — the
            # data still lands at the correct step in the history.
            wandb.log(payload, step=result.step, commit=False)
            logger.info(
                "[async_eval/reserved] logged %d metrics at step %d",
                len(result.metrics),
                result.step,
            )
        except ImportError:
            pass
        except Exception:
            logger.warning("[async_eval/reserved] wandb.log failed", exc_info=True)
