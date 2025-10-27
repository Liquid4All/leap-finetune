"""
Weave Evaluation Callback for periodic model evaluation during training.

This module provides:
- WeaveEvaluationCallback: Custom callback for running Weave evaluations periodically
- Scorer functions: Multiple metrics for evaluating model outputs
"""

import asyncio
import copy
import difflib
from datetime import datetime
from typing import List

import pandas as pd
import wandb
import weave
from transformers import TrainerCallback


import structlog_sentry_logger

LOGGER = structlog_sentry_logger.get_logger()

# =============================================================================
# Scorer Functions
# =============================================================================


@weave.op()
async def conversation_scorer(output: str, reference_text: str) -> float:
    """
    Simple scoring function that checks the similarity between the assistant text
    in the dataset and the generated output from the model given the same conversation
    user message.

    Args:
        output: model output for the given sample after fine-tuning
        reference_text: reference assistant response of the given sample

    Returns:
        Similarity ratio between the model output and assistant response in the text
    """
    return difflib.SequenceMatcher(None, output, reference_text).ratio()


@weave.op()
async def conversation_improvement_scorer(
    output: str, base_model_output: str, reference_text: str
) -> bool:
    """
    Simple scoring function that checks the difference between the assistant text
    in the dataset and the generated output from the model given the same conversation
    user message.

    Args:
        output: model output for the given sample after fine-tuning
        base_model_output: model output for the given sample before fine-tuning
        reference_text: reference assistant response of the given sample

    Returns:
        If the similarity ratio between the model output and assistant response in the
        text is greater as compared to the ratio for the base model
    """
    similarity_ratio_output = await conversation_scorer(output, reference_text)
    similarity_ratio_baseline = await conversation_scorer(
        base_model_output, reference_text
    )
    return similarity_ratio_output > similarity_ratio_baseline


@weave.op()
async def response_length_scorer(output: str) -> float:
    """
    Evaluate response length appropriateness (0-1 score).

    Args:
        output: model output to evaluate

    Returns:
        Score between 0 and 1 based on deviation from ideal length
    """
    ideal_length = 100  # Ideal response length
    actual_length = len(output.split())

    # Convert deviation from ideal length to score
    deviation = abs(actual_length - ideal_length) / ideal_length
    score = max(0, 1 - deviation)
    return score


@weave.op()
async def coherence_scorer(output: str) -> float:
    """
    Evaluate response coherence (simplified version).

    Args:
        output: model output to evaluate

    Returns:
        Coherence score between 0 and 1
    """
    # Count sentences
    sentences = output.split(".")
    if len(sentences) < 2:
        return 0.5

    # Calculate word overlap between sentences (simple coherence metric)
    word_sets = [set(sent.lower().split()) for sent in sentences if sent.strip()]
    if len(word_sets) < 2:
        return 0.5

    total_overlap = 0
    comparisons = 0

    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            if word_sets[i] and word_sets[j]:
                overlap = len(word_sets[i] & word_sets[j]) / min(
                    len(word_sets[i]), len(word_sets[j])
                )
                total_overlap += overlap
                comparisons += 1

    if comparisons > 0:
        return min(1.0, total_overlap / comparisons * 2)  # Scale adjustment
    return 0.5


@weave.op()
async def diversity_scorer(output: str) -> float:
    """
    Evaluate vocabulary diversity.

    Args:
        output: model output to evaluate

    Returns:
        Diversity ratio (unique words / total words)
    """
    words = output.lower().split()
    if not words:
        return 0.0

    unique_words = set(words)
    diversity_ratio = len(unique_words) / len(words)
    return diversity_ratio


# =============================================================================
# Weave Evaluation Callback
# =============================================================================


class WeaveEvaluationCallback(TrainerCallback):
    """
    Callback to run Weave Evaluation periodically and trace results.

    This callback enables periodic evaluation during training with full observability
    through Weave tracing and W&B logging.
    """

    def __init__(
        self,
        model_wrapper,
        eval_dataset,
        eval_interval: str = "epoch",
        eval_steps: int = 100,
        scorers: List = None,
        max_eval_samples: int = 20,
    ):
        """
        Args:
            model_wrapper: Weave Model object
            eval_dataset: Dataset for evaluation
            eval_interval: Evaluation interval ("epoch" or "steps")
            eval_steps: Step interval for step-based evaluation
            scorers: List of scorer functions for evaluation
            max_eval_samples: Maximum number of samples to use for evaluation
        """
        self.model_wrapper = model_wrapper
        self.model_wrapper_base = copy.deepcopy(model_wrapper)
        self.eval_dataset = eval_dataset
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.scorers = scorers or []
        self.max_eval_samples = max_eval_samples
        self.evaluation_results = []
        self.last_eval_step = 0
        self.total_callbacks_triggered = 0

        LOGGER.info(
            "✅ WeaveEvaluationCallback initialized:",
            eval_interval=eval_interval,
            eval_steps=f"{eval_steps if eval_interval == 'steps' else 'N/A (epoch-based)'}",
            num_scorers=len(self.scorers),
            max_eval_samples=max_eval_samples,
            eval_dataset_size=len(eval_dataset),
            model_wrapper_type=type(model_wrapper).__name__,
        )

        # Get evaluation data subset (same size as Final Test Evaluation)
        eval_samples = min(50, len(self.eval_dataset))
        eval_subset = self.eval_dataset.select(range(eval_samples))

        # Prepare dataset in the same format as Final Test Evaluation
        from leap_finetune import preprocess_dataset_for_weave_evaluation

        self.eval_data = eval_subset.map(
            preprocess_dataset_for_weave_evaluation,
            batched=True,
            fn_kwargs={"base_model": self.model_wrapper_base},
        ).to_list()

    @weave.op()
    async def run_evaluation(self, state):
        """Run evaluation and trace with Weave - same format as Final Test Evaluation"""

        LOGGER.debug(
            "run_evaluation invoked", step=state.global_step, epoch=state.epoch
        )
        self.total_callbacks_triggered += 1

        # Use model/tokenizer from the Weave model wrapper
        model = self.model_wrapper.model

        # Set model to evaluation mode
        was_training = model.training
        model.eval()

        # Create Weave Evaluation with same configuration
        evaluation = weave.Evaluation(
            dataset=self.eval_data,
            scorers=self.scorers,
            evaluation_name=f"Training Evaluation - Epoch {state.epoch:.1f} - Step {state.global_step}",
        )

        # Associate with current W&B run (same attributes as Final Test)
        with weave.attributes(
            {
                "wandb-run-id": wandb.run.id if wandb.run else None,
                "evaluation_type": "training_checkpoint",
                "global_step": state.global_step,
                "epoch": state.epoch,
                "training_loss": (
                    state.log_history[-1].get("loss", None)
                    if state.log_history
                    else None
                ),
            }
        ):
            # Run evaluation
            try:
                summary, call = await evaluation.evaluate.call(
                    evaluation, self.model_wrapper
                )
            except Exception as e:
                LOGGER.exception("Error in evaluation.evaluate.call", exc_info=e)
                raise

        # Save results
        result = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "weave_call_id": call.id,
            "num_samples": len(self.eval_data),
        }
        self.evaluation_results.append(result)

        # Log to W&B with same format as Final Test
        if wandb.run:
            # Log individual metrics
            for metric, value in summary.items():
                if isinstance(value, dict) and "mean" in value:
                    wandb.run.log(
                        {f"training_eval/{metric}": value["mean"]},
                        step=state.global_step,
                    )

            # Also log the full normalized summary
            wandb.run.log(
                {
                    f"training_eval/{key}": value
                    for key, value in pd.json_normalize(summary, sep="/")
                    .to_dict(orient="records")[0]
                    .items()
                },
                step=state.global_step,
            )

        # Print summary like Final Test Evaluation
        metric_value_dict = {
            f"{metric}": f"{value['mean']:.4f}"
            for metric, value in summary.items()
            if isinstance(value, dict) and "mean" in value
        }
        LOGGER.info(
            "\n📊 Evaluation results", step={state.global_step}, **metric_value_dict
        )

        # Restore training mode if it was set
        if was_training:
            model.train()

        return result

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics - use this instead of on_step_end"""
        if logs and self.eval_interval == "steps":
            LOGGER.debug(
                "on_log called",
                step=state.global_step,
                log_keys=(list(logs.keys()) if logs else "None"),
            )

            # Check if we're at an evaluation step
            if (
                state.global_step % self.eval_steps == 0
                or state.global_step == 1  # Baseline evaluation on first step
            ):
                if (
                    state.global_step != self.last_eval_step
                ):  # Prevent duplicate evaluation
                    LOGGER.info(
                        "🔍 Running Weave evaluation...", step=state.global_step
                    )
                    try:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            # Create a new event loop for this thread if none exists
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(self.run_evaluation(state))
                        self.last_eval_step = state.global_step
                        LOGGER.info(
                            "✅ Evaluation completed",
                            num_metrics_computed=len(result["summary"]),
                        )
                    except Exception as e:
                        LOGGER.exception("❌ Error during evaluation", exc_info=e)
                        import traceback

                        traceback.print_exc()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Process at epoch end"""
        if self.eval_interval == "epoch":
            LOGGER.info(
                "🔍 Running Weave evaluation (epoch completed)", epoch=state.epoch
            )
            # Run async task synchronously
            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(self.run_evaluation(state))
                LOGGER.info("✅ Evaluation completed", summary=result["summary"])
            except Exception as e:
                LOGGER.exception("❌ Error during evaluation", exc_info=e)
                import traceback

                traceback.print_exc()

    def on_evaluate(self, args, state, control, **kwargs):
        """Process during standard HF evaluation phase - this is called when eval_loss is computed"""
        LOGGER.info("🔍 Standard HF evaluation completed", step=state.global_step)

        # Run our custom Weave evaluation after HF evaluation
        if (
            state.global_step != self.last_eval_step
        ):  # Prevent duplicate if already run via on_log
            LOGGER.info("Running Weave evaluation...")
            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(self.run_evaluation(state))
                self.last_eval_step = state.global_step
                LOGGER.info(
                    "✅ Weave evaluation completed",
                    num_metrics_computed=len(result["summary"]),
                )
            except Exception as e:
                LOGGER.exception("❌ Error during Weave evaluation", exc_info=e)
                import traceback

                traceback.print_exc()

    def on_train_begin(self, args, state, control, **kwargs):
        """Run evaluation at training start for baseline"""
        self.on_evaluate(args, state, control, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        """Display evaluation results summary at training end"""
        if self.evaluation_results:
            # Display summary metrics from each evaluation
            d = {
                f"{i + 1}": {"step": result["step"], "epoch": f"{result['epoch']:.2f})"}
                | {
                    metric: f"{value['mean']:.4f}"
                    for metric, value in result["summary"].items()
                    if isinstance(value, dict) and "mean" in value
                }
                for i, result in enumerate(self.evaluation_results)
            }
            LOGGER.info(
                "📊 Evaluation Results Summary",
                total_callbacks_triggered=self.total_callbacks_triggered,
                final_last_eval_step=self.last_eval_step,
                num_evals_performed=len(self.evaluation_results),
                **d,
            )
            LOGGER.info(
                "💡 Check Weave UI for automatic visualizations (radar charts, bar charts, traces)"
            )
        else:
            LOGGER.warning("⚠️ No evaluation results collected during training")
