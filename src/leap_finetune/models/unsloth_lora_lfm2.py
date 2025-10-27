"""
UnslothLoRALFM2 - Weave Model wrapper for LFM2 with LoRA/PEFT support.

This module provides a Weave-compatible model wrapper for fine-tuning and inference
with LFM2 models using Unsloth for optimization.
"""

import time
from typing import Any, List, Optional

import weave
from peft import LoraConfig
from pydantic import Field


class UnslothLoRALFM2(weave.Model):
    """
    Weave Model wrapper for LFM2 with LoRA/PEFT support using Unsloth.

    This class stores/versions more parameters than just the model name.
    Especially relevant for fine-tuning (locally or aaS) because of specific parameters.
    """

    base_model: str
    revision: str | None

    # Initialization parameters
    is_training: bool
    peft_config: Optional[LoraConfig]
    cm_temperature: float
    max_seq_length: int
    load_in_4bit: bool
    inference_batch_size: int
    dtype: Any
    device: str

    # Generation parameters
    # Recommended Liquid settings!
    max_new_tokens: int = Field(
        default=128,
        description="[generation parameter] The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
    )
    temperature: float = Field(
        default=0.3,
        description="[generation parameter] The value used to modulate the next token probabilities.",
    )
    min_p: float = Field(
        default=0.15,
        description="[generation parameter] Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in the 0.99-0.8 range (use the opposite of normal `top_p` values).",
    )
    repetition_penalty: float = Field(
        default=1.05,
        description="[generation parameter]The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://huggingface.co/papers/1909.05858) for more details.",
    )

    # Provenance
    model: Any = Field(
        default=None, exclude=True
    )  # Exclude from serialization as their reprs are noisy
    tokenizer: Any = Field(
        default=None, exclude=True
    )  # Exclude from serialization as their reprs are noisy

    def model_post_init(self, __context):
        """Initialize model and tokenizer after Pydantic validation."""
        from unsloth import FastModel
        from transformers import Lfm2ForCausalLM
        import torch  # noqa: F401
        from peft import PeftModelForCausalLM

        # unsloth version (enable native 2x faster inference)
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.base_model,
                revision=self.revision,
                max_seq_length=self.max_seq_length,
                dtype=eval(self.dtype),  # e.g., torch.bfloat16
                auto_model=Lfm2ForCausalLM,
                load_in_4bit=self.load_in_4bit,
                # If PEFT config is not specified, let's assume a full fine-tuning
                full_finetuning=self.peft_config is None,
                device_map="balanced",
            )

        # Training vs inference setup (may mutate tokenizer: pad token/chat template/etc.)
        if self.is_training:
            # If not full fine-tuning not already a LoRA model, add LoRA adapters
            if self.peft_config is not None and not isinstance(
                self.model, PeftModelForCausalLM
            ):
                print("Using PEFT for finetuning")
                self.model = FastModel.get_peft_model(
                    self.model,
                    peft_config=self.peft_config,
                )
            FastModel.for_training(self.model)
        else:
            FastModel.for_inference(self.model)

    @weave.op()
    async def predict(self, messages: List[str]) -> str:
        """Async prediction interface."""
        return self.predict_sync(messages=messages)

    @weave.op()
    def predict_sync(self, messages: list[str]) -> str:
        """
        Synchronous prediction with usage tracking.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Generated text response
        """
        start_time = time.perf_counter()
        input_ids, output_ids, output = self._generate_response(messages)

        # Log usage information like OpenAI API spec
        prompt_tokens = input_ids.shape[1]
        completion_tokens = output_ids.shape[1] - input_ids.shape[1]
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        for k, v in {
            "usage.prompt_tokens": prompt_tokens,
            "usage.completion_tokens": completion_tokens,
            "usage.total_tokens": prompt_tokens + completion_tokens,
            "usage.latency_ms": latency_ms,
        }.items():
            weave.get_current_call().summary[k] = v

        return output

    def _generate_response(self, messages: list[str]) -> tuple[Any, Any, str]:
        """Internal method to generate response from messages."""
        # Include current hashes in Weave Trace
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            temperature=self.temperature,
            min_p=self.min_p,
        )
        decoded_outputs = self.tokenizer.batch_decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        output = "".join(decoded_outputs).strip()
        return input_ids, output_ids, output

    @weave.op()
    async def predict_stream(self, messages: List[str]) -> None:
        """
        Streaming prediction with usage tracking.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
        """
        # Track timing
        start_time = time.perf_counter()

        inputs, output_ids = await self._generate_response_stream(messages)

        # Count prompt tokens
        prompt_tokens = (
            inputs.shape[1]
            if hasattr(inputs, "shape")
            else inputs["input_ids"].shape[1]
        )
        # Count completion tokens (output excluding input)
        completion_tokens = output_ids.shape[1] - prompt_tokens
        # Track timing
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Log usage information like OpenAI API spec
        for k, v in {
            "usage.prompt_tokens": prompt_tokens,
            "usage.completion_tokens": completion_tokens,
            "usage.total_tokens": (prompt_tokens + completion_tokens),
            "usage.latency_ms": latency_ms,
        }.items():
            weave.get_current_call().summary[k] = v

    async def _generate_response_stream(self, messages: list[str]) -> tuple[Any, Any]:
        """Internal method to generate streaming response."""
        from transformers import TextStreamer

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True),
        )
        return inputs, output_ids
