import copy
import logging

import torch

from leap_finetune.data_loaders.image_loader import load_image

logger = logging.getLogger(__name__)


def _find_template(seq, template):
    """Yield start indices where template occurs in seq."""
    tlen = len(template)
    for i in range(len(seq) - tlen + 1):
        if seq[i : i + tlen] == template:
            yield i


def create_vlm_collate_fn(processor):
    """Create a collate function with assistant-only label masking.

    Only assistant content + <|im_end|> contribute to loss (matching liquid-vlm).
    Images are loaded as PIL and passed to the processor for resize/tiling.
    Bad samples are skipped with a warning instead of crashing the batch.
    """
    tokenizer = processor.tokenizer

    # ChatML: <|im_start|>assistant\n{content}<|im_end|>\n
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _build_labels(input_ids):
        """Mask everything except assistant content + <|im_end|> with -100."""
        labels = torch.full_like(input_ids, -100)
        batch_size, seq_len = input_ids.shape

        for b in range(batch_size):
            ids = input_ids[b].tolist()
            for tmpl_start in _find_template(ids, response_template_ids):
                content_start = tmpl_start + len(response_template_ids)
                j = content_start
                while j < seq_len and ids[j] != end_token_id:
                    j += 1
                if j >= seq_len:
                    # Truncated turn — no <|im_end|> found, skip to avoid
                    # unmasking garbage (padding / partial next turn)
                    continue
                # Unmask content + <|im_end|>
                content_end = j + 1
                labels[b, content_start:content_end] = input_ids[
                    b, content_start:content_end
                ]

        return labels

    def collate_fn(samples):
        valid_samples = []
        all_loaded_images = []
        skip_count = 0

        for raw in samples:
            # Trainer's dataloader yields {"messages": [...]} dicts from HF Dataset
            conversation = raw["messages"] if isinstance(raw, dict) else raw
            sample_copy = copy.deepcopy(conversation)
            loaded_images = []
            try:
                for message in sample_copy:
                    if message["role"] == "user":
                        for content in message["content"]:
                            if content["type"] == "image" and isinstance(
                                content["image"], str
                            ):
                                img = load_image(content["image"])
                                content["image"] = img
                                loaded_images.append(img)
                valid_samples.append(sample_copy)
                all_loaded_images.extend(loaded_images)
            except Exception as e:
                skip_count += 1
                logger.warning(f"Skipping sample in collate: {e}")
                for img in loaded_images:
                    if hasattr(img, "close"):
                        img.close()

        if skip_count > 0:
            logger.info(f"Collate skipped {skip_count}/{len(samples)} samples")

        if len(valid_samples) == 0:
            raise RuntimeError(
                f"Entire batch failed: all {len(samples)} samples had errors"
            )

        try:
            batch = processor.apply_chat_template(
                valid_samples,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            batch["labels"] = _build_labels(batch["input_ids"])
            return batch

        finally:
            for img in all_loaded_images:
                if hasattr(img, "close"):
                    img.close()
            all_loaded_images.clear()

    return collate_fn
