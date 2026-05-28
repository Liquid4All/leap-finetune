from pathlib import Path

from leap_finetune.data_loading import ray_data_utils
from leap_finetune.data_loading.dataset_loader import DatasetLoader
from leap_finetune.data_loading.ray_data_utils import (
    _build_tokenization_cache_key,
    _load_tokenization_cache,
    _save_tokenization_cache,
    ray_dataset_to_hf,
)


class _RowShard:
    def __init__(self, rows):
        self._rows = rows

    def iter_rows(self):
        yield from self._rows


def test_ray_dataset_to_hf_materializes_rows():
    shard = _RowShard(
        [
            {"input_ids": [1, 2], "length": 2},
            {"input_ids": [3], "length": 1},
        ]
    )

    ds = ray_dataset_to_hf(shard)

    assert len(ds) == 2
    assert ds[0]["input_ids"] == [1, 2]
    assert ds[1]["input_ids"] == [3]


class _FakeRayDataset:
    def __init__(self, name):
        self.name = name

    def write_parquet(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{self.name}.parquet").write_text(self.name)


def test_cache_write_is_atomic_and_requires_success_marker(monkeypatch, tmp_path):
    cache_root = tmp_path / "tokenized"
    monkeypatch.setattr(ray_data_utils, "TOKENIZATION_CACHE_DIR", cache_root)

    loaded = []

    def fake_read_parquet(path):
        loaded.append(Path(path).name)
        return path

    monkeypatch.setattr(ray_data_utils.ray.data, "read_parquet", fake_read_parquet)

    fingerprint = "abc123"
    train = _FakeRayDataset("train")
    eval_ds = _FakeRayDataset("eval")
    key = {"dataset_path": "/data/train", "has_eval": True}

    assert _load_tokenization_cache(fingerprint) is None

    _save_tokenization_cache(fingerprint, train, eval_ds, key)

    cache_dir = cache_root / fingerprint
    assert (cache_dir / "_SUCCESS").exists()
    assert (cache_dir / "fingerprint.json").exists()
    assert not list(cache_root.glob(f"{fingerprint}.tmp-*"))

    cached = _load_tokenization_cache(fingerprint)
    assert cached == (str(cache_dir / "train"), str(cache_dir / "eval"))
    assert loaded == ["train", "eval"]


def test_sft_cache_key_includes_template_hash_and_overlength_policy(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text("template v1")
    loader = DatasetLoader(
        dataset_path="/data/train",
        dataset_type="sft",
        val_dataset_path="/data/eval",
        test_size=None,
    )

    fingerprint_v1, key_v1 = _build_tokenization_cache_key(
        loader,
        shuffle_seed=42,
        tokenizer_id="model",
        training_config={
            "max_length": 120000,
            "drop_overlength": True,
            "chat_template_path": str(template),
        },
    )

    template.write_text("template v2")
    fingerprint_v2, key_v2 = _build_tokenization_cache_key(
        loader,
        shuffle_seed=42,
        tokenizer_id="model",
        training_config={
            "max_length": 120000,
            "drop_overlength": True,
            "chat_template_path": str(template),
        },
    )

    assert key_v1["drop_overlength"] is True
    assert key_v1["chat_template_path_sha256"] != key_v2["chat_template_path_sha256"]
    assert fingerprint_v1 != fingerprint_v2
