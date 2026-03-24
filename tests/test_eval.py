import json

import pytest

pytestmark = pytest.mark.data


# === Eval Data Loaders ===


class TestEvalDataLoaders:
    def test_detect_format(self):
        from leap_finetune.evaluation.data_loaders import _detect_format

        assert _detect_format("data.jsonl") == "jsonl"
        assert _detect_format("data.ndjson") == "jsonl"
        assert _detect_format("data.json") == "json"
        assert _detect_format("data.parquet") == "parquet"
        assert _detect_format("data.pq") == "parquet"
        assert _detect_format("data.csv") == "csv"

    def test_load_jsonl(self, tmp_path):
        from leap_finetune.evaluation.data_loaders import _load_jsonl

        p = tmp_path / "test.jsonl"
        rows = [{"messages": [{"role": "user", "content": "hi"}]}, {"messages": []}]
        p.write_text("\n".join(json.dumps(r) for r in rows))

        result = _load_jsonl(str(p), limit=None)
        assert len(result) == 2

    def test_load_jsonl_with_limit(self, tmp_path):
        from leap_finetune.evaluation.data_loaders import _load_jsonl

        p = tmp_path / "test.jsonl"
        rows = [{"id": i} for i in range(10)]
        p.write_text("\n".join(json.dumps(r) for r in rows))

        result = _load_jsonl(str(p), limit=3)
        assert len(result) == 3

    def test_load_json(self, tmp_path):
        from leap_finetune.evaluation.data_loaders import _load_json

        p = tmp_path / "test.json"
        data = [{"id": 1}, {"id": 2}]
        p.write_text(json.dumps(data))

        result = _load_json(str(p), limit=None)
        assert len(result) == 2

    def test_load_json_non_array_raises(self, tmp_path):
        from leap_finetune.evaluation.data_loaders import _load_json

        p = tmp_path / "test.json"
        p.write_text(json.dumps({"key": "value"}))

        with pytest.raises(ValueError, match="top-level array"):
            _load_json(str(p), limit=None)

    def test_load_benchmark_samples_normalizes(self, tmp_path):
        from leap_finetune.evaluation.data_loaders import load_benchmark_samples

        p = tmp_path / "test.jsonl"
        row = {"conversation": [{"role": "user", "content": "hello"}]}
        p.write_text(json.dumps(row))

        samples = load_benchmark_samples(str(p))
        assert "messages" in samples[0]

    def test_convert_legacy_to_hf_format(self):
        from leap_finetune.evaluation.data_loaders import _convert_legacy_to_hf_format

        sample = {
            "messages": [{"role": "user", "content": "<image>What is this?"}],
            "images": ["/data/img.jpg"],
        }
        result = _convert_legacy_to_hf_format(sample)
        content = result["messages"][0]["content"]

        assert isinstance(content, list)
        assert content[0] == {"type": "image", "image": "/data/img.jpg"}
        assert content[1] == {"type": "text", "text": "What is this?"}
        assert "images" not in result

    def test_convert_legacy_preserves_structured(self):
        from leap_finetune.evaluation.data_loaders import _convert_legacy_to_hf_format

        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ]
        }
        result = _convert_legacy_to_hf_format(sample)
        assert result == sample

    def test_convert_legacy_prepends_image_root(self):
        from leap_finetune.evaluation.data_loaders import _convert_legacy_to_hf_format

        sample = {
            "messages": [{"role": "user", "content": "<image>describe"}],
            "images": ["relative/img.jpg"],
        }
        result = _convert_legacy_to_hf_format(sample, image_root="/data")
        img_item = result["messages"][0]["content"][0]
        assert img_item["image"] == "/data/relative/img.jpg"


# === Benchmark Base Class ===


class TestBenchmarkBase:
    def test_evaluate_excludes_failures_from_count(self):
        from leap_finetune.evaluation.base import Benchmark

        class FailingBenchmark(Benchmark):
            def load_samples(self):
                return []

            def score_sample(self, model, sample, device):
                if sample.get("fail"):
                    raise ValueError("bad sample")
                return 1.0

        bench = FailingBenchmark(name="test")
        samples = [{"id": 0}, {"fail": True}, {"id": 2}]
        result = bench.evaluate(None, samples, None)

        assert result.count == 2
        assert result.metrics["score"] == 2.0

    def test_get_samples_caches(self):
        from leap_finetune.evaluation.base import Benchmark

        call_count = 0

        class CountingBenchmark(Benchmark):
            def load_samples(self):
                nonlocal call_count
                call_count += 1
                return [{"id": 1}]

            def score_sample(self, model, sample, device):
                return 1.0

        bench = CountingBenchmark(name="test")
        bench.get_samples()
        bench.get_samples()
        assert call_count == 1
