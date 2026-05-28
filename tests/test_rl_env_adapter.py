import pytest

from leap_finetune.rl.environments.adapter import (
    build_openenv_rollout_func,
    connect_openenv,
)
from leap_finetune.rl.environments.env_reward import env_reward

pytestmark = pytest.mark.configs


# === env_reward extractor ===


class TestEnvReward:
    def test_forwards_env_reward_kwarg(self):
        completions = [
            [{"role": "assistant", "content": "a"}],
            [{"role": "assistant", "content": "b"}],
        ]
        out = env_reward(completions, env_reward=[0.9, 0.1])
        assert out == [0.9, 0.1]

    def test_none_values_become_zero(self):
        out = env_reward(
            [[{"role": "assistant", "content": "a"}]],
            env_reward=[None],
        )
        assert out == [0.0]

    def test_missing_kwarg_returns_zeros(self):
        out = env_reward(
            [
                [{"role": "assistant", "content": "a"}],
                [{"role": "assistant", "content": "b"}],
            ]
        )
        assert out == [0.0, 0.0]


# === Fake env + fake trainer ===


class FakeEnvResult:
    def __init__(self, reward=None):
        self.reward = reward
        self.observation = None
        self.done = True


class FakeEnv:
    """Minimal synchronous env client imitating openenv.SyncEnvClient."""

    def __init__(self, rewards):
        self._rewards = list(rewards)
        self._reset_count = 0
        self._step_count = 0
        self._last_actions = []

    def reset(self, **kwargs):
        self._reset_count += 1
        return FakeEnvResult()

    def step(self, action):
        self._last_actions.append(action)
        reward = self._rewards[self._step_count]
        self._step_count += 1
        return FakeEnvResult(reward=reward)


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return f"decoded({len(ids)})"


class FakeTrainer:
    processing_class = FakeTokenizer()


# === rollout_func ===


class TestBuildOpenEnvRolloutFunc:
    def _monkeypatch_generate(self, monkeypatch, completions):
        """Replace trl.experimental.openenv.generate_rollout_completions."""
        import sys
        import types

        def fake_generate(trainer, prompts, **kwargs):
            # Return one fake completion per prompt
            return [
                {
                    "prompt_ids": [1, 2, 3],
                    "completion_ids": [10, 20],
                    "logprobs": [0.1, 0.2],
                    "text": completions[i],
                }
                for i in range(len(prompts))
            ]

        trl_mod = types.ModuleType("trl")
        exp_mod = types.ModuleType("trl.experimental")
        oe_mod = types.ModuleType("trl.experimental.openenv")
        oe_mod.generate_rollout_completions = fake_generate
        monkeypatch.setitem(sys.modules, "trl", trl_mod)
        monkeypatch.setitem(sys.modules, "trl.experimental", exp_mod)
        monkeypatch.setitem(sys.modules, "trl.experimental.openenv", oe_mod)

    def test_single_turn_dict_shape(self, monkeypatch):
        env = FakeEnv(rewards=[0.9, 0.1])
        rollout = build_openenv_rollout_func(env, max_turns=1)
        self._monkeypatch_generate(monkeypatch, ["hello", "world"])

        result = rollout(["p1", "p2"], FakeTrainer())
        assert set(result.keys()) == {
            "prompt_ids",
            "completion_ids",
            "logprobs",
            "env_reward",
        }
        assert result["env_reward"] == [0.9, 0.1]
        # Shape: one inner list per prompt
        assert len(result["prompt_ids"]) == 2
        assert len(result["completion_ids"]) == 2
        assert len(result["logprobs"]) == 2

    def test_env_step_receives_completion_text(self, monkeypatch):
        env = FakeEnv(rewards=[0.5, 0.5])
        rollout = build_openenv_rollout_func(env, max_turns=1)
        self._monkeypatch_generate(monkeypatch, ["first", "second"])

        rollout(["p1", "p2"], FakeTrainer())
        assert env._last_actions == [{"message": "first"}, {"message": "second"}]

    def test_custom_action_key(self, monkeypatch):
        env = FakeEnv(rewards=[0.5])
        rollout = build_openenv_rollout_func(env, max_turns=1, action_key="completion")
        self._monkeypatch_generate(monkeypatch, ["hi"])

        rollout(["p1"], FakeTrainer())
        assert env._last_actions == [{"completion": "hi"}]

    def test_reset_kwargs_forwarded(self, monkeypatch):
        class CountingEnv(FakeEnv):
            def __init__(self, rewards):
                super().__init__(rewards)
                self.reset_kwargs_seen = []

            def reset(self, **kwargs):
                self.reset_kwargs_seen.append(kwargs)
                return super().reset(**kwargs)

        env = CountingEnv(rewards=[0.5, 0.5])
        rollout = build_openenv_rollout_func(
            env, max_turns=1, reset_kwargs={"seed": 42}
        )
        self._monkeypatch_generate(monkeypatch, ["x", "y"])
        rollout(["p1", "p2"], FakeTrainer())

        assert env.reset_kwargs_seen == [{"seed": 42}, {"seed": 42}]

    def test_env_reward_fallback_to_zero_for_none(self, monkeypatch):
        env = FakeEnv(rewards=[None, 0.3])
        rollout = build_openenv_rollout_func(env, max_turns=1)
        self._monkeypatch_generate(monkeypatch, ["a", "b"])

        result = rollout(["p1", "p2"], FakeTrainer())
        assert result["env_reward"] == [0.0, 0.3]

    def test_multi_turn_not_supported(self):
        env = FakeEnv(rewards=[0.5])
        with pytest.raises(NotImplementedError, match="max_turns"):
            build_openenv_rollout_func(env, max_turns=3)


# === connect_openenv config validation ===


class TestConnectOpenEnv:
    def test_missing_all_sources_raises(self):
        with pytest.raises(ValueError, match="source.*base_url.*docker_image"):
            connect_openenv({})

    def test_missing_openenv_package_error_is_clear(self, monkeypatch):
        # Simulate openenv-core not being installed by faking ImportError
        import sys

        # Pre-insert a broken openenv module entry so `from openenv import AutoEnv` fails
        # (only works if the real package isn't already imported)
        if "openenv" not in sys.modules:
            import builtins

            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "openenv":
                    raise ImportError("fake: openenv-core not installed")
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", fake_import)
            with pytest.raises(ImportError, match="openenv-core is required"):
                connect_openenv({"source": "qgallouedec/echo_env"})
