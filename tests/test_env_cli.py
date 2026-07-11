from leap_finetune.cli import env


def test_matching_pinned_wheel_detects_cuda_cp312_target():
    target = env.RuntimeTarget(
        backend="cuda",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_version="2.11.0+cu130",
        cuda_version="13.0",
        hip_version=None,
    )

    wheel = env._matching_pinned_wheel(target)

    assert wheel is not None
    assert wheel.backend == "cuda"


def test_matching_pinned_wheel_rejects_cuda_torch_mismatch():
    target = env.RuntimeTarget(
        backend="cuda",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_version="2.10.0+cu128",
        cuda_version="12.8",
        hip_version=None,
    )

    assert env._matching_pinned_wheel(target) is None


def test_matching_pinned_wheel_detects_rocm_cp312_target():
    target = env.RuntimeTarget(
        backend="rocm",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_version="2.10.0+git8514f05",
        cuda_version=None,
        hip_version="7.2.53211",
    )

    wheel = env._matching_pinned_wheel(target)

    assert wheel is not None
    assert wheel.backend == "rocm"


def test_fa2_status_reports_sdpa_fallback(monkeypatch, capsys):
    target = env.RuntimeTarget(
        backend="cuda",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_version="2.11.0+cu130",
        cuda_version="13.0",
        hip_version=None,
    )
    monkeypatch.setattr(env, "detect_runtime_target", lambda: target)
    monkeypatch.setattr(
        env,
        "_flash_attn_2_status",
        lambda: (False, "flash-attn import failed: broken extension"),
    )
    monkeypatch.setattr(env, "_package_version", lambda _: "2.8.3")

    exit_code = env.fa2_status(require=False)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "fa2_status: unavailable" in output
    assert "attn_implementation: sdpa" in output
    assert "broken extension" in output


def test_fa2_status_require_exits_nonzero(monkeypatch):
    target = env.RuntimeTarget(
        backend="unknown",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_version=None,
        cuda_version=None,
        hip_version=None,
    )
    monkeypatch.setattr(env, "detect_runtime_target", lambda: target)
    monkeypatch.setattr(env, "_flash_attn_2_status", lambda: (False, "missing"))
    monkeypatch.setattr(env, "_package_version", lambda _: None)

    assert env.fa2_status(require=True) == 1


def test_install_fa2_tries_matching_pin_first(monkeypatch):
    target = env.RuntimeTarget(
        backend="cuda",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_version="2.11.0+cu130",
        cuda_version="13.0",
        hip_version=None,
    )
    installs = []
    monkeypatch.setattr(env, "detect_runtime_target", lambda: target)
    monkeypatch.setattr(
        env, "_run_uv_pip_install", lambda *args: installs.append(args) or True
    )
    monkeypatch.setattr(env, "fa2_status", lambda *, require=False: 0)

    assert env.install_fa2() == 0

    assert len(installs) == 1
    assert installs[0][0].startswith("https://github.com/adithyaxx/flash-attention/")


def test_install_fa2_uses_binary_resolution_when_no_pin_matches(monkeypatch):
    target = env.RuntimeTarget(
        backend="unknown",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_version="2.11.0",
        cuda_version=None,
        hip_version=None,
    )
    installs = []
    monkeypatch.setattr(env, "detect_runtime_target", lambda: target)
    monkeypatch.setattr(
        env, "_run_uv_pip_install", lambda *args: installs.append(args) or True
    )
    monkeypatch.setattr(env, "fa2_status", lambda *, require=False: 0)

    assert env.install_fa2() == 0

    assert installs == [("flash-attn==2.8.3", "--only-binary", "flash-attn")]


def test_install_fa2_source_build_is_explicit(monkeypatch):
    target = env.RuntimeTarget(
        backend="unknown",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_version="2.11.0",
        cuda_version=None,
        hip_version=None,
    )
    installs = []

    def fake_install(*args):
        installs.append(args)
        return len(installs) == 2

    monkeypatch.setattr(env, "detect_runtime_target", lambda: target)
    monkeypatch.setattr(env, "_run_uv_pip_install", fake_install)
    monkeypatch.setattr(env, "fa2_status", lambda *, require=False: 0)

    assert env.install_fa2(allow_source_build=True) == 0

    assert installs == [
        ("flash-attn==2.8.3", "--only-binary", "flash-attn"),
        (
            "flash-attn==2.8.3",
            "--no-binary",
            "flash-attn",
            "--no-build-isolation-package",
            "flash-attn",
        ),
    ]
