# Proposal: One-Line Installation for leap-finetune

## Problem

Right now, a new user who wants to try `leap-finetune` has to do this:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv first
git clone <repository-url>
cd leap-finetune
uv sync
```

That is four commands before they can even run the tool. More importantly, **requiring a `git clone` signals that this is a work-in-progress research repo, not a production-ready tool**. Developers evaluating the tool will bounce off this friction before they ever run a training job.

The goal of this proposal is to reduce the install story to a single command, regardless of whether the user prefers `pip` or `uv`:

```bash
pip install leap-finetune
# or
uv tool install leap-finetune
```

## Why This Matters for Adoption

Tools in this space that have strong adoption (`trl`, `axolotl`, `unsloth`, `litgpt`) are all one-line installable from PyPI. Users expect this. A `git clone` workflow creates several real barriers:

- **Cognitive overhead**: the user has to manage a cloned repo alongside their own project.
- **Notebook and CI incompatibility**: `!git clone` in a Colab notebook or Docker image is awkward; `!pip install` is natural.
- **Discoverability**: PyPI search and `pip install` are how most developers find and evaluate tools. No PyPI listing means no organic discovery.
- **Trust signal**: a `pip install`-able package reads as maintained and stable. A clone-to-run repo reads as experimental.

The cost of this change is low (a few hours of work). The potential upside in adoption is high.

## What Needs to Change

The package already has the right foundation: a `pyproject.toml` with `hatchling` as the build backend and a `[project.scripts]` entry point for the CLI. Three targeted changes are needed to make it work correctly when installed rather than cloned.

### 1. Move `job_configs/` inside the package

Currently `job_configs/` lives at the repo root. When the package is installed from PyPI, the repo root does not exist. The bundled example configs need to travel with the package.

**Change:** move `job_configs/` to `src/leap_finetune/job_configs/`.

### 2. Fix path resolution in `constants.py`

`constants.py` currently computes `LEAP_FINETUNE_DIR` by walking three directory levels up from `__file__`. This works when the repo is cloned, but points to the wrong location when the package is installed into `site-packages`.

**Change:** replace the directory-walking approach with `importlib.resources`, which is the standard library mechanism for locating bundled package data regardless of how the package was installed.

Use a user-local directory for the tokenization cache (e.g., `~/.cache/leap-finetune/tokenized/`) instead of a path relative to the repo root.

### 3. Fix `config_resolver.py` step 3

The fallback lookup for bundled configs currently uses `LEAP_FINETUNE_DIR / "job_configs"`. This needs to be updated to use `importlib.resources` to find the configs bundled inside the installed package.

## What Stays the Same

- Users who clone the repo and run `uv sync` continue to work exactly as before. The lookup order in `config_resolver.py` checks local paths first, so a cloned-repo workflow is unaffected.
- The `[tool.uv] no-build-isolation-package = ["flash-attn"]` setting handles the `flash-attn` build requirement automatically for `uv` users. `pip` users on Linux will need to run `pip install flash-attn --no-build-isolation` before installing `leap-finetune`. This is a pre-existing constraint of `flash-attn` itself, not something introduced by these changes.

## Nice-to-Have (Out of Scope for This PR)

A `leap-finetune init` subcommand that writes a starter config file to the current directory. This would complete the getting-started flow:

```bash
pip install leap-finetune
leap-finetune init        # scaffolds a starter config in the current directory
leap-finetune my_job.yaml
```

This is a separate, incremental improvement and does not need to land in the same change.

## Summary

| | Before | After |
|---|---|---|
| Install steps | 4 (install uv, clone, cd, sync) | 1 |
| Works with pip | No | Yes |
| Works with uv tool install | No | Yes |
| Works in notebooks/Docker | No | Yes |
| Discoverable on PyPI | No | Yes |
| Code changes required | | ~3 small file changes |
