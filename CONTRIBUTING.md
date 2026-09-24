# Contributing

Thanks for helping. Bug reports, benchmark prompts, scanner rules, and new gates are all welcome.

## Setup

```bash
git clone https://github.com/SahilBachu/skill-issue
cd skill-issue
uv sync                      # creates .venv with dev, bench and train groups
uv run pytest -q             # fast tests: no model downloads, no GPU
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

`uv sync` on Windows installs CUDA 12.8 torch wheels. On Linux and macOS the PyPI wheels are used
(CUDA on Linux, MPS on Apple Silicon). For a CPU-only environment:
`uv pip install --torch-backend cpu -e ".[mcp]"`.

## Layout

| Path | What lives there |
|---|---|
| `src/skillissue/` | the package: discovery, retrieval, gates, router, daemon, hook, MCP server, CLI |
| `src/skillissue/hook.py` | the Claude Code hook client. Standard library only, keep it that way |
| `src/skillissue/approved/` | approved index, static scanner, installer |
| `hooks/`, `.claude-plugin/` | the Claude Code plugin and marketplace |
| `skills/skill-issue/` | the companion Agent Skill (`npx skills add`) |
| `bench/` | benchmark: corpus, dataset build, systems, metrics, charts |
| `training/` | gate fine-tuning |
| `data/` | committed benchmark prompts and metadata. No third-party skill content |

## Ground rules

- **The hook must fail open.** Any change to `hook.py` or `hooks/run-hook.sh` needs a test
  showing the prompt still goes through, with no output, when the daemon is down, slow, or broken.
- **Never execute skill code** in indexing, scanning, or tests of third-party skills.
- **Numbers in the README come from scripts in this repo.** If you change the benchmark, rerun
  it (`python -m bench.run`, then `python -m bench.report`) and commit the new results JSON with the
  README change. Do not hand-edit results.
- **Test split stays held out.** Nothing under `data/bench/test.jsonl` or the test-split skills may
  be used for training, calibration, or threshold tuning.

## Adding benchmark prompts

Hand-written prompts go in `data/bench/handwritten_draft.txt` (one per line:
`labels<TAB>kind<TAB>prompt`), then run `python -m bench.handwritten`. Labels must be test-split
skill names. Explain anything unusual in the PR.

## Adding a gate

Subclass `skillissue.gates.base.Gate`, implement `raw_scores(prompt, candidates)`, register it in
`skillissue/gates/__init__.py`, add a `System` in `bench/systems.py`, and run the benchmark. The
harness calibrates it on the validation split and freezes it before the test split.

## Commits and PRs

Small focused PRs, a clear description, tests for behavior changes. CI must pass.
