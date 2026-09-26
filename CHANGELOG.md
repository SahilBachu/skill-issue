# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- Hybrid retrieval: BM25 (bm25s) over skill name, description and body, plus a small embedding
  model, fused with reciprocal rank fusion.
- Swappable gate interface with calibrated probabilities: Laya decision model, any cross-encoder
  reranker, a retrieval-feature gate, and an optional TypeSafe (Jev) gate.
- Warm localhost daemon with lazy start, health check, per-start auth token and idle shutdown.
- Claude Code plugin and marketplace: `SessionStart` bootstrap and a fail-open
  `UserPromptSubmit` hook.
- MCP server with `find_skills` and `explain_route` for Codex, Gemini CLI, Cursor and others.
- Companion Agent Skill installable with `npx skills add`.
- Approved mode: 1,003 public skills from pinned repositories with trust tiers, license
  filtering, a static scanner, and hash-verified, confirm-first installs.
- CLI: `route`, `index`, `mode`, `doctor`, `daemon`, `install`, `scan`, `approved`, `models`,
  `config`, `mcp`.
- Benchmark: 300 core skills split by similarity cluster, 2,621 generated prompts, a
  hand-written draft set, catalog sizes from 10 to 18,719, SkillRet as an external set, and
  baselines including vanilla Claude Code.
- Fine-tuned gates (gte-reranker-modernbert-base and MiniLM-L6) shipped as `gates-v1` release
  assets, pinned by SHA-256, with fallback to the public base models.
- `cloud/`: run the GPU jobs on Colab or a local GPU (`cloud/local.sh`) and merge the results.

### Fixed

- JSONL readers split on newlines only; `str.splitlines()` also split on U+2028 inside skill
  text and broke every cloud scoring job.
- Re-ingesting cloud results no longer drops the calibration fitted into `checkpoints/`.
- The agent benchmark records the `UserPromptSubmit` hook output, not the last hook event.
- The SkillRouter reranker baseline keeps only last-position logits, so it fits a 6 GB GPU.
