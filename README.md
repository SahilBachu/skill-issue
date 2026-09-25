<p align="center">
  <img src="assets/banner.svg" alt="skill-issue: the right Agent Skill for every prompt, or none" width="100%">
</p>

<p align="center">
  <a href="https://github.com/SahilBachu/skill-issue/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/SahilBachu/skill-issue/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Claude Code plugin" src="https://img.shields.io/badge/Claude%20Code-plugin-d97757">
  <img alt="MCP server" src="https://img.shields.io/badge/MCP-server-6e56cf">
</p>

**skill-issue** picks the Agent Skills that matter for each prompt, or none, in a fraction of a
second, on your machine. It retrieves candidates from every installed skill (name, description and
body), asks a small calibrated model "is this skill relevant?" for each one, and tells your agent
to load only the ones that clear the bar.

<!-- demo -->

## Why

Coding agents decide which skill to use by reading every skill's one-line description in their
context. That works for a handful of skills. It degrades as the catalog grows: Claude Code gives
the skill listing about 1% of the context window and starts dropping descriptions when it
overflows. skill-issue moves the decision out of the prompt and into a fast local router that
reads the whole skill, not just its description.

## Install

### Claude Code (plugin, recommended)

```text
/plugin marketplace add SahilBachu/skill-issue
/plugin install skill-issue@skill-issue
```

The first session installs the router into the plugin's data folder in the background (needs
[uv](https://docs.astral.sh/uv/) or Python 3.10+) and downloads the models from Hugging Face.
Prompts work normally meanwhile. Check progress with `skill-issue doctor`.

### Codex, Gemini CLI, Cursor, and other MCP clients

```bash
uv tool install "skillissue[mcp] @ git+https://github.com/SahilBachu/skill-issue"
codex mcp add skill-issue -- skill-issue mcp            # Codex
gemini mcp add -s user skill-issue skill-issue mcp       # Gemini CLI
```

For Cursor, add this to `~/.cursor/mcp.json`:

```json
{ "mcpServers": { "skill-issue": { "command": "skill-issue", "args": ["mcp"] } } }
```

### Companion skill (any agent that supports Agent Skills)

```bash
npx skills add SahilBachu/skill-issue --skill skill-issue
```

It teaches the agent to call `find_skills` (or `skill-issue route`) and how to handle
suggestions safely.

> The repository is private during development. The commands above work once it is public, or
> today for anyone with access and GitHub credentials configured for git.

## How it works

```mermaid
flowchart LR
    P([Your prompt]) --> H[UserPromptSubmit hook<br/>stdlib only, 1.5 s cap]
    H -->|localhost + token| D{{Warm daemon}}
    subgraph D2 [daemon]
      direction LR
      C[(Installed skills<br/>user, project, plugins)] --> R[Hybrid retrieval<br/>BM25 + embeddings, RRF<br/>top 40]
      R --> G[Gate<br/>binary relevance per candidate<br/>top 12, one batch]
      G --> T[Calibrated threshold]
    end
    D --- D2
    T -->|0 to 3 skills| I[additionalContext<br/>load these skills]
    T -->|nothing fits| N[inject nothing]
    H -. daemon down or slow .-> N
```

1. **Retrieval.** BM25 over the skill's name, description and body (fields weighted by
   repetition) plus a small embedding model (`bge-small-en-v1.5`), fused with reciprocal rank
   fusion. The body matters: it is where a skill says what it actually does.
2. **Gate.** Each of the top candidates is scored independently: "would this skill help with this
   request?" The gate is a swappable model behind one interface. Scoring candidates one by one,
   instead of asking one model to pick from dozens of options, keeps it accurate as the catalog grows.
3. **Threshold.** Scores are Platt-calibrated on held-out data, so 0.8 means about 80%. Skills
   above the threshold are injected, three at most. Injecting nothing is the most common outcome.

The Claude Code hook is a standard-library Python script with a hard timeout. If the daemon is
starting, slow, or broken, the hook prints nothing and your prompt goes through untouched.

## Results

<!-- results:findings -->
<!-- /results:findings -->

### Exact match at 100 installed skills (test split, 527 prompts)

<!-- results:main -->
<!-- /results:main -->

### Scaling from 10 to 18,719 skills

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/scaling-dark.svg">
  <img alt="Exact match versus catalog size" src="assets/charts/scaling-light.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/false-injection-dark.svg">
  <img alt="False injection rate versus catalog size" src="assets/charts/false-injection-light.svg" width="100%">
</picture>

<!-- results:scaling -->
<!-- /results:scaling -->

### Claude Code with and without skill-issue

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/agent-dark.svg">
  <img alt="Claude Code skill selection with and without skill-issue" src="assets/charts/agent-light.svg" width="80%">
</picture>

<!-- results:agent -->
<!-- /results:agent -->

### Hand-written prompts (draft set, 64 prompts)

<!-- results:handwritten -->
<!-- /results:handwritten -->

### External benchmark: SkillRet

<!-- results:skillret -->
<!-- /results:skillret -->

### Latency and memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/latency-dark.svg">
  <img alt="Gate latency by model" src="assets/charts/latency-light.svg" width="80%">
</picture>

<!-- results:latency -->
<!-- /results:latency -->

### Calibration

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/calibration-dark.svg">
  <img alt="Reliability diagram" src="assets/charts/calibration-light.svg" width="45%">
</picture>

Full tables, including every system at every catalog size: [docs/results.md](docs/results.md).
Method, splits and leakage precautions: [docs/benchmark.md](docs/benchmark.md).

## Modes and trust

| Mode | What the router can suggest | Switch |
|---|---|---|
| `installed` (default) | skills you already have: `~/.claude/skills`, project `.claude/skills`, enabled plugin skills, `.agents/skills` | `skill-issue mode installed` |
| `approved` | the above, plus a vetted index of 1,003 public skills | `skill-issue mode approved` |

**The approved index** is built from pinned commits of eight public repositories. Each skill has a
trust tier:

| Tier | Meaning | Skills |
|---|---|---|
| 1 | official registry: `anthropics/skills`, `openai/skills`, `huggingface/skills`, `anthropics/claude-plugins-official` | 106 |
| 2 | its repository is linked from at least two well-known curated lists | 15 |
| 3 | neither, but passes our static scan with zero findings | 882 |

Every approved skill must also have a permissive license (MIT, Apache-2.0, BSD, ISC, CC-BY), and
no high-severity scan finding. 95 skills were left out; each reason is in
[`data/approved/rejected.jsonl`](data/approved/rejected.jsonl). Anthropic's docx, pdf, pptx and
xlsx skills are excluded because their license does not allow redistribution.

**Nothing is ever installed automatically.** When the router suggests an uninstalled skill, the
agent is told to show you the name, source, commit, tier, license, SHA-256 and script list, and to
ask. Only after you agree does it run:

```bash
skill-issue install <name> --confirm <first 12 chars of the sha256>
```

The installer downloads that exact commit, extracts only the skill folder, recomputes the hash
over every file, and refuses on any mismatch. The static scanner (`skill-issue scan <dir>`) checks
for network calls, credential and environment access, shell and dynamic code execution,
obfuscation, and prompt-injection text. It reads files and never runs them. See
[SECURITY.md](SECURITY.md) for the full threat model and the scanner's limits.

## CLI

| Command | What it does |
|---|---|
| `skill-issue route "prompt" [-v] [--json]` | route one prompt and show what would be injected |
| `skill-issue index [-v]` | list the skills the router can see |
| `skill-issue mode [installed\|approved]` | show or switch the mode |
| `skill-issue doctor` | check Python, torch, models, config, daemon, and measure latency |
| `skill-issue daemon start\|stop\|status\|restart` | manage the warm daemon (it also starts on first use) |
| `skill-issue models download` | fetch model weights now instead of on first use |
| `skill-issue approved [words]` | browse the approved index |
| `skill-issue install <name> --confirm <hash>` | install an approved skill |
| `skill-issue scan <dir>` | static scan of a skill folder |
| `skill-issue config show\|path\|set <key> <value>` | inspect or edit the config |
| `skill-issue mcp` | run the MCP server on stdio |

## Configuration

`~/.skill-issue/config.toml` (or `$SKILL_ISSUE_HOME/config.toml`). Everything is optional.

```toml
mode = "installed"            # or "approved"

[retrieval]
top_k = 40                    # candidates from retrieval
embed_model = "BAAI/bge-small-en-v1.5"   # or "none" for BM25 only
body_chars = 1500             # how much of each SKILL.md body retrieval reads

[gate]
name = "auto"                 # auto | cross-encoder | laya | retrieval | typesafe
                              # auto = gte-reranker-modernbert on a GPU or Apple Silicon, MiniLM-L6 on CPU
model = ""                    # HF repo id or local path; empty = the default for that gate
max_candidates = 12           # how many candidates the gate scores (latency scales with this)
threshold = 0.5               # leave unset to use the value tuned on the validation split
max_skills = 3
device = "auto"               # auto | cuda | mps | cpu

[daemon]
idle_timeout_min = 240

[sources]
claude_user = true            # ~/.claude/skills
claude_project = true         # <project>/.claude/skills
claude_plugins = true         # skills from enabled Claude Code plugins
agents = true                 # ~/.agents/skills and <project>/.agents/skills
extra_dirs = []
```

Environment: `SKILL_ISSUE_HOME` moves everything, `SKILL_ISSUE_DISABLE=1` turns the hook off,
`SKILL_ISSUE_HOOK_TIMEOUT_MS` changes the hook's cap (default 1500).

## Reproduce the results

```bash
git clone https://github.com/SahilBachu/skill-issue && cd skill-issue
uv sync
python -m bench.corpus fetch && python -m bench.corpus pool   # pinned repos + SkillRet
python -m bench.select_core                                     # core catalog and splits
python -m bench.run                                             # all systems, all catalog sizes
python -m training.build_pairs                                  # fine-tuning data
python -m training.finetune_cross --base Alibaba-NLP/gte-reranker-modernbert-base --out checkpoints/gte-mb-ft
python -m bench.run --systems gte-mb-ft
python -m bench.skillret_eval
python -m bench.hook_latency --gate cross-encoder --model checkpoints/gte-mb-ft
python -m bench.agent_baseline --arm vanilla --size 100 --n 100 # uses your Claude Code login
python -m bench.report                                          # charts + README tables
```

The prompt splits are committed (`data/bench/`), so you do not need to regenerate them. Their
SHA-256 hashes are in `data/bench/stats.json`. Regenerating them uses LLM subagents and will not
reproduce byte for byte; the instructions they followed are in
[`bench/GENERATION_INSTRUCTIONS.md`](bench/GENERATION_INSTRUCTIONS.md).

## FAQ

**Does anything leave my machine?** No. Routing is local. The network is used to download models
from Hugging Face, and for approved-mode installs you confirm. The optional TypeSafe gate calls
TypeSafe's API, and only if you set `TYPESAFE_API_KEY` and pick that gate.

**What if I have no GPU?** It runs on CPU. The default gate stays usable there; see the latency
table. Set `gate.device = "cpu"` to force it, or `gate.name = "retrieval"` for the fastest
model-free mode.

**Does it replace Claude Code's own skill selection?** No, it adds a hint. Claude still sees its
normal skill listing and still decides. The hook adds a short block naming the skills that fit.

**Why not let the main model pick?** It can, and at small catalogs it does well (see the results).
The router matters when you have many skills, or skills whose descriptions undersell them.

**Windows?** Linux, macOS and WSL2 are the supported targets and run in CI. Native Windows works
for development (this project was built on it) but the hook goes through Git Bash's `sh`.

## Roadmap

- Publish the fine-tuned gate weights on Hugging Face (pending).
- ONNX export of the default gate for faster CPU inference.
- A feedback loop: learn per-user thresholds from which suggested skills actually get loaded.
- More agents with native hooks (Codex and Gemini CLI hooks, once stable).
- Refresh the approved index on a schedule, with a diff of what changed and why.

## License

Apache-2.0. The approved index includes names, descriptions and short excerpts of third-party
skills under their own licenses; see [APPROVED_NOTICES.md](src/skillissue/data/APPROVED_NOTICES.md).
