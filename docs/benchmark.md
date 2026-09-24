# Benchmark methodology

Everything here is produced by scripts in `bench/` and `training/`. Results live in
`bench/results/*.json`; the README tables and charts are generated from those files by
`python -m bench.report`.

## Skill pool

| Source | Skills | How we got them |
|---|---|---|
| 8 GitHub repos pinned by commit (`data/corpus/sources.lock.json`) | 1,098 | `python -m bench.corpus fetch` streams each tarball at the pinned commit and keeps text files only |
| SkillRet skills (ThakiCloud/SKILLRET, Apache-2.0) | 17,757 after removing copies of the GitHub skills | downloaded from Hugging Face |

Skills without a permissive license (MIT, Apache-2.0, BSD, ISC, CC-BY) are dropped, and so are
templates and skills with a description under 20 characters or a body under 200. That leaves
**18,755 skills**. Skill content is cached locally (`.cache/`, git-ignored) and never committed.
Only metadata and hashes are (`data/corpus/manifest.jsonl`).

## Core catalog and splits

`python -m bench.select_core` picks **300 core skills**. It starts with the official registries,
then goes round-robin across the other sources, skipping any skill whose embedding cosine to one
already picked is 0.85 or more, or whose name is already taken.

Skills are split **by similarity cluster**, not at random. Any two core skills with cosine ≥ 0.80
are linked, linked skills form clusters, and whole clusters go to one split. At 0.75, single-linkage
chaining produced one 140-skill cluster, which would have pushed a whole domain into train. At 0.80
the largest cluster has 10 skills.

| Split | Skills | Used for |
|---|---|---|
| train | 180 | fine-tuning the gates |
| val | 45 | calibration (Platt scaling), threshold, early stopping |
| test | 75 | reporting only. Never trained or tuned on |

## Prompts

Synthetic prompts were written by Claude Opus 5.5 subagents inside the development session, using
the instructions in [`bench/GENERATION_INSTRUCTIONS.md`](../bench/GENERATION_INSTRUCTIONS.md).
Each generator saw only one split's skills.

| Kind | What it is | Label |
|---|---|---|
| `positive` | a request the skill should handle (terse, normal, detailed) | the skill |
| `hard_negative_none` | same topic or words, needs no skill | none |
| `hard_negative_sibling` | belongs to a similar skill in the same split | the sibling |
| `multi` | needs two or three skills at once | all of them |
| `none` | everyday coding-agent requests that no core skill fits | none |

`python -m bench.build_dataset` validates every row (labels must belong to the batch's split),
removes exact duplicates, and drops any prompt with cosine ≥ 0.92 to a prompt in another split
(from the train or val side). Only about 5% of positive prompts mention their skill's name, so the
task cannot be solved by keyword matching on names.

| Split | Prompts | positive | hard neg (none) | hard neg (sibling) | multi | none |
|---|---|---|---|---|---|---|
| train | 1,782 | 1,079 | 176 | 179 | 80 | 268 |
| val | 312 | 135 | 45 | 45 | 20 | 67 |
| test | 527 | 225 | 75 | 75 | 40 | 112 |

**Hand-written subset.** `data/bench/handwritten_draft.txt` holds 64 prompts written by hand
against test-split skills: short, messy, realistic coding-agent messages. It is a **draft pending
review by the project owner** and is reported separately from the synthetic test split.

## Catalog sizes

For catalog size N, test prompts are packed into chunks. Each chunk gets its own catalog of N
skills: its required skills (gold labels, plus the skill each hard negative was written against),
then random distractors from the pool. Catalogs are shuffled so gold skills never sit first.
N = 100,000 means the whole pool (18,719 after held-out duplicates are removed).

Distractors exclude near-duplicates of the chunk's required skills (same normalized name, or
cosine ≥ 0.90). Without that, a copy of the gold skill from another repo would be scored as a
wrong answer. **This makes large catalogs somewhat easier than real life**, where duplicates do
exist.

## Systems

Every system is **retrieval + scorer + calibration + threshold**:

1. Retrieve the top 40 candidates (BM25, embeddings, hybrid RRF, or SkillRouter's embedder).
2. Score the top G (12 by default; 20 for SkillRouter, as its card recommends) with the scorer.
3. Fit Platt scaling on the **validation** split at N = 100, and choose the threshold that
   maximizes exact-set match on validation.
4. Freeze all of that, then run the **test** split at every catalog size.

Retrieval-only systems use their retrieval features as the score (a logistic regression over
cosine, log BM25, and RRF for hybrid), calibrated the same way.

## Metrics

| Metric | Meaning |
|---|---|
| exact match | the injected set equals the gold set (empty for "none" prompts). The headline number |
| top-1 | on prompts with a gold skill, the highest-probability candidate is gold |
| recall@k | fraction of gold skills in the retrieval top k |
| precision / recall / F1 | over injected skills |
| none accuracy | on no-skill prompts, the fraction where nothing was injected |
| false injection rate | the fraction of all prompts where at least one wrong skill was injected |
| ECE | expected calibration error over all gated (prompt, candidate) pairs, 15 bins |
| latency | see `bench/gate_latency.py` (gate only) and `bench/hook_latency.py` (end to end) |

## Leakage precautions

- Test skills and their pool near-duplicates are removed from every training catalog, and
  `training/build_pairs.py` asserts that no held-out skill id appears in the training pairs.
- SkillRet training queries whose gold skill is a near-duplicate of a val/test skill are dropped.
- Calibration and thresholds use validation only.
- Split files are hashed in `data/bench/stats.json`.

## Known biases

- The synthetic prompts and the vanilla Claude Code baseline both come from Claude models. That
  could help Claude (the prompts sound like what Claude expects) or hurt it (the prompts were
  written to be hard for a router). The hand-written set is the check on this.
- Hard negatives were written against their anchor skill. A system that is good at rejecting
  near misses benefits.
- The SkillRet test queries (written by Claude Opus 4.6 per its card) are long, detailed task
  descriptions, not short chat messages.
