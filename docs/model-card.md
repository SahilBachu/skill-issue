---
license: apache-2.0
base_model:
  - Alibaba-NLP/gte-reranker-modernbert-base
  - cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
tags:
  - cross-encoder
  - reranker
  - agent-skills
  - skill-routing
  - claude-code
datasets:
  - ThakiCloud/SKILLRET
language:
  - en
---

# skill-issue gates v1

Two cross-encoders that answer one question for a (request, skill) pair: **would this Agent Skill
help a coding agent with this request?** They are the gates of
[skill-issue](https://github.com/SahilBachu/skill-issue), a local skill router for Claude Code,
Codex, Gemini CLI and Cursor.

| File | Base model | Size (fp16) | Used when |
|---|---|---|---|
| `gte-skill-gate-v1.zip` | `Alibaba-NLP/gte-reranker-modernbert-base` (150M) | about 300 MB | a GPU or Apple Silicon is available |
| `minilm-skill-gate-v1.zip` | `cross-encoder/ms-marco-MiniLM-L6-v2` (22M) | about 45 MB | CPU only |

They are attached to the `gates-v1` GitHub release of the repository and downloaded on first use,
checked against a SHA-256 pinned in `src/skillissue/models.py`. If the download is not possible,
skill-issue falls back to the public base model with calibration fitted for it.

## Input format

Query: the user's message, truncated to 1,200 characters.
Document:

```
Skill: <name>
Description: <description>
Details: <first 300 characters of the SKILL.md body, code blocks removed>
```

The output is one relevance logit. skill-issue turns it into a probability with Platt scaling
fitted on a held-out validation split (`skill_issue_gate.json` holds `a`, `b` and the tuned
threshold) and injects the skills whose probability clears the threshold, three at most.

## Training

- Data: 44,703 (request, skill, label) pairs, about one positive to eight negatives. Negatives are
  the candidates the router's own hybrid retriever returns, so the model learns to reject near
  misses.
  - skill-issue train split: 1,782 prompts over 180 train-only skills (single-skill, hard
    negative, multi-skill and no-skill prompts), written by Claude subagents from the skills'
    real SKILL.md files.
  - SkillRet train split (Apache-2.0): 3,000 queries over SkillRet's train skills.
- Leakage guard: every validation and test skill of the skill-issue benchmark, and each of their
  near-duplicates in the pool (same normalized name, or embedding cosine of at least 0.90), was
  removed from every training catalog. SkillRet queries whose gold skill fell in that set were
  dropped. No validation or test prompt is used in training.
- Objective: binary cross-entropy with positive weight 2, one epoch (2,794 steps), learning rate
  2e-5, batch 16, max length 384, fp16, best checkpoint by validation loss.
- Hardware: one free-tier Colab T4 GPU.

## Evaluation

Exact match: the injected set equals the gold set, including injecting nothing when no skill
fits. Test split: 527 prompts whose 75 skills never appear in training. Calibration and the
threshold are fitted on the validation split only.

| Catalog size | gte v1 | gte base | MiniLM v1 | MiniLM base |
|---|---|---|---|---|
| 10 | 89.8% | 81.2% | 84.6% | 61.9% |
| 100 | 85.8% | 75.1% | 74.4% | 57.9% |
| 1,000 | 78.2% | 65.8% | 57.7% | 49.5% |
| 10,000 | 51.8% | 46.3% | 30.0% | 34.9% |
| 18,719 | 43.3% | 39.1% | 24.1% | 31.5% |

On the SkillRet test split (1,000 queries, 6,006 skills; reranking the top 20 of the hybrid
retriever), gte v1 reaches nDCG@10 0.742 against 0.705 for the base model. SkillRet test skills are
disjoint from its train skills, but the training mix did include SkillRet train queries, so this
is an in-distribution result, not a zero-shot one.

Full tables: `docs/results.md` in the repository. Reproduce with `python -m bench.run --systems
gte-mb-ft minilm-ft`.

## Limitations

- English only.
- The training and test prompts are synthetic (LLM-written). A small hand-written set is reported
  separately. Real user traffic is messier.
- It judges relevance from the name, description and the start of the body. Skills whose value is
  deep in the body can be under-scored.
- Calibration was fitted at catalog size 100. At very large catalogs the retriever hands the gate
  harder distractors and the false-injection rate rises; MiniLM v1 falls below its base model past
  10,000 skills.
