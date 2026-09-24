---
license: apache-2.0
base_model: Alibaba-NLP/gte-reranker-modernbert-base
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

# skill-issue gate (gte-reranker-modernbert-base, fine-tuned)

A cross-encoder that answers one question for a (request, skill) pair: **would this Agent Skill
help a coding agent with this request?** It is the default gate of
[skill-issue](https://github.com/SahilBachu/skill-issue), a local skill router for Claude Code,
Codex, Gemini CLI and Cursor.

Status: **not published yet.** This card ships with the checkpoint once the owner approves a
release.

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
threshold) and injects skills whose probability clears the threshold.

## Training

- Base: `Alibaba-NLP/gte-reranker-modernbert-base` (Apache-2.0, 150M parameters).
- Data: 44.7k (request, skill, label) pairs, 1 positive to about 8 negatives. Negatives are the
  candidates the router's own hybrid retriever returns, so the model learns to reject near misses.
  - skill-issue train split: 1,782 prompts over 180 train-only skills (positives, hard
    negatives, multi-skill, and no-skill prompts), written by Claude Opus 5.5 subagents.
  - SkillRet train split: 3,000 queries over SkillRet train skills (Apache-2.0).
- Leakage guard: every validation and test skill of the skill-issue benchmark, and each of their
  near-duplicates in the pool (same normalized name or embedding cosine at least 0.90), was removed
  from all training catalogs. SkillRet queries whose gold skill fell in that set were dropped.
- Objective: binary cross-entropy with positive weight 2, 2 epochs, learning rate 2e-5, effective
  batch 16, bf16, max length 384, best checkpoint by validation loss.
- Hardware: one laptop RTX 4050 (6 GB).

## Evaluation

See the benchmark tables in the skill-issue README and `docs/results.md`, produced by
`python -m bench.run --systems gte-mb-ft`. The test split (527 prompts, 75 skills never seen in
training) is never used for training, calibration, or threshold selection.

## Limitations

- English only.
- The training and test prompts are synthetic (LLM-written), plus a small hand-written set.
  Real user traffic is messier.
- It judges relevance from the name, description and the start of the body. Skills whose value is
  deep in the body can be under-scored.
- Calibration was fitted at catalog size 100. At very large catalogs the retriever hands the gate
  harder distractors and the false-injection rate rises; see the scaling results.
