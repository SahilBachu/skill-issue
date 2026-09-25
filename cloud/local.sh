#!/usr/bin/env bash
# Run the notebook's GPU jobs on a local GPU instead of Colab, then ingest and rebuild the report.
#
#   bash cloud/local.sh            # our gates, then SkillRouter, then the Laya fine-tune
#   SKIP_LAYA=1 bash cloud/local.sh
#
# Same scripts and inputs as cloud/skill_issue_cloud.ipynb (python -m cloud.prepare builds them).
# Every step skips work that is already on disk, so rerunning after an interruption resumes.
set -uo pipefail
cd "$(dirname "$0")/.."

D=${D:-.cache/cloud/bundle/data}
OUT=${OUT:-.cache/cloud/out}
LOG=${LOG:-.cache/cloud/local.log}
PY=${PY:-uv run --no-sync python}
PAIRS="--pairs $D/pairs.jsonl --skills $D/skills.json --val $D/val_pairs.jsonl"
mkdir -p "$OUT"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
step() {  # step <name> <done-file or -> <command...>
    local name=$1 done=$2
    shift 2
    if [ "$done" != "-" ] && [ -s "$done" ]; then
        say "skip $name (have $done)"
        return 0
    fi
    say "start $name"
    local t0=$SECONDS
    if "$@" >>"$LOG" 2>&1; then
        say "done $name in $(((SECONDS - t0) / 60)) min"
    else
        say "FAILED $name (exit $?), continuing"
    fi
}

score() {  # score <system> <model spec> <bench|skillret>
    local f="$OUT/scores_$1_$3.json"
    # cloud.score resumes from a partial file, so only skip when it is complete (marked by .done).
    step "score $1 $3" "$f.done" bash -c "$PY -m cloud.score --model $2 --requests $D/requests_$3.jsonl --out $f && touch $f.done"
}

report() {
    step "ingest" - $PY -m cloud.ingest --from "$OUT"
    step "bench.run $*" - $PY -m bench.run --systems "$@"
    step "skillret_eval" - $PY -m bench.skillret_eval --device cuda --systems bm25 dense hybrid gte-mb-zs \
        $(for s in gte-mb-ft minilm-ft laya-ft; do [ -s "$OUT/scores_${s}_skillret.json.done" ] && echo "$s"; done)
    step "ingest (SkillRouter SkillRet rows)" - $PY -m cloud.ingest --from "$OUT"
    step "report" - $PY -m bench.report
}

say "=== local GPU run: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

# Keep Windows from idle-sleeping while the jobs run (ES_CONTINUOUS | ES_SYSTEM_REQUIRED, released on exit).
if [ "${OS:-}" = "Windows_NT" ] && [ -x .venv/Scripts/python.exe ]; then
    .venv/Scripts/python.exe -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep(10**9)" &
    AWAKE=$!
    trap 'kill $AWAKE 2>/dev/null' EXIT
fi

# Part 1: fine-tuned cross-encoders (checkpoints already trained) and zero-shot gte on SkillRet.
score gte-mb-ft "cross:$OUT/gte-mb-ft" bench
score gte-mb-ft "cross:$OUT/gte-mb-ft" skillret
score minilm-ft "cross:$OUT/minilm-ft" bench
score minilm-ft "cross:$OUT/minilm-ft" skillret
score gte-mb-zs "cross:Alibaba-NLP/gte-reranker-modernbert-base" skillret
report gte-mb-ft minilm-ft
say "=== part 1 finished ==="

# Optional hand-off: while $OUT/PAUSE exists (for example during the Claude Code agent benchmark, which
# needs the GPU for its gate), wait before the long jobs. Gives up waiting after PAUSE_MAX_MIN minutes.
waited=0
while [ -e "$OUT/PAUSE" ] && [ $waited -lt $((${PAUSE_MAX_MIN:-180} * 60)) ]; do
    [ $waited -eq 0 ] && say "paused (remove $OUT/PAUSE to continue)"
    sleep 30
    waited=$((waited + 30))
done
[ $waited -gt 0 ] && say "resumed after $((waited / 60)) min"

# Part 2: SkillRouter baseline (slowest on a small GPU: about 0.25 s per document to embed).
step "SkillRouter embed (bench)" "$OUT/sr_vecs.npz" \
    $PY -m cloud.sr_embed --docs "$D/sr_docs.jsonl" --queries "$D/sr_queries.jsonl" --out "$OUT/sr_vecs.npz"
step "SkillRouter rerank (bench)" "$OUT/sr_bench.json.done" bash -c \
    "$PY -m cloud.sr_bench --vecs $OUT/sr_vecs.npz --chunks $D/chunks.json --docs $D/sr_docs.jsonl --queries $D/sr_queries.jsonl --out $OUT/sr_bench.json && touch $OUT/sr_bench.json.done"
step "SkillRouter embed (SkillRet)" "$OUT/skillret_sr_vecs.npz" \
    $PY -m cloud.sr_embed --docs "$D/skillret_docs.jsonl" --queries "$D/skillret_queries.jsonl" --out "$OUT/skillret_sr_vecs.npz"
step "SkillRouter (SkillRet)" "$OUT/skillret_skillrouter.json" \
    $PY -m cloud.skillret_sr --vecs "$OUT/skillret_sr_vecs.npz" --docs "$D/skillret_docs.jsonl" \
    --queries "$D/skillret_queries.jsonl" --out "$OUT/skillret_skillrouter.json"

report skillrouter
say "=== part 2 finished ==="

# Part 3: Laya fine-tune (long), then its scores and a final report.
if [ -z "${SKIP_LAYA:-}" ]; then
    step "Laya fine-tune" "$OUT/laya-ft-trainlog.json" \
        $PY -m training.finetune_laya --out "$OUT/laya-ft" --epochs 1 --state "$OUT/laya-state.pt" $PAIRS
    if [ -s "$OUT/laya-ft-trainlog.json" ]; then
        score laya-ft "laya:$OUT/laya-ft" bench
        score laya-ft "laya:$OUT/laya-ft" skillret
        report laya-ft
    fi
    say "=== part 3 finished ==="
fi
