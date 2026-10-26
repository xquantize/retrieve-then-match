#!/usr/bin/env bash
# The single, final evaluation on the held-out test split.
# Run once, after docs/preregistration.md is committed. Settings are frozen.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run: commit all changes first, so results map to one commit."
  exit 1
fi

OUT="runs/final-test-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
exec > >(tee "$OUT/final_test.log") 2>&1
echo "commit: $(git rev-parse --short HEAD)"

MODEL="models/abt-buy-ms-marco-MiniLM-L-6-v2-ft-229d6fda27ea"
[[ -f "$MODEL/training_meta.json" ]] || { echo "missing $MODEL"; exit 1; }
R=(--split test --set "output.runs_dir=$OUT")
NAME_DESC=(--set "dataset.query_fields=[name, description]" --set "dataset.corpus_fields=[name, description]")
compare() { printf '\n--- %s\n' "$3"; uv run rtm compare --runs-dir "$OUT" "$1" "$2"; }

uv run rtm retrieve "${R[@]}" --tag bm25-simple --set retrieval.bm25.tokenizer=simple "${NAME_DESC[@]}"
uv run rtm retrieve "${R[@]}" --tag bm25-codeaware "${NAME_DESC[@]}"
uv run rtm retrieve "${R[@]}" --tag bm25-best
uv run rtm retrieve -m dense "${R[@]}" --tag dense-name
uv run rtm retrieve -m hybrid "${R[@]}" --tag hybrid
uv run rtm match "${R[@]}" --tag match-bm25 --set matching.scorer=retriever
uv run rtm match "${R[@]}" --tag match-ce-k20
uv run rtm match "${R[@]}" --tag match-ft-k20 --set "matching.model=$MODEL"
uv run rtm match "${R[@]}" --tag match-ft-k5 --set "matching.model=$MODEL" --set matching.candidate_k=5

printf '\n==================== Pre-registered comparisons ====================\n'
compare match-bm25     match-ft-k20   "H1 (primary): BM25 matcher -> fine-tuned cross-encoder"
compare match-ce-k20   match-ft-k20   "H2: zero-shot -> fine-tuned cross-encoder"
compare bm25-simple    bm25-codeaware "H3: simple -> code-aware tokeniser"
compare bm25-best      dense-name     "H4: BM25 -> dense"
compare bm25-best      hybrid         "H5: BM25 -> hybrid RRF"
compare match-ft-k20   match-ft-k5    "H6: fine-tuned candidates 20 -> 5"
uv run rtm runs --runs-dir "$OUT"
printf '\nSaved to %s/final_test.log\n' "$OUT"
