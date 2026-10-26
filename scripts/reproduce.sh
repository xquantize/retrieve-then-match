#!/usr/bin/env bash
# Re-run every validation-split experiment and the key paired comparisons.
#
#   scripts/reproduce.sh
#
# Each invocation writes to a fresh runs/reproduce-<UTC timestamp>/ folder,
# including reproduce.log with the full console output, so results from
# different reproductions never mix. The test split is never touched here.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="runs/reproduce-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
exec > >(tee "$OUT/reproduce.log") 2>&1

R=(--set "output.runs_dir=$OUT")
NAME_DESC=(--set "dataset.query_fields=[name, description]" --set "dataset.corpus_fields=[name, description]")

section() { printf '\n==================== %s ====================\n' "$1"; }
compare() {
  printf '\n--- %s\n' "$3"
  uv run rtm compare --runs-dir "$OUT" "$1" "$2"
}

section "BM25: tokeniser and field ablations"
uv run rtm retrieve "${R[@]}" --tag bm25-simple --set retrieval.bm25.tokenizer=simple "${NAME_DESC[@]}"
uv run rtm retrieve "${R[@]}" --tag bm25-codeaware "${NAME_DESC[@]}"
uv run rtm retrieve "${R[@]}" --tag bm25-nameonly --set "dataset.corpus_fields=[name, description]"
uv run rtm retrieve "${R[@]}" --tag bm25-best

section "Dense and hybrid retrieval"
uv run rtm retrieve -m dense "${R[@]}" --tag dense-name
uv run rtm retrieve -m dense "${R[@]}" --tag dense-desc --set "dataset.query_fields=[name, description]"
uv run rtm retrieve -m hybrid "${R[@]}" --tag hybrid

section "Matching (threshold tuned on val)"
uv run rtm match "${R[@]}" --tag match-bm25 --set matching.scorer=retriever
uv run rtm match "${R[@]}" --tag match-ce-k20
uv run rtm match "${R[@]}" --tag match-ce-k10 --set matching.candidate_k=10
uv run rtm match "${R[@]}" --tag match-ce-k5 --set matching.candidate_k=5

section "Fine-tuned cross-encoder (trained on the train split only)"
MODEL="$OUT/model"
uv run rtm train "${R[@]}" --output "$MODEL"
uv run rtm match "${R[@]}" --tag match-ft-k20 --set "matching.model=$MODEL"
uv run rtm match "${R[@]}" --tag match-ft-k5 --set "matching.model=$MODEL" --set matching.candidate_k=5

section "Paired comparisons (A -> B, Holm-corrected)"
compare bm25-simple    bm25-codeaware "BM25 tokeniser: simple -> code-aware"
compare bm25-codeaware bm25-nameonly  "BM25 queries: name+description -> name only"
compare bm25-nameonly  bm25-best      "BM25 corpus: + manufacturer"
compare bm25-best      dense-name     "BM25 -> dense (name-only queries)"
compare bm25-best      dense-desc     "BM25 -> dense (name+description queries)"
compare dense-name     dense-desc     "Dense queries: name only -> name+description"
compare bm25-best      hybrid         "BM25 -> hybrid RRF"
compare match-bm25     match-ce-k20   "Matcher: BM25 scores -> zero-shot cross-encoder"
compare match-ce-k20   match-ce-k5    "Cross-encoder candidates: 20 -> 5"
compare match-ce-k20   match-ft-k20   "Cross-encoder: zero-shot -> fine-tuned"
compare match-bm25     match-ft-k20   "Matcher: BM25 scores -> fine-tuned cross-encoder"
compare match-ft-k20   match-ft-k5    "Fine-tuned candidates: 20 -> 5"

section "All runs"
uv run rtm runs --runs-dir "$OUT"
printf '\nEverything above is saved in %s/reproduce.log\n' "$OUT"
