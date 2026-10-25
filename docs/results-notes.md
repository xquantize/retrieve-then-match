# Results notes (validation split)

Split: 40% train / 20% validation / 40% test (seed 42, stratified).
All numbers below are on validation (n = 216 queries). Settings are chosen
on these numbers; final results come from a single run on the test split.
Reproduce with `scripts/reproduce.sh`.

Significance: paired tests on per-query outcomes (exact McNemar for hit@k,
sign-flip permutation for MRR), bootstrap 95% CIs, Holm correction across
the metrics within each comparison.

## Retrieval

| Run | hit@1 | MRR@20 |
|---|---|---|
| BM25, simple tokeniser (name+desc) | 0.671 | 0.757 |
| BM25, code-aware (name+desc) | 0.815 | 0.881 |
| + name-only queries | 0.861 | 0.913 |
| + manufacturer in corpus (best BM25) | 0.870 | 0.919 |
| Dense BGE-small, name-only queries | 0.787 | 0.860 |
| Dense BGE-small, name+desc queries | 0.819 | 0.885 |
| Hybrid RRF (BM25 + dense) | 0.870 | 0.918 |

## Matching (top-1 policy, threshold tuned on validation)

| Run | hit@1 | MRR@20 |
|---|---|---|
| BM25 scores only (no reranker) | 0.870 | 0.919 |
| Zero-shot cross-encoder, k=20 | 0.796 | 0.870 |
| Zero-shot cross-encoder, k=10 | 0.792 | 0.867 |
| Zero-shot cross-encoder, k=5 | 0.796 | 0.874 |

Every validation query has a true match, so the F1-optimal threshold accepts
every query's top candidate and top-1 F1 is effectively reranked hit@1.

## Findings

Significant:
- Code-aware tokenisation: +14.4 points hit@1 (95% CI [+9.3, +19.4]),
  34 queries fixed vs 3 broken, Holm p < 0.0001.
- BM25 beats dense with name-only queries: -8.3 points hit@1 for dense
  (Holm p = 0.004).
- Zero-shot cross-encoder reranking hurts: -7.4 points hit@1 vs BM25 alone
  (23 queries lost vs 7 gained, Holm p = 0.021).

Not significant (no detectable difference):
- BM25 vs hybrid RRF: 6 queries won each way (Holm p = 1.0).
- Adding manufacturer to the corpus (Holm p = 1.0).
- Cross-encoder with 5 vs 20 candidates (Holm p = 1.0), at half the latency.

Consistent direction, not significant at n = 216:
- Name-only queries for BM25: +4.6 points hit@1 (Holm p = 0.17;
  significant on the earlier 324-query split).
- BM25 vs dense with name+desc queries: -5.1 points (Holm p = 0.17).
