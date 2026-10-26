# Pre-registration: final test-split evaluation

Written and committed before any test-split result was viewed. All settings
are frozen at this commit; nothing will be changed after seeing test results.

## Frozen setup
- Split: 40/20/40 train/val/test, seed 42 (test = 433 queries, never used before).
- Retrieval: BM25, code-aware tokeniser, name-only queries, corpus
  name+description+manufacturer (configs/default.yaml).
- Fine-tuned matcher: models/abt-buy-ms-marco-MiniLM-L-6-v2-ft-229d6fda27ea
  (trained on the train split only; 3 epochs, 4 BM25 hard negatives per query).
- Match thresholds are tuned on validation and applied unchanged to test.

## Primary hypothesis
H1. The fine-tuned cross-encoder (k=20) achieves higher hit@1 than the
BM25-only matcher. Test: exact McNemar, two-sided, alpha = 0.05, Holm-corrected
across the metrics of that comparison.

## Secondary comparisons (reported whatever the outcome)
H2. Fine-tuned vs zero-shot cross-encoder (k=20).
H3. Code-aware vs simple BM25 tokeniser (name+description fields).
H4. Best BM25 vs dense BGE-small (name-only queries).
H5. Best BM25 vs hybrid RRF (validation showed no difference).
H6. Fine-tuned cross-encoder with 5 vs 20 candidates (validation showed no
    difference; report the CI, not a claim of equivalence).

## Reporting
Every comparison above is reported, including non-significant and negative
results. Holm correction is within each comparison's metrics; the primary
hypothesis is H1.
