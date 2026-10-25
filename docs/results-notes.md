# Results notes (validation split)

Working notes from the validation split (n = 324 queries). Settings were chosen
on these numbers; final reported results come from a single run on the test split.

## Retrieval

| Method | Config | hit@1 | hit@10 | MRR@20 | p50 latency |
|---|---|---|---|---|---|
| BM25, simple tokeniser | name+desc | 0.685 | 0.941 | 0.771 | 0.07 ms |
| BM25, code-aware | name+desc | 0.818 | 0.988 | 0.884 | 0.10 ms |
| BM25, code-aware | name-only queries | 0.870 | 0.994 | 0.917 | 0.05 ms |
| BM25, code-aware | + manufacturer in corpus | 0.883 | 0.994 | 0.926 | 0.04 ms |
| Dense, BGE-small | name-only queries | 0.799 | 0.981 | 0.871 | 8.3 ms |
| Dense, BGE-small | name+desc queries | 0.821 | 0.981 | 0.885 | 9.8 ms |
| Hybrid, RRF | BM25 + dense | 0.880 | 1.000 | 0.922 | 19.4 ms |

## Findings (paired tests, Holm-corrected)

- **Code-aware tokenisation is the largest single gain.** hit@1 rises from 0.685
  to 0.818 (95% CI [+9.6, +17.3] points, McNemar p < 0.0001), fixing 46 queries
  while breaking 3. From hit@5 upward it never loses a query.
- **Name-only queries improve rank 1, not coverage.** +5.2 points hit@1
  (Holm p = 0.018) and higher MRR (p = 0.009); no significant change at hit@5+.
- **BM25 beats dense.** The tuned BM25 outperforms BGE-small in both field
  configurations (hit@1 0.883 vs 0.821, Holm p = 0.009; MRR p = 0.003).
- **Hybrid adds nothing measurable over BM25.** No metric differs significantly
  (all Holm p = 1.0), at roughly 500x the per-query latency.
- **Small gains did not survive testing.** Adding manufacturer (+1.2 points,
  Holm p = 0.875) and adding descriptions to dense queries (+2.2 points,
  Holm p = 1.0) are within noise at n = 324.

## Reproducibility

Identical configs reproduce identical fingerprints and results
(e.g. `1056691f277e`, `2c261ff5ddca`, `3a7f4e2ca73e`, `34435c57757c`).
