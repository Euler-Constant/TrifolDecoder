# Hierarchical trust prototype: build report

This submission implements the Team Lead's raw-syndrome hierarchy. D1 and D2 receive the same original observations. The gate does not correct observations. D3 is selectively invoked only on logical disagreement with a diagnostic and structural trigger. Failed specialist verification returns D2.

## Independent evaluation

Settings were selected using 6,000 tuning shots, then frozen before a new 20,000-shot evaluation.

| Basis | D2 LER | Hierarchical LER | D3 invocation rate |
|---|---:|---:|---:|
| X | 3.8875% | 3.7025% | 4.74% |
| Z | 4.0100% | 3.8425% | 4.96% |
| Overall | 3.94875% | 3.77250% | 4.85% |

Relative reduction in wrong bits: 4.46%. Absolute improvement: 0.17625 percentage points. Paired 95% interval: [0.10618, 0.24632] points, treating each shot as the independent unit.

D3 corrected 291 wrong D2 bits and introduced 150 new wrong bits, a net gain of 141. Agreement is not proof of correctness: 2193 wrong D2 bits occurred where D1 and D2 agreed. The disagreement-only gate cannot address those shared failures.

These are simulated validation results, not a Kaggle score or a state-of-the-art comparison. D3 is a broader candidate-search specialist, not an exact logical maximum-likelihood solver.

## Multithreading

On the same warmed 512-shot local workload, one thread took 4.3072 s and eight threads took 0.9525 s (medians of three runs). Measured speedup: 4.52x. Predictions and diagnostics were identical across thread counts.

Parallelism is across independent shots through Numba's native worker threads. A shared raw-input BP prefix avoids duplicate D1/D2 arithmetic; independent-run equivalence was checked exactly. There are no nested decoder thread pools. The code follows NUMBA_NUM_THREADS and does not hard-code the organiser's worker count.

## Submission checks

- 200,000 CSV rows, exact original test IDs and order, `id,L0,L1,L2,L3` columns, binary predictions.
- Original syndrome arrays remained unchanged during generation and tests.
- Agreement cases and unrouted cases return D2; specialist candidates are checked against the original detector equations.
- Exact small-case probability/cost checks and reordered-input checks passed.
- A separate fresh Python 3.12 environment reproduced 512 actual test rows exactly (256 per basis), and checked source/configuration hashes. A second full 200,000-shot fresh-environment run was not performed.
- All decode-time model logic is in decoder.py, kernels.py, circuit_model.py, and gate.json. It does not load fixtures or stored predictions.

Full test generation on eight threads took 268.2 s across both bases (138.0 s X, 130.1 s Z), including progress reporting, deduplication and repeated subset checks. This is a local generation measurement, not the organisers' median-of-five timing benchmark. Their Python 3.13 / NVIDIA B200 / 48-core environment remains untested.

CSV SHA-256: `13c51c546cdec5ae514ef8113fe3b22b97543b3f42b2a5f9a2aa188e565349a4`

The consistency verifier cannot certify the logical answer or guarantee fault tolerance. The candidate-weighted bit decisions are approximate. See README.md for architecture choices and reproduction commands.
