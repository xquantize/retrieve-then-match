"""Significance-test tests against hand-computed values."""

from __future__ import annotations

import pytest

from retrieve_then_match.stats import (
    holm,
    mcnemar_exact,
    paired_bootstrap_ci,
    paired_permutation_test,
)


def test_mcnemar_one_sided_discordance() -> None:
    a = [False] * 5 + [True] * 10
    b = [True] * 5 + [True] * 10  # B wins 5 discordant pairs, A wins none
    a_only, b_only, p = mcnemar_exact(a, b)
    assert (a_only, b_only) == (0, 5)
    assert p == pytest.approx(2 * 0.5**5)  # 0.0625


def test_mcnemar_balanced_and_identical() -> None:
    assert mcnemar_exact([True, False], [False, True])[2] == pytest.approx(1.0)
    assert mcnemar_exact([True, True], [True, True]) == (0, 0, 1.0)


def test_permutation_identical_is_one_and_large_effect_is_small() -> None:
    xs = [0.1 * i for i in range(30)]
    assert paired_permutation_test(xs, xs) == 1.0
    assert paired_permutation_test(xs, [x + 1.0 for x in xs]) < 0.001


def test_permutation_is_seeded() -> None:
    a = [0.0, 1.0, 0.5, 0.2, 0.9, 0.3]
    b = [0.1, 0.8, 0.9, 0.2, 1.0, 0.6]
    assert paired_permutation_test(a, b, seed=3) == paired_permutation_test(a, b, seed=3)


def test_bootstrap_ci_contains_true_shift() -> None:
    a = [float(i % 7) for i in range(200)]
    b = [x + 0.5 for x in a]
    low, high = paired_bootstrap_ci(a, b)
    assert low == pytest.approx(0.5) and high == pytest.approx(0.5)


def test_holm_matches_hand_computation() -> None:
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01=0.03, 2*0.02=0.04, 1*0.04=0.04
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    # sorted p: 0.3, 0.9 -> 2*0.3=0.6, max(0.6, 1*0.9)=0.9 (monotone)
    assert holm([0.9, 0.3]) == pytest.approx([0.9, 0.6])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # capped at 1


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])
    with pytest.raises(ValueError):
        paired_permutation_test([1.0], [1.0, 2.0])
