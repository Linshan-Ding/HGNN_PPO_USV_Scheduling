"""Shared statistics helpers for experiment scripts."""

import math
from typing import List, Tuple


def _rank_abs(values: List[float]) -> List[float]:
    """Average ranks for absolute non-zero differences."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank_less(ours_values: List[float],
                              other_values: List[float]) -> Tuple[float, float]:
    """
    One-sided Wilcoxon signed-rank test for H1: ours < other.

    Returns:
        statistic_w_plus, p_value
    """
    diffs = [p - r for p, r in zip(ours_values, other_values) if abs(p - r) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    abs_diffs = [abs(d) for d in diffs]
    ranks = _rank_abs(abs_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0)

    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return w_plus, 1.0

    # Continuity-corrected normal approximation. Small W+ supports ours < other.
    z = (w_plus - mean_w + 0.5) / math.sqrt(var_w)
    p_value = 0.5 * math.erfc(-z / math.sqrt(2.0))
    return w_plus, p_value


def parse_seeds(seed_text: str) -> List[int]:
    return [int(item.strip()) for item in seed_text.split(',') if item.strip()]
