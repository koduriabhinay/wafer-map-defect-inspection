"""Excursion-lot detection: per-lot binomial tests with Benjamini-Hochberg FDR.

A production lot is flagged when its defect rate is significantly higher than
the fab-wide baseline rate. Multiple testing across lots is controlled with
the Benjamini-Hochberg procedure at FDR = 1%.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def benjamini_hochberg(p_values: np.ndarray, fdr: float = 0.01) -> np.ndarray:
    """BH step-up procedure. Returns boolean mask of rejected null hypotheses."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * fdr
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    k_max = np.max(np.where(below)[0])
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: k_max + 1]] = True
    return rejected


def flag_excursion_lots(
    lot_defects: np.ndarray,
    lot_sizes: np.ndarray,
    baseline_rate: float,
    fdr: float = 0.01,
    alternative: str = "greater",
) -> tuple[np.ndarray, np.ndarray]:
    """Flag lots whose defect rate exceeds the fab-wide baseline.

    Args:
        lot_defects: defective wafer counts per lot.
        lot_sizes: total wafer counts per lot.
        baseline_rate: fab-wide defect rate under the null.
        fdr: Benjamini-Hochberg false discovery rate.
        alternative: one-sided "greater" test (excursions are excess defects).

    Returns:
        (flags, p_values): boolean excursion flags and raw per-lot p-values.
    """
    lot_defects = np.asarray(lot_defects, dtype=int)
    lot_sizes = np.asarray(lot_sizes, dtype=int)
    p_values = np.array(
        [
            stats.binomtest(int(k), int(n), baseline_rate, alternative=alternative).pvalue
            for k, n in zip(lot_defects, lot_sizes)
        ]
    )
    return benjamini_hochberg(p_values, fdr=fdr), p_values


def simulate_lots(
    n_lots: int,
    lot_size: int,
    baseline_rate: float,
    excursion_fraction: float = 0.1,
    excursion_rate: float = 0.35,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate lot defect counts. Returns (defects, sizes, is_true_excursion)."""
    rng = np.random.default_rng(seed)
    is_excursion = rng.random(n_lots) < excursion_fraction
    rates = np.where(is_excursion, excursion_rate, baseline_rate)
    defects = rng.binomial(lot_size, rates)
    return defects, np.full(n_lots, lot_size), is_excursion
