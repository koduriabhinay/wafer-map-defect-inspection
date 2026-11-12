import numpy as np

from src.excursions import benjamini_hochberg, flag_excursion_lots, simulate_lots


def test_bh_rejects_nothing_when_all_p_equal_one():
    flags = benjamini_hochberg(np.ones(50), fdr=0.01)
    assert flags.shape == (50,)
    assert not flags.any()


def test_bh_rejects_strong_signals():
    p = np.concatenate([np.full(5, 1e-9), np.full(95, 0.9)])
    flags = benjamini_hochberg(p, fdr=0.01)
    assert flags[:5].all()
    assert not flags[5:].any()


def test_bh_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    flags = benjamini_hochberg(rng.random(20), fdr=0.05)
    assert flags.dtype == bool and flags.shape == (20,)


def test_flag_excursion_lots_finds_injected_excursions():
    defects, sizes, is_exc = simulate_lots(
        n_lots=200, lot_size=100, baseline_rate=0.05,
        excursion_fraction=0.1, excursion_rate=0.35, seed=7,
    )
    flags, pvals = flag_excursion_lots(defects, sizes, baseline_rate=0.05, fdr=0.01)
    assert pvals.shape == (200,)
    assert ((pvals >= 0) & (pvals <= 1)).all()
    # Most truly-excursion lots are caught...
    assert flags[is_exc].mean() > 0.8
    # ...and almost no in-control lot is falsely flagged at FDR 1%.
    assert flags[~is_exc].mean() < 0.05


def test_flag_excursion_lots_clean_process():
    defects, sizes, _ = simulate_lots(
        n_lots=200, lot_size=100, baseline_rate=0.05,
        excursion_fraction=0.0, seed=11,
    )
    flags, _ = flag_excursion_lots(defects, sizes, baseline_rate=0.05, fdr=0.01)
    assert flags.mean() < 0.02  # essentially no false alarms
