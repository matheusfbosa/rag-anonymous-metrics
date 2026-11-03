import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW

ALPHA = 0.05


def delta_series(paired: pd.DataFrame, col: str) -> pd.Series:
    """Per-question On-Demand minus Offline, indexed by question id."""
    off = pd.to_numeric(paired[f"{col}_offline"], errors="coerce")
    ond = pd.to_numeric(paired[f"{col}_ondemand"], errors="coerce")
    return pd.Series(
        (ond - off).to_numpy(dtype="float64"),
        index=paired["question_id"].to_numpy(),
    ).dropna()


def contrast_series(
    paired_a: pd.DataFrame, paired_b: pd.DataFrame, col: str
) -> pd.Series:
    """Per-question difference in differences, aligned by question id."""
    return (delta_series(paired_a, col) - delta_series(paired_b, col)).dropna()


def mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    """Mean and t confidence interval at 1 - ALPHA."""
    d = values.to_numpy(dtype="float64")
    if d.size < 2:
        return float("nan"), float("nan"), float("nan"), int(d.size)
    ds = DescrStatsW(d)
    lo, hi = ds.tconfint_mean(ALPHA)
    return float(ds.mean), float(lo), float(hi), int(d.size)


def mean_test(values: pd.Series) -> tuple[float, float, float, int, float]:
    """`mean_ci` plus the one-sample t-test p against zero."""
    mean, lo, hi, n = mean_ci(values)
    d = values.to_numpy(dtype="float64")
    has_spread = n > 1 and d.std() > 0
    p = float(DescrStatsW(d).ttest_mean(0)[1]) if has_spread else float("nan")
    return mean, lo, hi, n, p


def bonferroni(p: float, tests: int) -> float:
    return min(1.0, p * tests)


def resampling_test(
    values: pd.Series, reps: int = 20_000, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI and sign-flip permutation p, without normality."""
    d = values.to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (reps, d.size)).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    flipped = (rng.choice([-1.0, 1.0], (reps, d.size)) * d).mean(axis=1)
    p = float((np.abs(flipped) >= abs(d.mean())).mean())
    return float(lo), float(hi), p
