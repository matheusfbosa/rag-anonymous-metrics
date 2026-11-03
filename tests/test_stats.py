import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_rel

from rag_anonymous_metrics.analysis.stats import (
    bonferroni,
    contrast_series,
    delta_series,
    mean_test,
    resampling_test,
)


def _paired(question_ids, offline, ondemand, col="score"):
    return pd.DataFrame(
        {
            "question_id": question_ids,
            f"{col}_offline": offline,
            f"{col}_ondemand": ondemand,
        }
    )


def test_mean_test_matches_scipy_paired_t_test() -> None:
    offline = [0.2, 0.5, 0.4, 0.9, 0.3, 0.6]
    ondemand = [0.3, 0.7, 0.4, 1.0, 0.6, 0.5]
    paired = _paired(["1", "2", "3", "4", "5", "6"], offline, ondemand)

    mean, lo, hi, n, p = mean_test(delta_series(paired, "score"))

    expected = ttest_rel(ondemand, offline)
    assert n == 6
    assert mean == pytest.approx(np.mean(np.subtract(ondemand, offline)))
    assert p == pytest.approx(expected.pvalue)
    assert lo < mean < hi


def test_contrast_series_aligns_questions_by_id() -> None:
    paired_a = _paired(["1", "2", "3"], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3])
    paired_b = _paired(["3", "1", "2"], [0.0, 0.0, 0.0], [0.3, 0.0, 0.1])

    contrast = contrast_series(paired_a, paired_b, "score")

    assert contrast.sort_index().to_dict() == pytest.approx(
        {"1": 0.1, "2": 0.1, "3": 0.0}
    )


def test_bonferroni_multiplies_by_tests_and_caps_at_one() -> None:
    assert bonferroni(0.01, 3) == pytest.approx(0.03)
    assert bonferroni(0.4, 3) == 1.0


def test_resampling_test_separates_null_from_strong_effect() -> None:
    null = pd.Series([-0.5, 0.5] * 15)
    strong = pd.Series(np.linspace(0.1, 0.4, 30))

    assert resampling_test(null)[2] == 1.0
    assert resampling_test(strong)[2] < 0.01
