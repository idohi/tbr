"""Phase 0 behavioral contracts for time-granularity-neutral inputs."""

import warnings

import numpy as np
import pandas as pd
import pytest

import tbr
import tbr.core as core
import tbr.core.effects as core_effects
import tbr.functional.tbr_functions as functional_functions

pytestmark = pytest.mark.integration


@pytest.fixture
def summary_frame():
    """Return deterministic model metadata for subinterval calculations."""
    return pd.DataFrame(
        {
            "sigma": [1.5],
            "t_dist_df": [20],
            "var_alpha": [0.25],
            "var_beta": [0.01],
            "alpha_beta_cov": [-0.02],
        }
    )


def _behavior_frame(time_values):
    """Build interleaved rows whose test effects identify the selected slice."""
    predictions = np.array([900.0, 10.0, 800.0, 20.0, 30.0, 700.0, 40.0])
    effects = np.array([0.0, 1.0, 0.0, 10.0, 100.0, 0.0, 1000.0])
    return pd.DataFrame(
        {
            "time": time_values,
            "period": [0, 1, 0, 1, 1, 0, 1],
            "y": predictions + effects,
            "x": [90.0, 1.0, 80.0, 2.0, 3.0, 70.0, 4.0],
            "pred": predictions,
        },
        index=[91, 17, 72, 4, 88, 31, 6],
    )


@pytest.mark.parametrize(
    "time_values",
    [
        pd.date_range("2026-01-01", periods=7, freq="D"),
        pd.date_range("2026-01-01", periods=7, freq="h"),
        pd.date_range("2026-01-01", periods=7, freq="W"),
        np.array([70, 10, 60, 20, 30, 50, 40], dtype=np.int64),
        np.array([7.5, 1.5, 6.5, 2.5, 3.5, 5.5, 4.5], dtype=np.float64),
        pd.to_datetime(
            [
                "2026-01-07",
                "2026-01-01",
                "2026-01-06",
                "2026-01-03",
                "2026-01-03",
                "2026-01-05",
                "2026-01-04",
            ]
        ),
    ],
    ids=[
        "daily",
        "hourly",
        "weekly",
        "integer-unsorted",
        "floating-unsorted",
        "irregular-duplicate-unsorted",
    ],
)
def test_subinterval_selection_is_positional_and_granularity_invariant(
    time_values, summary_frame
):
    """Freeze row order, selected identities, results, and input non-mutation."""
    tbr_df = _behavior_frame(time_values)
    original_df = tbr_df.copy(deep=True)
    original_summary = summary_frame.copy(deep=True)

    routes = [
        tbr.compute_interval_estimate_and_ci,
        core.compute_interval_estimate_and_ci,
        core_effects.compute_interval_estimate_and_ci,
        functional_functions.compute_interval_estimate_and_ci,
    ]
    results = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for route in routes:
            results.append(route(tbr_df, summary_frame, 2, 3, 0.80))

    # Test positions 2 and 3 retain the existing filtered-row order, selecting
    # the uniquely identifying effects 10 and 100 rather than sorting by time
    # or DataFrame index.
    assert results[0]["estimate"] == pytest.approx(110.0)
    assert results[0] == pytest.approx(results[1])
    assert results[0] == pytest.approx(results[2])
    assert results[0] == pytest.approx(results[3])
    assert list(results[0]) == ["estimate", "se", "precision", "lower", "upper"]
    assert caught == []

    pd.testing.assert_frame_equal(tbr_df, original_df)
    pd.testing.assert_frame_equal(summary_frame, original_summary)


def test_multiple_subintervals_preserve_requested_order_and_inputs(summary_frame):
    """Freeze input interval order and non-mutation for batch calculations."""
    tbr_df = _behavior_frame(np.arange(7, dtype=np.int64))
    intervals = [(4, 4), (1, 2), (2, 3)]
    original_df = tbr_df.copy(deep=True)
    original_summary = summary_frame.copy(deep=True)
    original_intervals = list(intervals)

    results = tbr.analyze_multiple_subintervals(
        tbr_df, summary_frame, intervals, ci_level=0.80
    )

    assert [result["estimate"] for result in results] == pytest.approx(
        [1000.0, 11.0, 110.0]
    )
    assert intervals == original_intervals
    pd.testing.assert_frame_equal(tbr_df, original_df)
    pd.testing.assert_frame_equal(summary_frame, original_summary)
