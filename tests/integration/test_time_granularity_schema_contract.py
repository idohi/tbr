"""Phase 0 schema and dtype contracts for legacy time terminology."""

import numpy as np
import pandas as pd
import pytest

import tbr
from tbr.core.posterior import calculate_incremental_posterior_probabilities

pytestmark = pytest.mark.integration


@pytest.fixture
def legacy_frames():
    """Return deterministic inputs shared by legacy schema contracts."""
    tbr_df = pd.DataFrame(
        {
            "period": [0, 1, 1, 1],
            "y": [0.0, 11.0, 22.0, 33.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "pred": [0.0, 10.0, 20.0, 30.0],
            "cumdif": [0.0, 1.0, 3.0, 6.0],
            "cumsd": [1.0, 1.5, 2.0, 2.5],
        },
        index=[90, 70, 50, 30],
    )
    tbr_summary = pd.DataFrame(
        {
            "sigma": [1.0],
            "t_dist_df": [10],
            "var_alpha": [0.1],
            "var_beta": [0.01],
            "alpha_beta_cov": [0.0],
        }
    )
    return tbr_df, tbr_summary


def test_subinterval_summary_schema_values_dtypes_and_inputs(legacy_frames):
    """Freeze legacy summary columns, labels, values, dtypes, and non-mutation."""
    tbr_df, tbr_summary = legacy_frames
    original_df = tbr_df.copy(deep=True)
    original_summary = tbr_summary.copy(deep=True)
    intervals = [(1, 1), (2, 3)]

    result = tbr.create_subinterval_summary(
        tbr_df,
        tbr_summary,
        intervals,
        ci_level=0.80,
        threshold=0.0,
    )

    assert list(result.columns) == [
        "interval",
        "start_day",
        "end_day",
        "days",
        "estimate",
        "se",
        "precision",
        "lower",
        "upper",
        "interval_excludes_threshold",
        "avg_daily_effect",
        "ci_level",
    ]
    assert result[["interval", "start_day", "end_day", "days"]].to_dict("records") == [
        {
            "interval": "Test-period observation 1",
            "start_day": 1,
            "end_day": 1,
            "days": 1,
        },
        {
            "interval": "Test-period observations 2 through 3 (inclusive)",
            "start_day": 2,
            "end_day": 3,
            "days": 2,
        },
    ]
    assert result["estimate"].tolist() == pytest.approx([1.0, 5.0])
    assert result["se"].tolist() == pytest.approx(
        [1.0535653752852738, 1.6278820596099706]
    )
    assert result["precision"].tolist() == pytest.approx(
        [1.4456851728066724, 2.233753131853722]
    )
    assert result["lower"].tolist() == pytest.approx(
        [-0.4456851728066724, 2.766246868146278]
    )
    assert result["upper"].tolist() == pytest.approx(
        [2.445685172806672, 7.233753131853722]
    )
    assert result["avg_daily_effect"].tolist() == pytest.approx([1.0, 2.5])
    assert result["interval_excludes_threshold"].tolist() == [False, True]
    assert result["ci_level"].tolist() == pytest.approx([0.80, 0.80])
    assert result.dtypes.astype(str).to_dict() == {
        # pandas 3 uses its default ``str`` dtype here; earlier supported
        # pandas versions infer ``object`` for the same string-only input.
        "interval": str(pd.Series(["Test-period observation 1"]).dtype),
        "start_day": "int64",
        "end_day": "int64",
        "days": "int64",
        "estimate": "float64",
        "se": "float64",
        "precision": "float64",
        "lower": "float64",
        "upper": "float64",
        "interval_excludes_threshold": "bool",
        "avg_daily_effect": "float64",
        "ci_level": "float64",
    }
    assert intervals == [(1, 1), (2, 3)]
    pd.testing.assert_frame_equal(tbr_df, original_df)
    pd.testing.assert_frame_equal(tbr_summary, original_summary)


def test_incremental_summary_schema_values_dtypes_and_inputs(legacy_frames):
    """Freeze incremental columns, values, dtypes, order, and non-mutation."""
    tbr_df, _ = legacy_frames
    original_df = tbr_df.copy(deep=True)

    result = tbr.create_incremental_tbr_summaries(
        tbr_df,
        alpha=1.0,
        beta=2.0,
        sigma=1.0,
        var_alpha=0.1,
        var_beta=0.01,
        cov_alpha_beta=0.0,
        degrees_freedom=10,
        level=0.80,
        threshold=0.0,
    )

    assert list(result.columns) == [
        "test_day",
        "estimate",
        "precision",
        "lower",
        "upper",
        "se",
        "level",
        "thres",
        "prob",
        "alpha",
        "beta",
        "alpha_beta_cov",
        "var_alpha",
        "var_beta",
        "sigma",
        "t_dist_df",
    ]
    assert result["test_day"].tolist() == [1, 2, 3]
    assert result["estimate"].tolist() == pytest.approx([1.0, 3.0, 6.0])
    assert result["precision"].tolist() == pytest.approx(
        [2.0582754616654295, 2.7443672822205727, 3.430459102775716]
    )
    assert result["lower"].tolist() == pytest.approx(
        [-1.0582754616654295, 0.2556327177794273, 2.569540897224284]
    )
    assert result["upper"].tolist() == pytest.approx(
        [3.0582754616654295, 5.744367282220573, 9.430459102775716]
    )
    assert result["se"].tolist() == pytest.approx([1.5, 2.0, 2.5])
    assert result["level"].tolist() == pytest.approx([0.80, 0.80, 0.80])
    assert result["thres"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert result["prob"].tolist() == pytest.approx(
        [0.7399652916672854, 0.9177463367772799, 0.9813421769690215]
    )
    for column, value in {
        "alpha": 1.0,
        "beta": 2.0,
        "alpha_beta_cov": 0.0,
        "var_alpha": 0.1,
        "var_beta": 0.01,
        "sigma": 1.0,
        "t_dist_df": 10.0,
    }.items():
        assert result[column].tolist() == pytest.approx([value, value, value])
    assert result.dtypes.astype(str).to_dict() == {
        "test_day": "int64",
        "estimate": "float64",
        "precision": "float64",
        "lower": "float64",
        "upper": "float64",
        "se": "float64",
        "level": "float64",
        "thres": "float64",
        "prob": "float64",
        "alpha": "float64",
        "beta": "float64",
        "alpha_beta_cov": "float64",
        "var_alpha": "float64",
        "var_beta": "float64",
        "sigma": "float64",
        "t_dist_df": "float64",
    }
    pd.testing.assert_frame_equal(tbr_df, original_df)


def test_incremental_posterior_mapping_order_dtypes_and_copies():
    """Freeze the legacy mapping key and array ownership contracts."""
    estimates = np.array([1, 2, 4], dtype=np.int16)
    standard_errors = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    original_estimates = estimates.copy()
    original_standard_errors = standard_errors.copy()

    result = calculate_incremental_posterior_probabilities(
        estimates,
        standard_errors,
        degrees_freedom=10,
        threshold=0.0,
    )

    assert list(result) == [
        "day",
        "estimates",
        "standard_errors",
        "probabilities",
        "probability_change",
    ]
    assert result["day"].tolist() == [1, 2, 3]
    np.testing.assert_array_equal(result["estimates"], [1, 2, 4])
    np.testing.assert_array_equal(result["standard_errors"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        result["probabilities"],
        [0.82955343384897, 0.82955343384897, 0.8940009344424447],
    )
    np.testing.assert_allclose(
        result["probability_change"],
        [0.0, 0.0, 0.0644475005934747],
    )
    assert {key: str(value.dtype) for key, value in result.items()} == {
        "day": str(np.dtype(np.int_)),
        "estimates": "int16",
        "standard_errors": "float32",
        "probabilities": "float64",
        "probability_change": "float64",
    }
    assert not np.shares_memory(result["estimates"], estimates)
    assert not np.shares_memory(result["standard_errors"], standard_errors)
    np.testing.assert_array_equal(estimates, original_estimates)
    np.testing.assert_array_equal(standard_errors, original_standard_errors)


def test_n_test_days_matches_n_test_for_hourly_integer_data():
    """Freeze the legacy count property as equal to the neutral count."""
    hours = np.arange(10, dtype=np.int64)
    control = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0])
    data = pd.DataFrame(
        {
            "hour": hours,
            "control": control,
            "test": 4.0 + 1.5 * control + np.linspace(-0.2, 0.2, 10),
        }
    )

    result = tbr.perform_tbr_analysis(
        data=data,
        time_col="hour",
        control_col="control",
        test_col="test",
        pretest_start=0,
        test_start=7,
        test_end=9,
        test_end_inclusive=True,
        level=0.80,
        threshold=0.0,
    )

    assert result.n_test == 3
    assert result.n_test_days == result.n_test
    assert isinstance(result.n_test_days, int)
