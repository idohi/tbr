"""
Shared pytest fixtures and utilities for TBR package testing.

This module provides common test fixtures, utilities, and configuration
for all test modules in the TBR package test suite.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest


@pytest.fixture  # type: ignore[misc]
def sample_time_series_data() -> pd.DataFrame:
    """
    Generate sample time series data for testing TBR analysis.

    Returns
    -------
    pd.DataFrame
        Sample data with date, control, and test columns for TBR analysis
    """
    np.random.seed(42)  # Reproducible results

    # Create 100 days of data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Generate control group data (baseline)
    control_baseline = 1000
    control_noise = np.random.normal(0, 50, 100)
    control_trend = np.linspace(0, 100, 100)  # Slight upward trend
    control_data = control_baseline + control_trend + control_noise

    # Generate test group data (with treatment effect after day 60)
    test_data = control_data.copy()
    treatment_effect = 50  # 5% lift
    test_data[60:] += treatment_effect  # Treatment starts at day 60
    test_noise = np.random.normal(0, 55, 100)
    test_data += test_noise

    return pd.DataFrame({"date": dates, "control": control_data, "test": test_data})


@pytest.fixture  # type: ignore[misc]
def sample_analysis_parameters() -> Dict[str, Any]:
    """
    Generate standard parameters for TBR analysis testing.

    Returns
    -------
    Dict[str, Any]
        Dictionary with standard analysis parameters
    """
    return {
        "time_col": "date",
        "control_col": "control",
        "test_col": "test",
        "pretest_start": pd.Timestamp("2023-01-01"),
        "test_start": pd.Timestamp("2023-03-02"),  # Day 60
        "test_end": pd.Timestamp("2023-04-10"),  # Day 100
        "level": 0.80,
        "threshold": 0.0,
    }


@pytest.fixture  # type: ignore[misc]
def minimal_valid_data() -> pd.DataFrame:
    """
    Minimal valid dataset for edge case testing.

    Returns
    -------
    pd.DataFrame
        Smallest valid dataset for TBR analysis
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "control": [100, 105, 95, 110, 98, 102, 107, 99, 104, 101],
            "test": [102, 108, 97, 115, 101, 105, 112, 103, 109, 106],
        }
    )


@pytest.fixture  # type: ignore[misc]
def invalid_data_scenarios() -> Dict[str, pd.DataFrame]:
    """
    Various invalid data scenarios for error testing.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of invalid datasets for testing error handling
    """
    return {
        "empty_data": pd.DataFrame(),
        "missing_columns": pd.DataFrame({"date": [1, 2, 3]}),
        "null_values": pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "control": [100, None, 95, 110, 98],
                "test": [102, 108, None, 115, 101],
            }
        ),
        "non_numeric_data": pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "control": ["a", "b", "c", "d", "e"],
                "test": [102, 108, 97, 115, 101],
            }
        ),
        "single_row": pd.DataFrame(
            {"date": [pd.Timestamp("2023-01-01")], "control": [100], "test": [102]}
        ),
    }


@pytest.fixture  # type: ignore[misc]
def expected_output_structure() -> Dict[str, list]:
    """
    Provide expected structure for TBR analysis output validation.

    Returns
    -------
    Dict[str, list]
        Expected column names for TBR output DataFrames
    """
    return {
        "tbr_dataframe_columns": [
            "date",
            "period",
            "control",
            "test",
            "pred",
            "predsd",
            "dif",
            "cumdif",
            "cumsd",
            "estsd",
        ],
        "summary_columns": [
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
        ],
    }


class TestDataGenerator:
    """Utility class for generating various test data scenarios."""

    @staticmethod
    def create_data_with_trend(
        n_days: int = 100,
        trend_slope: float = 1.0,
        noise_std: float = 10.0,
        treatment_effect: float = 0.0,
        treatment_start_day: int = 60,
    ) -> pd.DataFrame:
        """Generate time series data with specified trend and treatment effect."""
        np.random.seed(42)

        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        trend = np.linspace(0, trend_slope * n_days, n_days)
        noise = np.random.normal(0, noise_std, n_days)

        control_data = 1000 + trend + noise
        test_data = control_data.copy()

        if treatment_effect != 0.0:
            test_data[treatment_start_day:] += treatment_effect

        return pd.DataFrame({"date": dates, "control": control_data, "test": test_data})

    @staticmethod
    def create_data_with_seasonality(
        n_days: int = 365,
        seasonal_amplitude: float = 50.0,
        treatment_effect: float = 0.0,
        treatment_start_day: int = 200,
    ) -> pd.DataFrame:
        """Generate time series data with seasonal patterns."""
        np.random.seed(42)

        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        seasonal_pattern = seasonal_amplitude * np.sin(
            2 * np.pi * np.arange(n_days) / 365.25
        )
        noise = np.random.normal(0, 20, n_days)

        control_data = 1000 + seasonal_pattern + noise
        test_data = control_data.copy()

        if treatment_effect != 0.0:
            test_data[treatment_start_day:] += treatment_effect

        return pd.DataFrame({"date": dates, "control": control_data, "test": test_data})


def assert_dataframe_structure(df: pd.DataFrame, expected_columns: list) -> None:
    """
    Assert that DataFrame has expected structure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    expected_columns : list
        Expected column names
    """
    assert isinstance(df, pd.DataFrame), "Output must be a DataFrame"
    assert not df.empty, "DataFrame cannot be empty"
    assert (
        list(df.columns) == expected_columns
    ), f"Expected columns {expected_columns}, got {list(df.columns)}"


def assert_statistical_validity(
    estimate: float, lower: float, upper: float, level: float
) -> None:
    """
    Assert statistical validity of confidence intervals.

    Parameters
    ----------
    estimate : float
        Point estimate
    lower : float
        Lower confidence bound
    upper : float
        Upper confidence bound
    level : float
        Confidence level
    """
    assert lower <= estimate <= upper, "Estimate must be within confidence interval"
    assert lower < upper, "Lower bound must be less than upper bound"
    assert 0 < level < 1, "Confidence level must be between 0 and 1"
    assert not np.isnan(estimate), "Estimate cannot be NaN"
    assert not np.isnan(lower), "Lower bound cannot be NaN"
    assert not np.isnan(upper), "Upper bound cannot be NaN"
