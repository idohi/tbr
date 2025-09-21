"""TBR Effects and Lift Calculation Module.

This module provides clean interfaces for calculating treatment effects and lift
in Time-Based Regression (TBR) analysis. It wraps the functional implementations
with professional, modular interfaces while maintaining full backward compatibility.

The effects module focuses on:
- Cumulative treatment effect calculations
- Lift measurement and uncertainty quantification
- Subinterval effect analysis
- Summary statistics generation

All functions maintain mathematical accuracy and statistical rigor while providing
clean, documented interfaces for production use.
"""

from typing import Dict

import numpy as np
import pandas as pd

# Export list for clean imports
__all__ = [
    "calculate_cumulative_standard_deviation",
    "calculate_cumulative_variance",
    "compute_interval_estimate_and_ci",
    "create_tbr_summary",
    "create_incremental_tbr_summaries",
]


def calculate_cumulative_standard_deviation(
    test_x_values: np.ndarray,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
) -> np.ndarray:
    """
    Calculate standard deviation of cumulative causal effect for TBR test period.

    This function computes the uncertainty in cumulative treatment effects as they
    accumulate over time during the test period, implementing the TBR formula for
    cumulative effect variance.

    Parameters
    ----------
    test_x_values : np.ndarray
        Control values during test period
    sigma : float
        Residual standard deviation from the model prediction over the learning set (σ)
    var_alpha : float
        Variance of intercept estimate (α)
    var_beta : float
        Variance of slope estimate (β)
    cov_alpha_beta : float
        Covariance between intercept and slope estimates

    Returns
    -------
    np.ndarray
        Cumulative standard deviations for each time point in test period

    Examples
    --------
    >>> import numpy as np
    >>> from tbr.core.effects import calculate_cumulative_standard_deviation
    >>> x_vals = np.array([1000, 1020, 1010, 1030])
    >>> cumsd = calculate_cumulative_standard_deviation(
    ...     x_vals, sigma=25, var_alpha=100, var_beta=0.001,
    ...     cov_alpha_beta=-0.05
    ... )
    >>> print(f"Cumulative std devs: {cumsd}")
    """
    from tbr.functional.tbr_functions import (
        calculate_cumulative_standard_deviation as _calculate_cumulative_standard_deviation,
    )

    return _calculate_cumulative_standard_deviation(
        test_x_values=test_x_values,
        sigma=sigma,
        var_alpha=var_alpha,
        var_beta=var_beta,
        cov_alpha_beta=cov_alpha_beta,
    )


def calculate_cumulative_variance(
    test_x_values: np.ndarray,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
) -> np.ndarray:
    """
    Calculate variance of cumulative causal effect for TBR test period.

    This function implements the TBR formula for cumulative effect variance directly,
    providing the mathematical foundation for statistical inference and credible intervals.
    The variance quantifies uncertainty in cumulative treatment effects as they
    accumulate over time during the test period.

    Mathematical Formula
    --------------------
    V[Δr(T)] = T · σ² + T² · v
    where:
    - T = time point (1, 2, 3, ..., n)
    - σ² = residual variance from regression model
    - v = Var(α̂) + 2·x̄_T·Cov(α̂,β̂) + x̄_T²·Var(β̂)
    - x̄_T = cumulative mean of control values up to time T

    This formula captures both:
    1. Residual uncertainty (T · σ²) - grows linearly with time
    2. Model parameter uncertainty (T² · v) - grows quadratically with time

    Parameters
    ----------
    test_x_values : np.ndarray
        Control values during test period
    sigma : float
        Residual standard deviation from the model prediction over the learning set (σ)
    var_alpha : float
        Variance of intercept estimate (α)
    var_beta : float
        Variance of slope estimate (β)
    cov_alpha_beta : float
        Covariance between intercept and slope estimates

    Returns
    -------
    np.ndarray
        Cumulative variances for each time point in test period

    Notes
    -----
    This function provides the variance directly, which is more efficient than
    computing standard deviation and squaring when variance is the desired output.
    For standard deviation, use calculate_cumulative_standard_deviation().

    The relationship between this function and calculate_cumulative_standard_deviation()
    is: variance = standard_deviation²

    Examples
    --------
    >>> import numpy as np
    >>> from tbr.core.effects import calculate_cumulative_variance
    >>> x_vals = np.array([1000, 1020, 1010, 1030])
    >>> cum_var = calculate_cumulative_variance(
    ...     x_vals, sigma=25, var_alpha=100, var_beta=0.001,
    ...     cov_alpha_beta=-0.05
    ... )
    >>> print(f"Cumulative variances: {cum_var}")

    References
    ----------
    .. [1] Time-Based Regression methodology for causal inference
    .. [2] Statistical inference for cumulative treatment effects
    """
    # Input validation following scientific Python standards
    if len(test_x_values) == 0:
        raise ValueError("test_x_values cannot be empty")

    n = len(test_x_values)
    T_values = np.arange(1, n + 1)  # [1, 2, 3, ..., n]

    # Calculate cumulative means efficiently using vectorized operations
    cumsum_x = np.cumsum(test_x_values)
    x_mean_cumulative = cumsum_x / T_values

    # Vectorized calculation of v for all time points
    # v = Var(α̂) + 2·x̄_T·Cov(α̂,β̂) + x̄_T²·Var(β̂)
    v_values = (
        var_alpha
        + 2 * x_mean_cumulative * cov_alpha_beta
        + (x_mean_cumulative**2) * var_beta
    )

    # Direct calculation of cumulative variance using TBR formula
    # V[Δr(T)] = T · σ² + T² · v
    cum_variance = T_values * (sigma**2) + (T_values**2) * v_values

    return cum_variance


def compute_interval_estimate_and_ci(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_day: int,
    end_day: int,
    ci_level: float,
) -> Dict[str, float]:
    """
    Compute cumulative treatment effect estimate and credible interval for a subinterval.

    This function calculates the cumulative treatment effect over a specified
    subinterval within the test period, along with its credible interval using
    t-distribution. This enables analysis of treatment effects for specific time
    ranges rather than the entire test period.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        TBR daily output with columns 'y', 'pred', 'period', 'estsd'
    tbr_summary : pd.DataFrame
        TBR summary containing 'sigma' and 't_dist_df' (degrees of freedom) parameters
    start_day : int
        Start day of subinterval (1-indexed within test period)
    end_day : int
        End day of subinterval (inclusive)
    ci_level : float
        Credible interval level (e.g., 0.80 for 80% interval)

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'estimate': Cumulative treatment effect for the subinterval
        - 'precision': Half-width of credible interval
        - 'lower': Lower bound of credible interval
        - 'upper': Upper bound of credible interval

    Examples
    --------
    >>> from tbr.core.effects import compute_interval_estimate_and_ci
    >>> result = compute_interval_estimate_and_ci(
    ...     tbr_results, daily_summaries, start_day=5, end_day=10, ci_level=0.80
    ... )
    >>> print(f"Effect estimate: {result['estimate']:.2f}")
    >>> print(f"80% CI: [{result['lower']:.2f}, {result['upper']:.2f}]")
    """
    from tbr.functional.tbr_functions import (
        compute_interval_estimate_and_ci as _compute_interval_estimate_and_ci,
    )

    return _compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=start_day,
        end_day=end_day,
        ci_level=ci_level,
    )


def create_tbr_summary(
    tbr_dataframe: pd.DataFrame,
    alpha: float,
    beta: float,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
    degrees_freedom: int,
    level: float,
    threshold: float,
) -> pd.DataFrame:
    """
    Create TBR summary statistics DataFrame with credible intervals and probabilities.

    This function generates a single-row summary DataFrame containing all key
    statistics for the TBR analysis, including the cumulative effect estimate,
    credible intervals, and model parameters.

    Parameters
    ----------
    tbr_dataframe : pd.DataFrame
        Complete TBR dataframe with all periods and statistics
    alpha : float
        Regression intercept coefficient (α)
    beta : float
        Regression slope coefficient (β)
    sigma : float
        Residual standard deviation from the model prediction over the learning set (σ)
    var_alpha : float
        Variance of intercept estimate (α)
    var_beta : float
        Variance of slope estimate (β)
    cov_alpha_beta : float
        Covariance between intercept and slope estimates
    degrees_freedom : int
        Residual degrees of freedom from regression
    level : float
        Credibility level for confidence intervals
    threshold : float
        Threshold for probability calculation

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with TBR summary statistics including:
        - 'estimate': Cumulative treatment effect
        - 'precision': Half-width of credible interval
        - 'lower', 'upper': Credible interval bounds
        - 'prob': Posterior probability of exceeding threshold
        - Model parameters and metadata

    Examples
    --------
    >>> from tbr.core.effects import create_tbr_summary
    >>> summary = create_tbr_summary(
    ...     tbr_results, alpha=50, beta=0.95, sigma=25,
    ...     var_alpha=100, var_beta=0.001, cov_alpha_beta=-0.05,
    ...     degrees_freedom=43, level=0.80, threshold=0.0
    ... )
    >>> print(f"Effect estimate: {summary['estimate'].iloc[0]:.2f}")
    """
    from tbr.functional.tbr_functions import create_tbr_summary as _create_tbr_summary

    return _create_tbr_summary(
        tbr_dataframe=tbr_dataframe,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        var_alpha=var_alpha,
        var_beta=var_beta,
        cov_alpha_beta=cov_alpha_beta,
        degrees_freedom=degrees_freedom,
        level=level,
        threshold=threshold,
    )


def create_incremental_tbr_summaries(
    tbr_dataframe: pd.DataFrame,
    alpha: float,
    beta: float,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
    degrees_freedom: int,
    level: float,
    threshold: float,
) -> pd.DataFrame:
    """
    Create incremental TBR summary statistics for each test period day.

    This function generates summary statistics for incremental test periods:
    - Day 1: Summary for first day only
    - Day 2: Summary for first two days (cumulative)
    - Day 3: Summary for first three days (cumulative)
    - ...and so on

    This enables day-by-day analysis of cumulative treatment effects during the
    test period, providing insights into when effects become detectable and stable.

    Parameters
    ----------
    tbr_dataframe : pd.DataFrame
        Complete TBR dataframe with all periods and statistics
    alpha : float
        Regression intercept coefficient (α)
    beta : float
        Regression slope coefficient (β)
    sigma : float
        Residual standard deviation from the model prediction over the learning set (σ)
    var_alpha : float
        Variance of intercept estimate (α)
    var_beta : float
        Variance of slope estimate (β)
    cov_alpha_beta : float
        Covariance between intercept and slope estimates
    degrees_freedom : int
        Residual degrees of freedom from regression
    level : float
        Credibility level for confidence intervals
    threshold : float
        Threshold for probability calculation

    Returns
    -------
    pd.DataFrame
        Multi-row DataFrame with incremental TBR summary statistics.
        Each row represents cumulative statistics up to that test day.
        Includes an additional 'test_day' column indicating the incremental period.

    Examples
    --------
    >>> from tbr.core.effects import create_incremental_tbr_summaries
    >>> incremental_summaries = create_incremental_tbr_summaries(
    ...     tbr_results, alpha=50, beta=0.95, sigma=25,
    ...     var_alpha=100, var_beta=0.001, cov_alpha_beta=-0.05,
    ...     degrees_freedom=43, level=0.80, threshold=0.0
    ... )
    >>> print(f"Day 1 effect: {incremental_summaries.iloc[0]['estimate']:.2f}")
    """
    from tbr.functional.tbr_functions import (
        create_incremental_tbr_summaries as _create_incremental_tbr_summaries,
    )

    return _create_incremental_tbr_summaries(
        tbr_dataframe=tbr_dataframe,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        var_alpha=var_alpha,
        var_beta=var_beta,
        cov_alpha_beta=cov_alpha_beta,
        degrees_freedom=degrees_freedom,
        level=level,
        threshold=threshold,
    )
