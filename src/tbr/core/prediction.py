"""
Core prediction module for Time-Based Regression (TBR) analysis.

This module provides clean, modular interfaces for TBR prediction functionality,
wrapping the proven implementations from the functional module. It focuses on
counterfactual predictions, uncertainty quantification, and interval estimation
for causal inference in time series experiments.

The module maintains full backward compatibility with the functional implementation
while providing improved organization and cleaner interfaces for the core TBR
prediction pipeline.

Functions
---------
generate_counterfactual_predictions : Generate counterfactual predictions with uncertainties
calculate_cumulative_standard_deviation : Calculate cumulative effect uncertainty
compute_interval_estimate_and_ci : Compute interval estimates and confidence intervals

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from tbr.core.prediction import generate_counterfactual_predictions
>>>
>>> # Test period data
>>> test_data = pd.DataFrame({
...     'date': pd.date_range('2023-02-15', periods=14),
...     'control': np.random.normal(1000, 50, 14)
... })
>>>
>>> # Generate counterfactual predictions
>>> predictions = generate_counterfactual_predictions(
...     alpha=50, beta=0.95, sigma=25, x_mean=1000, n_pretest=45,
...     test_period_data=test_data, control_col='control', time_col='date'
... )
>>> print(f"Predictions shape: {predictions.shape}")

Notes
-----
This module wraps functions from tbr.functional.tbr_functions while maintaining
the same interfaces and mathematical implementations. The functional module
remains the authoritative implementation.
"""

from typing import Dict

import numpy as np
import pandas as pd

# Import the proven functional implementations
from tbr.functional.tbr_functions import (
    calculate_cumulative_standard_deviation as _calculate_cumulative_standard_deviation,
)
from tbr.functional.tbr_functions import (
    compute_interval_estimate_and_ci as _compute_interval_estimate_and_ci,
)
from tbr.functional.tbr_functions import (
    generate_counterfactual_predictions as _generate_counterfactual_predictions,
)

# Export list for clean imports
__all__ = [
    "generate_counterfactual_predictions",
    "calculate_cumulative_standard_deviation",
    "compute_interval_estimate_and_ci",
]


def generate_counterfactual_predictions(
    alpha: float,
    beta: float,
    sigma: float,
    x_mean: float,
    n_pretest: int,
    test_period_data: pd.DataFrame,
    control_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Generate counterfactual predictions and prediction uncertainties for TBR test period.

    Creates counterfactual predictions using the fitted regression model and calculates
    their prediction standard deviations. These predictions represent what the test
    group values would have been without treatment intervention.

    This is a clean interface to the proven functional implementation, providing
    the core prediction functionality for TBR causal inference.

    Parameters
    ----------
    alpha : float
        Regression intercept coefficient (α)
    beta : float
        Regression slope coefficient (β)
    sigma : float
        Residual standard deviation from the model prediction over the learning set (σ)
    x_mean : float
        Mean of control values over the learning set (x̄)
    n_pretest : int
        Number of observations in learning set
    test_period_data : pd.DataFrame
        Test period data containing control values and time column
    control_col : str
        Name of control column
    time_col : str
        Name of the time column

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time column, control, pred, predsd where:
        - pred: counterfactual predictions (ŷ*)
        - predsd: prediction standard deviations including model uncertainty and residual noise

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> test_data = pd.DataFrame({
    ...     'date': pd.date_range('2023-02-15', periods=14),
    ...     'control': np.random.normal(1000, 50, 14)
    ... })
    >>> predictions = generate_counterfactual_predictions(
    ...     alpha=50, beta=0.95, sigma=25, x_mean=1000, n_pretest=45,
    ...     test_period_data=test_data, control_col='control', time_col='date'
    ... )
    >>> print(f"Predictions shape: {predictions.shape}")

    Notes
    -----
    Implements: ŷ* = α + β * x* with prediction variance V[y*] = σ² + V[ŷ*]
    where V[ŷ*] = σ² * (1/n + (x* - x̄)²/Σ(xi - x̄)²)
    """
    return _generate_counterfactual_predictions(
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        x_mean=x_mean,
        n_pretest=n_pretest,
        test_period_data=test_period_data,
        control_col=control_col,
        time_col=time_col,
    )


def calculate_cumulative_standard_deviation(
    test_x_values: np.ndarray,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
) -> np.ndarray:
    """
    Calculate standard deviation of cumulative causal effect for TBR test period.

    Implements the TBR formula for cumulative effect variance to quantify uncertainty
    in cumulative treatment effects as they accumulate over time during the test period.

    This provides the uncertainty quantification component of TBR analysis, essential
    for proper statistical inference about cumulative causal effects.

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
        Array of cumulative standard deviations for each day in test period

    Examples
    --------
    >>> import numpy as np
    >>> test_x = np.array([1000, 1020, 980, 1050, 990])
    >>> cumsd = calculate_cumulative_standard_deviation(
    ...     test_x_values=test_x, sigma=25.0, var_alpha=100.0,
    ...     var_beta=0.001, cov_alpha_beta=-0.05
    ... )
    >>> print(f"Cumulative std devs: {cumsd}")

    Notes
    -----
    Implements: V[Δr(T)] = T · σ² + T² · v
    where v = Var(α̂) + 2·x̄_T·Cov(α̂,β̂) + x̄_T²·Var(β̂)
    """
    return _calculate_cumulative_standard_deviation(
        test_x_values=test_x_values,
        sigma=sigma,
        var_alpha=var_alpha,
        var_beta=var_beta,
        cov_alpha_beta=cov_alpha_beta,
    )


def compute_interval_estimate_and_ci(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_day: int,
    end_day: int,
    ci_level: float,
) -> Dict[str, float]:
    """
    Compute cumulative treatment effect estimate and credible interval for a subinterval.

    Calculates the cumulative treatment effect over a specified subinterval within
    the test period, along with its credible interval using t-distribution. This
    enables analysis of treatment effects for specific time ranges rather than
    the entire test period.

    This provides the interval estimation component of TBR analysis, allowing
    flexible analysis of treatment effects over custom time windows.

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
    >>> result = compute_interval_estimate_and_ci(
    ...     tbr_results, daily_summaries, start_day=5, end_day=10, ci_level=0.80
    ... )
    >>> print(f"Effect estimate: {result['estimate']:.2f}")
    >>> print(f"80% CI: [{result['lower']:.2f}, {result['upper']:.2f}]")

    Notes
    -----
    Uses t-distribution for credible intervals with degrees of freedom from the
    regression model. Posterior variance combines model uncertainty and residual noise.
    """
    return _compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=start_day,
        end_day=end_day,
        ci_level=ci_level,
    )
