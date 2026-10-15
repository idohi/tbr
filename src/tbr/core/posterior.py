"""
Advanced Posterior Probability Calculations and Threshold Testing for TBR.

This module provides sophisticated posterior probability analysis and threshold testing
capabilities that extend beyond basic statistical inference. It focuses on advanced
Bayesian analysis, threshold sensitivity, and multi-scenario posterior comparisons
for comprehensive TBR decision-making.

The implementation builds upon the basic inference module to provide:
- Advanced posterior variance calculations
- Threshold sensitivity analysis
- Incremental posterior probability tracking
- Multi-scenario posterior comparisons

Key Features
------------
- Sophisticated threshold testing strategies
- Multi-threshold sensitivity analysis
- Time-series posterior probability evolution
- Posterior distribution characterization

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from tbr.core.posterior import perform_threshold_sensitivity_analysis
>>>
>>> # Threshold sensitivity analysis
>>> thresholds = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
>>> sensitivity = perform_threshold_sensitivity_analysis(
...     estimate=12.5, standard_error=4.2, degrees_freedom=45, thresholds=thresholds
... )
>>> print(f"Probabilities: {sensitivity['probabilities']}")

Mathematical Foundation
-----------------------
The advanced posterior probability functions implement sophisticated Bayesian
statistical methods:

1. **Posterior Variance**: V_post = T_s·σ² + T_s²·v
   Subinterval variance including the covariance of the predictions

2. **Threshold Sensitivity**: P(θ > τᵢ | data) for multiple thresholds τᵢ
   Comprehensive threshold testing across decision space

3. **Incremental Probabilities**: P(θₜ > τ | data₁:ₜ) for t = 1, 2, ..., T
   Time-series evolution of posterior beliefs

Notes
-----
All functions use lazy imports to minimize import overhead. Functions integrate
seamlessly with the basic inference module.

See individual function documentation for detailed usage and examples.
"""

import math
from typing import Dict, List, Union

import numpy as np


def calculate_posterior_variance(
    x_values: np.ndarray,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
) -> float:
    """
    Calculate posterior variance of the cumulative effect over a subinterval.

    Combines residual noise (σ²) with the variance of the summed counterfactual
    predictions. Because all predictions share the estimated coefficients, the
    model component uses their covariance rather than per-day model variances.

    Parameters
    ----------
    x_values : np.ndarray
        Control values for the test days in the subinterval
    sigma : float
        Residual standard deviation from the regression model
    var_alpha : float
        Variance of intercept estimate (α)
    var_beta : float
        Variance of slope estimate (β)
    cov_alpha_beta : float
        Covariance between intercept and slope estimates

    Returns
    -------
    float
        Posterior variance value

    Raises
    ------
    ValueError
        If x_values is empty, sigma is not positive, or the resulting
        variance is negative
    TypeError
        If inputs have incorrect types

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1000.0, 1020.0, 980.0, 1050.0, 990.0])
    >>> posterior_var = calculate_posterior_variance(
    ...     x, sigma=25.0, var_alpha=100.0, var_beta=0.001, cov_alpha_beta=-0.05
    ... )
    >>> print(f"Posterior variance: {posterior_var:.3f}")
    Posterior variance: 28506.600

    Mathematical Formula
    --------------------
    V[Δ(a,b)] = T_s × σ² + T_s² × v

    where v = Var(α̂) + 2·x̄·Cov(α̂,β̂) + x̄²·Var(β̂), T_s is the number of days
    in the subinterval, and x̄ is the mean of the control values over it.

    This decomposition separates:
    - Residual noise: T_s × σ²
    - Model uncertainty: T_s² × v, the variance of the summed predictions

    Notes
    -----
    This is the posterior variance used by compute_interval_estimate_and_ci.
    For a subinterval starting on the first test day it equals cumsd², and for
    a single day it equals predsd².
    """
    # Input validation
    if not isinstance(x_values, np.ndarray):
        raise TypeError(f"x_values must be numpy array, got {type(x_values)}")

    for name, value in (
        ("sigma", sigma),
        ("var_alpha", var_alpha),
        ("var_beta", var_beta),
        ("cov_alpha_beta", cov_alpha_beta),
    ):
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be numeric, got {type(value)}")

    if len(x_values) == 0:
        raise ValueError("x_values cannot be empty")

    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    # Calculate posterior variance components
    n_days = len(x_values)
    x_mean = np.mean(x_values)
    v = var_alpha + 2 * x_mean * cov_alpha_beta + x_mean**2 * var_beta
    residual_variance = n_days * (sigma**2)
    model_variance = n_days**2 * v
    posterior_variance = residual_variance + model_variance

    if posterior_variance < 0:
        raise ValueError(
            "Negative posterior variance detected. Check that var_alpha, "
            "var_beta, and cov_alpha_beta form a valid covariance matrix."
        )

    return float(posterior_variance)


def perform_threshold_sensitivity_analysis(
    estimate: float,
    standard_error: float,
    degrees_freedom: int,
    thresholds: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Perform sensitivity analysis across multiple threshold values.

    Calculates posterior probabilities for a range of threshold values to
    understand how the probability of exceeding different thresholds varies.
    This enables comprehensive threshold testing and sensitivity analysis
    for decision-making in TBR analysis.

    Parameters
    ----------
    estimate : float
        Point estimate of the treatment effect
    standard_error : float
        Standard error of the estimate
    degrees_freedom : int
        Degrees of freedom for the t-distribution
    thresholds : np.ndarray
        Array of threshold values to test

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'thresholds': Input threshold values
        - 'probabilities': Posterior probabilities for each threshold
        - 'log_odds': Log odds ratios for each threshold
        - 'sensitivity': Gradient of probability with respect to threshold

    Raises
    ------
    ValueError
        If estimate, standard_error, or any threshold is not finite;
        standard_error or degrees_freedom is not positive; or thresholds is
        empty
    TypeError
        If inputs have incorrect types

    Examples
    --------
    >>> import numpy as np
    >>> thresholds = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    >>> sensitivity = perform_threshold_sensitivity_analysis(
    ...     estimate=12.5, standard_error=4.2, degrees_freedom=45, thresholds=thresholds
    ... )
    >>> print(f"P(effect > 10): {sensitivity['probabilities'][2]:.3f}")
    P(effect > 10): 0.723

    Mathematical Formula
    --------------------
    For each threshold τᵢ:
    P(θ > τᵢ | data) = 1 - F_t((τᵢ - estimate) / standard_error; df)

    Sensitivity = dP/dτ = -f_t((τ - estimate) / SE; df) / SE
    where f_t is the t-distribution PDF

    Notes
    -----
    This function enables comprehensive threshold testing by evaluating
    posterior probabilities across a range of decision-relevant thresholds.
    The sensitivity analysis helps identify critical threshold regions.
    """
    # Lazy import to minimize overhead
    from scipy import stats

    # Input validation
    if not isinstance(estimate, (int, float, np.integer, np.floating)):
        raise TypeError(f"estimate must be numeric, got {type(estimate)}")

    if not isinstance(standard_error, (int, float, np.integer, np.floating)):
        raise TypeError(f"standard_error must be numeric, got {type(standard_error)}")

    if not isinstance(degrees_freedom, (int, np.integer)):
        raise TypeError(f"degrees_freedom must be integer, got {type(degrees_freedom)}")

    if not isinstance(thresholds, np.ndarray):
        raise TypeError(f"thresholds must be numpy array, got {type(thresholds)}")

    for name, value in (("estimate", estimate), ("standard_error", standard_error)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")

    if standard_error <= 0:
        raise ValueError(f"standard_error must be positive, got {standard_error}")

    if degrees_freedom <= 0:
        raise ValueError(f"degrees_freedom must be positive, got {degrees_freedom}")

    if len(thresholds) == 0:
        raise ValueError("thresholds array cannot be empty")

    if not np.all(np.isfinite(thresholds)):
        raise ValueError("thresholds must contain only finite values")

    # Calculate t-statistics for all thresholds
    t_statistics = (thresholds - estimate) / standard_error

    # Calculate posterior probabilities
    probabilities = stats.t.sf(t_statistics, df=degrees_freedom)

    # Calculate log odds ratios (avoiding division by zero)
    epsilon = 1e-15
    log_odds = np.log((probabilities + epsilon) / (1 - probabilities + epsilon))

    # Calculate sensitivity (gradient of probability w.r.t. threshold)
    # Sensitivity = -f_t(t_stat) / SE where f_t is PDF
    pdf_values = stats.t.pdf(t_statistics, df=degrees_freedom)
    sensitivity = -pdf_values / standard_error

    return {
        "thresholds": thresholds.copy(),
        "probabilities": probabilities,
        "log_odds": log_odds,
        "sensitivity": sensitivity,
    }


def calculate_incremental_posterior_probabilities(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    degrees_freedom: int,
    threshold: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Calculate incremental posterior probabilities for time-series analysis.

    Computes posterior probabilities for incremental time periods (day 1,
    days 1-2, days 1-3, etc.) to track how posterior beliefs evolve over
    the test period. This enables analysis of when effects become detectable
    and how confidence builds over time.

    Parameters
    ----------
    estimates : np.ndarray
        Array of cumulative effect estimates for each time period
    standard_errors : np.ndarray
        Array of cumulative standard errors for each time period
    degrees_freedom : int
        Degrees of freedom for the t-distribution
    threshold : float, default=0.0
        Threshold value for probability calculation

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'day': Day numbers (1, 2, 3, ...)
        - 'estimates': Cumulative estimates
        - 'standard_errors': Cumulative standard errors
        - 'probabilities': Posterior probabilities for each period
        - 'probability_change': Change in probability from previous day

    Raises
    ------
    ValueError
        If arrays have different lengths, are empty, or contain non-finite
        values; standard_errors contains a negative value; threshold is not
        finite; or degrees_freedom is not positive
    TypeError
        If inputs have incorrect types

    Examples
    --------
    >>> import numpy as np
    >>> estimates = np.array([2.1, 4.8, 7.2, 9.1, 11.5])
    >>> std_errors = np.array([3.2, 4.1, 4.8, 5.2, 5.6])
    >>> incremental = calculate_incremental_posterior_probabilities(
    ...     estimates, std_errors, degrees_freedom=45
    ... )
    >>> print(f"Day 5 probability: {incremental['probabilities'][4]:.3f}")
    Day 5 probability: 0.977

    Mathematical Formula
    --------------------
    For each day t with SE_t > 0:
    P(θₜ > threshold | data₁:ₜ) = 1 - F_t((threshold - estimate_t) / SE_t; df)

    When SE_t = 0, the probability is 1 if estimate_t > threshold and 0
    otherwise.

    Probability change: ΔP_t = P_t - P_{t-1}

    Notes
    -----
    This function enables tracking of how posterior beliefs evolve during
    the test period, providing insights into the temporal dynamics of
    treatment effect detection and confidence building.
    """
    # Lazy import to minimize overhead
    from scipy import stats

    # Input validation
    if not isinstance(estimates, np.ndarray):
        raise TypeError(f"estimates must be numpy array, got {type(estimates)}")

    if not isinstance(standard_errors, np.ndarray):
        raise TypeError(
            f"standard_errors must be numpy array, got {type(standard_errors)}"
        )

    if not isinstance(degrees_freedom, (int, np.integer)):
        raise TypeError(f"degrees_freedom must be integer, got {type(degrees_freedom)}")

    if not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError(f"threshold must be numeric, got {type(threshold)}")

    if len(estimates) == 0:
        raise ValueError("estimates array cannot be empty")

    if len(standard_errors) == 0:
        raise ValueError("standard_errors array cannot be empty")

    if len(estimates) != len(standard_errors):
        raise ValueError(
            f"estimates and standard_errors must have same length: "
            f"{len(estimates)} vs {len(standard_errors)}"
        )

    for name, values in (
        ("estimates", estimates),
        ("standard_errors", standard_errors),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")

    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")

    if np.any(standard_errors < 0):
        raise ValueError("standard_errors must be non-negative")

    if degrees_freedom <= 0:
        raise ValueError(f"degrees_freedom must be positive, got {degrees_freedom}")

    # Calculate posterior probabilities for uncertain and deterministic periods
    probabilities = np.empty_like(estimates, dtype=float)
    positive_se = standard_errors > 0
    t_statistics = (threshold - estimates[positive_se]) / standard_errors[positive_se]
    probabilities[positive_se] = stats.t.sf(t_statistics, df=degrees_freedom)
    probabilities[~positive_se] = (estimates[~positive_se] > threshold).astype(float)

    # Calculate probability changes (difference from previous day)
    probability_change = np.zeros_like(probabilities)
    probability_change[1:] = probabilities[1:] - probabilities[:-1]

    # Create day numbers
    days = np.arange(1, len(estimates) + 1)

    return {
        "day": days,
        "estimates": estimates.copy(),
        "standard_errors": standard_errors.copy(),
        "probabilities": probabilities,
        "probability_change": probability_change,
    }


def compare_posterior_probabilities(
    scenarios: List[Dict[str, float]], threshold: float = 0.0
) -> Dict[str, Union[List[str], List[float], List[int]]]:
    """
    Compare posterior probabilities across multiple scenarios.

    Calculates the posterior probability that the effect exceeds a common
    threshold for each scenario (e.g., different experiments, time periods,
    or model specifications) and ranks the scenarios by that probability.

    Parameters
    ----------
    scenarios : List[Dict[str, float]]
        List of scenario dictionaries, each containing:
        - 'estimate': Point estimate
        - 'standard_error': Standard error
        - 'degrees_freedom': Degrees of freedom
        - 'name': Scenario name (optional)
    threshold : float, default=0.0
        Threshold value for probability calculation

    Returns
    -------
    Dict[str, Union[List[str], List[float], List[int]]]
        Dictionary containing:
        - 'scenario_names': Names or indices of scenarios
        - 'probabilities': Posterior probabilities for each scenario
        - 'ranking': Ranking of scenarios by probability (1 = highest;
          tied scenarios share the same rank)

    Raises
    ------
    ValueError
        If scenarios list is empty, threshold is not finite, or scenarios have
        invalid parameters (missing keys, non-finite values, or a non-positive
        standard error or degrees of freedom)
    TypeError
        If inputs have incorrect types

    Examples
    --------
    >>> scenarios = [
    ...     {'estimate': 10.0, 'standard_error': 3.0, 'degrees_freedom': 30, 'name': 'Experiment A'},
    ...     {'estimate': 15.0, 'standard_error': 4.0, 'degrees_freedom': 30, 'name': 'Experiment B'},
    ...     {'estimate': 8.0, 'standard_error': 2.5, 'degrees_freedom': 30, 'name': 'Experiment C'}
    ... ]
    >>> comparison = compare_posterior_probabilities(scenarios)
    >>> best = comparison['scenario_names'][comparison['ranking'].index(1)]
    >>> print(f"Highest posterior probability: {best}")
    Highest posterior probability: Experiment B

    Mathematical Formula
    --------------------
    For each scenario i:
    P_i(θ > threshold | data_i) = 1 - F_t((threshold - estimate_i) / SE_i; df_i)
    """
    # Lazy import to minimize overhead
    from scipy import stats

    # Input validation
    if not isinstance(scenarios, list):
        raise TypeError(f"scenarios must be list, got {type(scenarios)}")

    if len(scenarios) == 0:
        raise ValueError("scenarios list cannot be empty")

    if not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError(f"threshold must be numeric, got {type(threshold)}")

    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")

    # Validate each scenario
    required_keys = ["estimate", "standard_error", "degrees_freedom"]
    for i, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise TypeError(f"scenario {i} must be dict, got {type(scenario)}")

        for key in required_keys:
            if key not in scenario:
                raise ValueError(f"scenario {i} missing required key: {key}")

        for key in required_keys:
            if not math.isfinite(scenario[key]):
                raise ValueError(
                    f"scenario {i} {key} must be finite, got {scenario[key]}"
                )

        if scenario["standard_error"] <= 0:
            raise ValueError(
                f"scenario {i} standard_error must be positive, got {scenario['standard_error']}"
            )

        if scenario["degrees_freedom"] <= 0:
            raise ValueError(
                f"scenario {i} degrees_freedom must be positive, got {scenario['degrees_freedom']}"
            )

    # Extract scenario information
    scenario_names = []
    probabilities_list = []

    for i, scenario in enumerate(scenarios):
        # Get scenario name
        name = str(scenario.get("name", f"Scenario_{i+1}"))
        scenario_names.append(name)

        # Calculate posterior probability
        estimate = scenario["estimate"]
        se = scenario["standard_error"]
        df = scenario["degrees_freedom"]

        t_stat = (threshold - estimate) / se
        prob = stats.t.sf(t_stat, df=df)
        probabilities_list.append(prob)

    probabilities = np.array(probabilities_list)

    # Calculate ranking (1 = highest probability; ties share the lowest rank)
    ranking = stats.rankdata(-probabilities, method="min").astype(int)

    return {
        "scenario_names": scenario_names,
        "probabilities": probabilities.tolist(),
        "ranking": ranking.tolist(),
    }
