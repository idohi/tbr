"""
TBR Regression Analysis Module.

This module provides a clean interface for TBR regression operations, wrapping
the core regression functions from the functional implementation with improved
organization and modularity.

The module focuses on:
- Linear regression model fitting for TBR analysis
- Variance calculations for model and prediction uncertainty
- Statistical parameter extraction and validation

All functions maintain full compatibility with the existing functional
implementation while providing better code organization.

Examples
--------
>>> from tbr.core.regression import fit_regression_model, calculate_variances
>>> import pandas as pd
>>> import numpy as np
>>>
>>> # Prepare learning data
>>> learning_data = pd.DataFrame({
...     'control': np.random.normal(1000, 50, 30),
...     'test': np.random.normal(1020, 55, 30)
... })
>>>
>>> # Fit regression model
>>> model_params = fit_regression_model(learning_data, 'control', 'test')
>>> print(f"Beta coefficient: {model_params['beta']:.3f}")
>>>
>>> # Calculate variances
>>> x_values = np.array([1000, 1010, 1020])
>>> model_vars, pred_vars = calculate_variances(
...     x_values, model_params['x_mean'], model_params['sigma'],
...     model_params['n_pretest'], 100.0  # sum_x_squared_deviations
... )
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..functional.tbr_functions import (
    calculate_sum_x_squared_deviations,
    extract_sum_x_squared_deviations,
    fit_tbr_regression_model,
    safe_int_conversion,
)


def fit_regression_model(
    learning_data: pd.DataFrame,
    control_col: str,
    test_col: str,
) -> Dict[str, float]:
    """
    Fit TBR regression model on pretest data.

    This is a clean interface to the core TBR regression fitting functionality,
    providing a modular wrapper around the functional implementation.

    Parameters
    ----------
    learning_data : pd.DataFrame
        Learning set data used for training the regression model
    control_col : str
        Name of the control group metric column
    test_col : str
        Name of the test group metric column

    Returns
    -------
    Dict[str, float]
        Dictionary containing regression parameters:
        - 'alpha': Intercept (α)
        - 'beta': Slope coefficient (β)
        - 'sigma': Residual standard deviation (σ)
        - 'var_alpha': Variance of intercept estimate
        - 'var_beta': Variance of slope estimate
        - 'cov_alpha_beta': Covariance between α and β estimates
        - 'degrees_freedom': Residual degrees of freedom
        - 'n_pretest': Number of pretest observations
        - 'x_mean': Mean of control values (x̄)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> learning_data = pd.DataFrame({
    ...     'control': np.random.normal(1000, 50, 30),
    ...     'test': np.random.normal(1020, 55, 30)
    ... })
    >>> params = fit_regression_model(learning_data, 'control', 'test')
    >>> print(f"Regression coefficients: α={params['alpha']:.2f}, β={params['beta']:.3f}")
    """
    return fit_tbr_regression_model(learning_data, control_col, test_col)


def calculate_model_variance(
    x_values: np.ndarray,
    pretest_x_mean: float,
    sigma: float,
    n_pretest: int,
    pretest_sum_x_squared_deviations: float,
) -> np.ndarray:
    """
    Calculate model variance for fitted values using TBR formula.

    Implements the TBR model variance formula for MODEL UNCERTAINTY ONLY:
    V[ŷ*] = σ² · (1/n + (x* - x̄)²/Σ(xi - x̄)²)

    This is a clean interface to the proven functional implementation, providing
    the core model variance calculation for TBR regression analysis.

    Parameters
    ----------
    x_values : np.ndarray
        Control values (predictor variable x) from the test period (prediction targets)
    pretest_x_mean : float
        Mean of control values from pretest period (x̄)
    sigma : float
        Residual standard deviation from the model prediction over the pretest period (σ)
    n_pretest : int
        Number of observations in pretest period
    pretest_sum_x_squared_deviations : float
        Sum of squared deviations from pretest period: Σ(xi - x̄)²

    Returns
    -------
    np.ndarray
        Model variances for each x value (model uncertainty only)

    Examples
    --------
    >>> import numpy as np
    >>> x_vals = np.array([1000, 1010, 1020])
    >>> model_vars = calculate_model_variance(
    ...     x_vals, pretest_x_mean=1005, sigma=25, n_pretest=30,
    ...     pretest_sum_x_squared_deviations=15000
    ... )
    >>> print(f"Model uncertainties: {model_vars}")
    """
    # Lazy import - only load when function is called
    from tbr.functional.tbr_functions import (
        calculate_model_variance as _calculate_model_variance,
    )

    return _calculate_model_variance(
        x_values=x_values,
        pretest_x_mean=pretest_x_mean,
        sigma=sigma,
        n_pretest=n_pretest,
        pretest_sum_x_squared_deviations=pretest_sum_x_squared_deviations,
    )


def calculate_prediction_variance(
    model_variances: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Calculate prediction variance by adding residual noise to model uncertainty.

    Implements the TBR prediction variance formula:
    V[y*] = σ² + V[ŷ*]

    This is a clean interface to the proven functional implementation, providing
    the core prediction variance calculation for TBR regression analysis.

    Parameters
    ----------
    model_variances : np.ndarray
        Model variances from calculate_model_variance() (model uncertainty only)
    sigma : float
        Residual standard deviation from regression model (σ)

    Returns
    -------
    np.ndarray
        Prediction variances (model uncertainty + residual noise)

    Examples
    --------
    >>> import numpy as np
    >>> # First calculate model variances
    >>> x_vals = np.array([1000, 1010, 1020])
    >>> model_vars = calculate_model_variance(
    ...     x_vals, pretest_x_mean=1005, sigma=25, n_pretest=30,
    ...     pretest_sum_x_squared_deviations=15000
    ... )
    >>> # Then calculate prediction variances
    >>> pred_vars = calculate_prediction_variance(model_vars, sigma=25)
    >>> print(f"Prediction uncertainties: {pred_vars}")
    """
    # Lazy import - only load when function is called
    from tbr.functional.tbr_functions import (
        calculate_prediction_variance as _calculate_prediction_variance,
    )

    return _calculate_prediction_variance(
        model_variances=model_variances,
        sigma=sigma,
    )


def calculate_variances(
    x_values: np.ndarray,
    pretest_x_mean: float,
    sigma: float,
    n_pretest: int,
    pretest_sum_x_squared_deviations: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate model and prediction variances for given x values.

    This function provides a convenient interface to calculate both model
    variances (uncertainty in fitted values) and prediction variances
    (total uncertainty including residual noise) in a single call.

    Parameters
    ----------
    x_values : np.ndarray
        Control values (predictor variable x) from the test period (prediction targets)
    pretest_x_mean : float
        Mean of control values from pretest period (x̄)
    sigma : float
        Residual standard deviation from the model prediction over the pretest period (σ)
    n_pretest : int
        Number of observations in pretest period
    pretest_sum_x_squared_deviations : float
        Sum of squared deviations from pretest period: Σ(xi - x̄)²

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (model_variances, prediction_variances) where:
        - model_variances: Uncertainty in fitted values only
        - prediction_variances: Total uncertainty including residual noise

    Examples
    --------
    >>> import numpy as np
    >>> x_vals = np.array([1000, 1010, 1020])
    >>> model_vars, pred_vars = calculate_variances(
    ...     x_vals, pretest_x_mean=1005, sigma=25, n_pretest=30,
    ...     pretest_sum_x_squared_deviations=15000
    ... )
    >>> print(f"Model uncertainties: {model_vars}")
    >>> print(f"Prediction uncertainties: {pred_vars}")
    """
    # Calculate model variances (fitted value uncertainty only)
    model_variances = calculate_model_variance(
        x_values=x_values,
        pretest_x_mean=pretest_x_mean,
        sigma=sigma,
        n_pretest=n_pretest,
        pretest_sum_x_squared_deviations=pretest_sum_x_squared_deviations,
    )

    # Calculate prediction variances (total uncertainty)
    prediction_variances = calculate_prediction_variance(
        model_variances=model_variances,
        sigma=sigma,
    )

    return model_variances, prediction_variances


def calculate_sum_squared_deviations(x: np.ndarray) -> float:
    """
    Calculate sum of squared deviations from the mean.

    This function provides a direct interface to the core mathematical
    calculation used throughout TBR variance computations.

    Parameters
    ----------
    x : np.ndarray
        Input array of values

    Returns
    -------
    float
        Sum of squared deviations from the mean: Σ(xi - x̄)²

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> ssd = calculate_sum_squared_deviations(x)
    >>> print(f"Sum squared deviations: {ssd}")
    """
    return calculate_sum_x_squared_deviations(x)


def extract_sum_squared_deviations_from_model(var_beta: float, sigma: float) -> float:
    """
    Extract sum of squared deviations from regression model parameters.

    This function provides access to the mathematical relationship for
    extracting sum of squared deviations when original data is not available.

    Parameters
    ----------
    var_beta : float
        Variance of the slope coefficient (β) from regression model
    sigma : float
        Residual standard deviation from regression model

    Returns
    -------
    float
        Sum of squared deviations: Σ(xi - x̄)²

    Examples
    --------
    >>> # Extract from model parameters when original data unavailable
    >>> ssd = extract_sum_squared_deviations_from_model(var_beta=0.001, sigma=25.0)
    >>> print(f"Extracted sum squared deviations: {ssd}")
    """
    return extract_sum_x_squared_deviations(var_beta, sigma)


def convert_to_integer(value: float, param_name: str) -> int:
    """
    Safely convert float to int with validation for statistical parameters.

    This function provides a clean interface to the safe integer conversion
    used for statistical parameters like degrees of freedom.

    Parameters
    ----------
    value : float
        Value to convert (should be very close to an integer)
    param_name : str
        Parameter name for error messages

    Returns
    -------
    int
        Rounded integer value

    Raises
    ------
    ValueError
        If value is not close to an integer (tolerance > 0.01)

    Examples
    --------
    >>> degrees_freedom = convert_to_integer(43.0, "degrees_freedom")
    >>> print(f"Degrees of freedom: {degrees_freedom}")
    """
    return safe_int_conversion(value, param_name)


# Export list for clean imports
__all__ = [
    "fit_regression_model",
    "calculate_variances",
    "calculate_sum_squared_deviations",
    "extract_sum_squared_deviations_from_model",
    "convert_to_integer",
]
