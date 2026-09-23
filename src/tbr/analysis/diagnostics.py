"""
TBR Analysis Diagnostics Module.

This module provides diagnostic tools for reviewing Time-Based Regression
(TBR) model fit, regression assumptions, residuals, and heuristic prediction
metrics. Diagnostic statuses identify evidence that merits review; they do not
prove model validity or establish a causal interpretation.

Functions
---------
validate_tbr_model : Aggregate model-fit and assumption diagnostics
diagnose_tbr_analysis : End-to-end diagnostic workflow for TBR analysis
check_tbr_assumptions : Regression-assumption diagnostic statuses
analyze_tbr_residuals : TBR-specific residual analysis and outlier detection
assess_tbr_performance : Prediction and data-sufficiency heuristics
create_tbr_diagnostic_report : Structured diagnostic reporting

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from tbr import perform_tbr_analysis, validate_tbr_model
>>>
>>> # Build example data and perform TBR analysis
>>> rng = np.random.default_rng(0)
>>> control = rng.normal(1000, 50, size=44)
>>> data = pd.DataFrame(
...     {
...         "date": pd.date_range("2023-01-01", periods=44),
...         "control": control,
...         "test": 1.05 * control + rng.normal(0, 10, size=44),
...     }
... )
>>> results = perform_tbr_analysis(
...     data=data,
...     time_col='date',
...     control_col='control',
...     test_col='test',
...     pretest_start=pd.Timestamp('2023-01-01'),
...     test_start=pd.Timestamp('2023-01-31'),
...     test_end=pd.Timestamp('2023-02-14'),
...     level=0.80,
...     threshold=0.0,
... )
>>> tbr_df = results.tbr_dataframe()
>>> summary_df = results.summary()
>>>
>>> # Review aggregate model diagnostics
>>> validation = validate_tbr_model(tbr_df, summary_df)
>>> print(f"No-warning diagnostic status: {validation['overall_validity']}")
>>> if validation['warnings']:
...     for warning in validation['warnings']:
...         print(f"Warning: {warning}")

Comprehensive diagnostic workflow:

>>> from tbr import diagnose_tbr_analysis
>>> diagnostics = diagnose_tbr_analysis(tbr_df, summary_df)
>>> validation = diagnostics['model_validation']
>>> print(f"R-squared: {validation['goodness_of_fit']['r_squared']:.3f}")
>>> print(f"All assumption checks passed: {validation['assumption_tests']['all_assumptions_valid']}")

Performance assessment:

>>> from tbr import assess_tbr_performance
>>> performance = assess_tbr_performance(tbr_df, summary_df)
>>> print(f"Prediction accuracy: {performance['prediction_metrics']['mape']:.2f}%")
>>> print(f"Heuristic efficiency score: {performance['efficiency_score']:.2f}")
"""

from typing import Any, Dict, List, Mapping, Optional, Union, cast

import numpy as np
import pandas as pd

# Import core diagnostic functions
from tbr.core.diagnostics import (
    GoodnessOfFitMetrics,
    calculate_goodness_of_fit,
    calculate_residuals,
    calculate_standardized_residuals,
    calculate_studentized_residuals,
    create_diagnostic_summary,
    validate_model_assumptions,
)

# Import core regression for model parameter extraction
# Import validation utilities
from tbr.utils.validation import validate_required_columns

_GOODNESS_OF_FIT_KEYS = frozenset(GoodnessOfFitMetrics.__annotations__)
_EXPECTED_DIAGNOSTIC_ERRORS = (
    ValueError,
    FloatingPointError,
    RuntimeWarning,
    ZeroDivisionError,
    np.linalg.LinAlgError,
)


def _validate_goodness_of_fit_metrics(
    goodness_of_fit: Mapping[str, object],
) -> GoodnessOfFitMetrics:
    """Validate the internal goodness-of-fit schema before exposing results."""
    missing_keys = _GOODNESS_OF_FIT_KEYS - goodness_of_fit.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing goodness-of-fit metric(s): {missing}")
    return cast(GoodnessOfFitMetrics, goodness_of_fit)


def validate_tbr_model(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    learning_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Aggregate TBR model-fit, assumption, residual, and prediction diagnostics.

    Runs a collection of formal tests and heuristic checks to identify
    assumptions, fit, residual, or prediction behavior that merits review.
    Its aggregate status is not proof that a model or causal interpretation is
    valid.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        Daily TBR output containing ``period``, ``y``, ``x``, ``pred``,
        ``predsd``, ``dif``, ``cumdif``, ``cumsd``, and ``estsd``.
    tbr_summary : pd.DataFrame
        Non-empty summary containing ``alpha``, ``beta``, ``sigma``,
        ``var_alpha``, ``var_beta``, legacy covariance column
        ``alpha_beta_cov``, and legacy degrees-of-freedom column ``t_dist_df``.
        Values are read from the first row.
    learning_data : pd.DataFrame, optional
        Learning-period data containing ``x`` and ``y``. If omitted, rows with
        ``period == 0`` are copied from ``tbr_df``.
    alpha : float, default 0.05
        Decision level for Shapiro-Wilk, Breusch-Pagan, F-test, and warning
        heuristics. This is not the regression intercept stored as ``alpha`` in
        ``tbr_summary``.

    Returns
    -------
    Dict[str, Any]
        Diagnostic mapping. See Notes for the stable top-level and nested
        schemas.

    Raises
    ------
    ValueError
        If a required input column is missing, ``tbr_summary`` is empty, no
        inferred learning rows exist, or a lower-level diagnostic rejects the
        supplied learning data.
    KeyError
        If explicitly supplied ``learning_data`` lacks ``x`` or ``y``.

    Notes
    -----
    The returned mapping has these stable top-level keys:

    ``overall_validity`` : bool
        Whether the function emitted no warnings. This is a heuristic
        aggregate, not formal proof of model validity.
    ``warnings`` : List[str]
        Assumption, fit, outlier, coverage, or calculation warnings.
    ``assumption_tests`` : Dict[str, Union[bool, float, str]]
        On success, contains the following nested keys.

        ``linearity_valid`` : bool
            Whether the F-test p-value is below ``alpha`` and
            :math:`R^2>0.1`.
        ``normality_valid`` : bool
            Whether the Shapiro-Wilk p-value exceeds ``alpha``.
        ``homoscedasticity_valid`` : bool
            Whether the Breusch-Pagan p-value exceeds ``alpha``.
        ``independence_valid`` : bool
            Whether the Durbin-Watson statistic is in ``[1.5, 2.5]``.
        ``all_assumptions_valid`` : bool
            Conjunction of the four preceding diagnostic statuses.
        ``significance_level`` : float
            Supplied frequentist decision level ``alpha``.
        ``validation_summary`` : str
            Human-readable status summary.
        ``error`` : str
            Sole nested key when an expected calculation failure is caught;
            absent on success.
    ``goodness_of_fit`` : Dict[str, float]
        On success, contains the following nested keys.

        ``r_squared`` : float
            Coefficient of determination.
        ``adj_r_squared`` : float
            Adjusted coefficient of determination.
        ``f_statistic`` : float
            Overall simple-regression F statistic.
        ``f_p_value`` : float
            Frequentist p-value for ``f_statistic``.
        ``mse`` : float
            Mean squared learning-period residual.
        ``rmse`` : float
            Root mean squared learning-period residual.
        ``error`` : str
            Sole nested key when an expected calculation failure is caught;
            absent on success.
    ``residual_analysis`` : Dict[str, Any]
        On success, contains the following nested keys.

        ``residuals`` : np.ndarray
            Raw learning-period residuals.
        ``standardized_residuals`` : np.ndarray
            Residuals divided by fitted ``sigma``.
        ``studentized_residuals`` : np.ndarray
            Residuals adjusted for fitted ``sigma`` and leverage.
        ``outliers`` : List[int]
            Zero-based positions with absolute studentized residual above
            2.5.
        ``outlier_threshold`` : float
            Fixed absolute threshold ``2.5``.
        ``outlier_percentage`` : float
            Percentage of learning observations flagged.
        ``error`` : str
            Sole nested key when an expected calculation failure is caught;
            absent on success.
    ``prediction_quality`` : Dict[str, Any]
        With test rows, contains the following nested keys.

        ``mae`` : numpy.float64
            Mean absolute test-period value of ``y - pred``.
        ``mse`` : numpy.float64
            Mean squared test-period value of ``y - pred``.
        ``rmse`` : numpy.float64
            Root mean squared test-period value of ``y - pred``.
        ``mape`` : numpy.float64
            Mean absolute percentage error, in percent.
        ``prediction_interval_coverage`` : numpy.float64
            Observed proportion inside ``pred ± 1.96 * predsd``.
        ``n_predictions`` : int
            Number of test rows.
        ``error`` : str
            Sole nested key with no test rows or after an expected
            calculation failure; absent on success.

    Assumption booleans combine formal test p-values with heuristic rules:
    ``linearity_valid`` requires F-test ``p < alpha`` and :math:`R^2>0.1`;
    ``independence_valid`` uses the Durbin-Watson range 1.5 through 2.5.
    Additional warnings use :math:`R^2<0.5`, outliers above 10%, and
    prediction coverage below 0.90. Prediction coverage is the observed
    fraction inside ``pred ± 1.96 * predsd`` and is independent of the
    configured TBR credibility level. Test-period prediction errors can include
    treatment effects, so these metrics do not isolate counterfactual forecast
    accuracy or establish causal validity.

    For mathematical background, see the
    :doc:`mathematical methodology guide </mathematical_methodology>`,
    especially the sections on model diagnostics, model assumptions, and
    limitations.

    Examples
    --------
    Basic model validation. ``tbr_df`` is the daily TBR output,
    ``tbr_summary`` is the summary table, and ``learning_data`` contains
    learning-period columns ``x`` and ``y``:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import perform_tbr_analysis, validate_tbr_model
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> results = perform_tbr_analysis(
    ...     data=data, time_col="date", control_col="control", test_col="test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ...     level=0.90, threshold=0.0,
    ... )
    >>> tbr_df = results.tbr_dataframe()
    >>> tbr_summary = results.summary()
    >>> learning_data = tbr_df[tbr_df["period"] == 0]
    >>> validation = validate_tbr_model(tbr_df, tbr_summary, learning_data)
    >>> list(validation) == [
    ...     "overall_validity", "warnings", "assumption_tests",
    ...     "goodness_of_fit", "residual_analysis", "prediction_quality",
    ... ]
    True
    >>> if validation['overall_validity']:
    ...     print("No diagnostic warnings were emitted")
    ... else:
    ...     print("Diagnostic warnings:")
    ...     for warning in validation['warnings']:
    ...         print(f"  - {warning}")

    Detailed diagnostic review:

    >>> validation = validate_tbr_model(tbr_df, tbr_summary, learning_data)
    >>> print(f"R²: {validation['goodness_of_fit']['r_squared']:.3f}")
    >>> print(f"Normality check passed: {validation['assumption_tests']['normality_valid']}")
    >>> print(f"Outliers detected: {len(validation['residual_analysis']['outliers'])}")
    """
    # Validate input DataFrames
    required_tbr_cols = [
        "period",
        "y",
        "x",
        "pred",
        "predsd",
        "dif",
        "cumdif",
        "cumsd",
        "estsd",
    ]
    validate_required_columns(tbr_df, required_tbr_cols, "tbr_df")

    required_summary_cols = [
        "alpha",
        "beta",
        "sigma",
        "var_alpha",
        "var_beta",
        "alpha_beta_cov",
        "t_dist_df",
    ]
    validate_required_columns(tbr_summary, required_summary_cols, "tbr_summary")

    if tbr_summary.empty:
        raise ValueError("tbr_summary DataFrame cannot be empty")

    # Extract learning data if not provided
    if learning_data is None:
        learning_data = tbr_df[tbr_df["period"] == 0].copy()
        if learning_data.empty:
            raise ValueError(
                "No learning period data found (period == 0) and learning_data not provided"
            )

    # Extract model parameters from summary
    summary_row = tbr_summary.iloc[0]
    model_params = {
        "alpha": float(summary_row["alpha"]),
        "beta": float(summary_row["beta"]),
        "sigma": float(summary_row["sigma"]),
        "var_alpha": float(summary_row["var_alpha"]),
        "var_beta": float(summary_row["var_beta"]),
        "alpha_beta_cov": float(summary_row["alpha_beta_cov"]),
        "degrees_freedom": int(summary_row["t_dist_df"]),  # Add degrees of freedom
    }

    warnings_list = []

    # 1. Statistical Assumption Tests
    try:
        assumption_tests = validate_model_assumptions(
            learning_data, model_params, "x", "y", alpha=alpha
        )

        if not assumption_tests["normality_valid"]:
            warnings_list.append(
                "Residuals fail normality test - consider data transformation"
            )
        if not assumption_tests["homoscedasticity_valid"]:
            warnings_list.append(
                "Residuals show heteroscedasticity - variance may be non-constant"
            )
        if not assumption_tests["independence_valid"]:
            warnings_list.append(
                "Residuals show autocorrelation - independence assumption violated"
            )

    except _EXPECTED_DIAGNOSTIC_ERRORS as e:
        warnings_list.append(f"Assumption testing failed: {str(e)}")
        assumption_tests = {"error": str(e)}

    # 2. Goodness of Fit Analysis
    goodness_of_fit: Dict[str, Any]
    try:
        goodness_of_fit = dict(
            _validate_goodness_of_fit_metrics(
                calculate_goodness_of_fit(learning_data, model_params, "x", "y")
            )
        )

        if goodness_of_fit["r_squared"] < 0.5:
            warnings_list.append(
                f"Low R² ({goodness_of_fit['r_squared']:.3f}) - model explains little variance"
            )
        if goodness_of_fit["f_p_value"] > alpha:
            warnings_list.append(
                "F-statistic not significant - overall model may not be meaningful"
            )

    except _EXPECTED_DIAGNOSTIC_ERRORS as e:
        warnings_list.append(f"Goodness of fit calculation failed: {str(e)}")
        goodness_of_fit = {"error": str(e)}

    # 3. Residual Analysis
    try:
        residuals = calculate_residuals(learning_data, model_params, "x", "y")
        standardized_residuals = calculate_standardized_residuals(
            learning_data, model_params, "x", "y"
        )
        studentized_residuals = calculate_studentized_residuals(
            learning_data, model_params, "x", "y"
        )

        # Identify outliers (|studentized residual| > 2.5)
        outlier_threshold = 2.5
        outliers = np.where(np.abs(studentized_residuals) > outlier_threshold)[0]

        if len(outliers) > 0.1 * len(learning_data):  # More than 10% outliers
            warnings_list.append(
                f"High number of outliers detected ({len(outliers)}/{len(learning_data)})"
            )

        residual_analysis = {
            "residuals": residuals,
            "standardized_residuals": standardized_residuals,
            "studentized_residuals": studentized_residuals,
            "outliers": outliers.tolist(),
            "outlier_threshold": outlier_threshold,
            "outlier_percentage": len(outliers) / len(learning_data) * 100,
        }

    except _EXPECTED_DIAGNOSTIC_ERRORS as e:
        warnings_list.append(f"Residual analysis failed: {str(e)}")
        residual_analysis = {"error": str(e)}

    # 4. Prediction Quality Assessment
    try:
        test_data = tbr_df[tbr_df["period"] == 1].copy()
        if not test_data.empty:
            # Calculate prediction accuracy metrics
            prediction_errors = test_data["y"] - test_data["pred"]
            mae = np.mean(np.abs(prediction_errors))
            mse = np.mean(prediction_errors**2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs(prediction_errors / test_data["y"])) * 100

            # Check prediction interval coverage
            pred_lower = test_data["pred"] - 1.96 * test_data["predsd"]
            pred_upper = test_data["pred"] + 1.96 * test_data["predsd"]
            coverage = np.mean(
                (test_data["y"] >= pred_lower) & (test_data["y"] <= pred_upper)
            )

            if coverage < 0.90:  # Less than 90% coverage for ~95% intervals
                warnings_list.append(
                    f"Poor prediction interval coverage ({coverage:.1%}) - uncertainty may be underestimated"
                )

            prediction_quality = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "prediction_interval_coverage": coverage,
                "n_predictions": len(test_data),
            }
        else:
            prediction_quality = {"error": "No test period data available"}

    except _EXPECTED_DIAGNOSTIC_ERRORS as e:
        warnings_list.append(f"Prediction quality assessment failed: {str(e)}")
        prediction_quality = {"error": str(e)}

    # Overall validity assessment
    overall_validity = len(warnings_list) == 0

    return {
        "overall_validity": overall_validity,
        "warnings": warnings_list,
        "assumption_tests": assumption_tests,
        "goodness_of_fit": goodness_of_fit,
        "residual_analysis": residual_analysis,
        "prediction_quality": prediction_quality,
    }


def diagnose_tbr_analysis(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    learning_data: Optional[pd.DataFrame] = None,
    include_performance: bool = True,
) -> Dict[str, Union[Dict, List, float, Any]]:
    """
    Run the end-to-end TBR diagnostic workflow.

    Combines model-fit, assumption, residual, and optional prediction/data
    heuristics into one structured result. The output supports model review; it
    does not certify analysis quality or reliability.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        Daily TBR output. The complete required schema is documented by
        :func:`validate_tbr_model`.
    tbr_summary : pd.DataFrame
        Non-empty summary with the seven model columns required by
        :func:`validate_tbr_model`.
    learning_data : pd.DataFrame, optional
        Learning-period data containing ``x`` and ``y``. If omitted, rows with
        ``period == 0`` are copied from ``tbr_df``.
    include_performance : bool, default True
        Whether to include prediction-quality and data-sufficiency diagnostics.

    Returns
    -------
    Dict[str, Union[Dict, List]]
        Diagnostic workflow mapping. See Notes for the stable top-level and
        nested schemas.

    Raises
    ------
    ValueError
        Propagated from validation when required columns are missing,
        ``tbr_summary`` is empty, or learning data are unusable.
    KeyError
        If explicitly supplied ``learning_data`` lacks ``x`` or ``y``.

    Notes
    -----
    The returned mapping has these stable keys:

    ``model_validation`` : Dict[str, Any]
        Exact mapping returned by :func:`validate_tbr_model`, including its
        six top-level keys and success/error variants.
    ``diagnostic_summary`` : Dict[str, Any]
        On success, contains the following nested keys.

        ``goodness_of_fit`` : Dict[str, float]
            The six typed metrics documented by
            :func:`validate_tbr_model`.
        ``information_criteria`` : Dict[str, float]
            Contains the following nested keys.

            ``aic`` : float
                Akaike information criterion.
            ``bic`` : float
                Bayesian information criterion.
            ``log_likelihood`` : float
                Gaussian model log-likelihood.
        ``normality_test`` : Dict[str, Union[float, bool, str]]
            Contains the following nested keys.

            ``statistic`` : float
                Shapiro-Wilk statistic.
            ``p_value`` : float
                Frequentist Shapiro-Wilk p-value.
            ``is_normal`` : bool
                Whether ``p_value > 0.05``.
            ``test_name`` : str
                ``"Shapiro-Wilk"``.
        ``homoscedasticity_test`` : Dict[str, Union[float, bool, str]]
            Contains the following nested keys.

            ``statistic`` : float
                Breusch-Pagan statistic.
            ``p_value`` : float
                Frequentist Breusch-Pagan p-value.
            ``is_homoscedastic`` : bool
                Whether ``p_value > 0.05``.
            ``test_name`` : str
                ``"Breusch-Pagan"``.
        ``independence_test`` : Dict[str, Union[float, bool, str]]
            Contains the following nested keys.

            ``statistic`` : float
                Durbin-Watson statistic.
            ``interpretation`` : str
                Fixed-range interpretation.
            ``is_independent`` : bool
                Whether ``statistic`` is in ``[1.5, 2.5]``.
            ``test_name`` : str
                ``"Durbin-Watson"``.
        ``overall_validity`` : bool
            Whether this component emitted no assumption warnings.
        ``warnings`` : List[str]
            Assumption-warning messages.
        ``error`` : str
            Present only after an expected calculation failure; in that
            variant, ``overall_validity`` is ``False`` and ``warnings`` is
            a one-element ``List[str]``.
    ``performance_metrics`` : Dict[str, Any]
        Exact mapping returned by :func:`assess_tbr_performance`, or an
        empty mapping when ``include_performance=False``.
    ``recommendations`` : List[str]
        Heuristic follow-up suggestions derived from validation and
        performance thresholds; these are not formal inferential results.

    This helper combines lower-level validation, assumption checks, residual
    diagnostics, and optional performance metrics. A failed diagnostic does not
    necessarily make a TBR analysis unusable; it identifies model assumptions or
    data quality issues that should be reviewed before interpreting results.

    For mathematical background, see the
    :doc:`mathematical methodology guide </mathematical_methodology>`,
    especially the sections on model diagnostics, model assumptions, and
    limitations.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import diagnose_tbr_analysis, perform_tbr_analysis
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> results = perform_tbr_analysis(
    ...     data=data, time_col="date", control_col="control", test_col="test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ...     level=0.90, threshold=0.0,
    ... )
    >>> tbr_df = results.tbr_dataframe()
    >>> tbr_summary = results.summary()
    >>> learning_data = tbr_df[tbr_df["period"] == 0]
    >>> diagnostics = diagnose_tbr_analysis(tbr_df, tbr_summary, learning_data)
    >>> list(diagnostics) == [
    ...     "model_validation", "diagnostic_summary",
    ...     "performance_metrics", "recommendations",
    ... ]
    True
    >>> print(f"No-warning status: {diagnostics['model_validation']['overall_validity']}")
    >>> print("Recommendations:")
    >>> for rec in diagnostics['recommendations']:
    ...     print(f"  - {rec}")
    """
    # Extract learning data if needed
    if learning_data is None:
        learning_data = tbr_df[tbr_df["period"] == 0].copy()

    # 1. Model Validation
    model_validation = validate_tbr_model(tbr_df, tbr_summary, learning_data)

    # 2. Core Diagnostic Summary
    summary_row = tbr_summary.iloc[0]
    model_params = {
        "alpha": float(summary_row["alpha"]),
        "beta": float(summary_row["beta"]),
        "sigma": float(summary_row["sigma"]),
        "var_alpha": float(summary_row["var_alpha"]),
        "var_beta": float(summary_row["var_beta"]),
        "alpha_beta_cov": float(summary_row["alpha_beta_cov"]),
        "degrees_freedom": int(summary_row["t_dist_df"]),
    }

    diagnostic_summary: Dict[str, Any]
    try:
        diagnostic_summary = dict(
            create_diagnostic_summary(learning_data, model_params, "x", "y")
        )
    except _EXPECTED_DIAGNOSTIC_ERRORS as e:
        diagnostic_summary = {
            "error": str(e),
            "overall_validity": False,
            "warnings": [f"Diagnostic summary failed: {str(e)}"],
        }

    # 3. Performance Metrics (if requested)
    performance_metrics = {}
    if include_performance:
        performance_metrics = assess_tbr_performance(tbr_df, tbr_summary)

    # 4. Generate Recommendations
    recommendations = _generate_diagnostic_recommendations(
        model_validation, performance_metrics
    )

    return {
        "model_validation": model_validation,
        "diagnostic_summary": diagnostic_summary,
        "performance_metrics": performance_metrics,
        "recommendations": recommendations,
    }


def check_tbr_assumptions(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    learning_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> Dict[str, Union[bool, float, str]]:
    """
    Compute regression-assumption diagnostic statuses for a TBR model.

    Performs diagnostic checks of regression assumptions used by TBR,
    including linearity, normality, homoscedasticity, and independence. Provides
    boolean indicators and supporting test statistics for model review.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        Daily TBR output containing ``period`` when ``learning_data`` is
        omitted. Only rows with ``period == 0`` are then used.
    tbr_summary : pd.DataFrame
        Non-empty summary whose first row contains ``alpha``, ``beta``,
        ``sigma``, ``var_alpha``, ``var_beta``, ``alpha_beta_cov``, and
        ``t_dist_df``.
    learning_data : pd.DataFrame, optional
        Learning period data for assumption testing. Must contain columns
        ``x`` and ``y`` when provided directly.
    alpha : float, default 0.05
        Decision level for p-value rules. This is distinct from summary
        ``alpha``, the regression intercept.

    Returns
    -------
    Dict[str, Union[bool, float, str]]
        Assumption-status mapping. See Notes for the stable schema.

    Raises
    ------
    KeyError
        If a required summary or learning-data column is absent.
    IndexError
        If ``tbr_summary`` has no rows.
    ValueError
        If lower-level residual diagnostics reject empty or insufficient
        learning data.

    Notes
    -----
    The returned mapping has these stable keys:

    ``linearity_valid`` : bool
        Whether F-test ``p < alpha`` and :math:`R^2>0.1`.
    ``normality_valid`` : bool
        Whether the Shapiro-Wilk p-value is greater than ``alpha``.
    ``homoscedasticity_valid`` : bool
        Whether the Breusch-Pagan p-value is greater than ``alpha``.
    ``independence_valid`` : bool
        Whether the Durbin-Watson statistic is between 1.5 and 2.5,
        inclusive.
    ``all_assumptions_valid`` : bool
        Conjunction of the four preceding heuristic statuses.
    ``significance_level`` : float
        The supplied ``alpha`` value.
    ``validation_summary`` : str
        Human-readable list of failed statuses, or a success label.

    Shapiro-Wilk and Breusch-Pagan provide formal test statistics internally,
    but this wrapper returns only thresholded booleans. The linearity status is
    a fit heuristic, and Durbin-Watson uses a rough fixed range rather than a
    p-value. Failed statuses guide model review; they do not by themselves
    prove that a treatment effect is invalid.

    See the :doc:`model-diagnostics methodology </mathematical_methodology>` for
    the regression assumptions and diagnostic definitions.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import check_tbr_assumptions, perform_tbr_analysis
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> results = perform_tbr_analysis(
    ...     data=data, time_col="date", control_col="control", test_col="test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ...     level=0.90, threshold=0.0,
    ... )
    >>> tbr_df = results.tbr_dataframe()
    >>> tbr_summary = results.summary()
    >>> learning_data = tbr_df[tbr_df["period"] == 0]
    >>> assumptions = check_tbr_assumptions(tbr_df, tbr_summary, learning_data)
    >>> list(assumptions) == [
    ...     "linearity_valid", "normality_valid", "homoscedasticity_valid",
    ...     "independence_valid", "all_assumptions_valid",
    ...     "significance_level", "validation_summary",
    ... ]
    True
    >>> print(f"All assumption checks passed: {assumptions['all_assumptions_valid']}")
    >>> print(f"Normality check passed: {assumptions['normality_valid']}")
    """
    if learning_data is None:
        learning_data = tbr_df[tbr_df["period"] == 0].copy()

    summary_row = tbr_summary.iloc[0]
    model_params = {
        "alpha": float(summary_row["alpha"]),
        "beta": float(summary_row["beta"]),
        "sigma": float(summary_row["sigma"]),
        "var_alpha": float(summary_row["var_alpha"]),
        "var_beta": float(summary_row["var_beta"]),
        "alpha_beta_cov": float(summary_row["alpha_beta_cov"]),
        "degrees_freedom": int(summary_row["t_dist_df"]),
    }

    return validate_model_assumptions(
        learning_data, model_params, "x", "y", alpha=alpha
    )


def analyze_tbr_residuals(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    learning_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[np.ndarray, List, float]]:
    r"""
    TBR-specific residual analysis and outlier detection.

    Performs residual analysis for TBR models including calculation of raw,
    standardized, and studentized residuals, outlier detection, and residual
    summary statistics.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        Daily TBR output containing ``period`` when ``learning_data`` is
        omitted.
    tbr_summary : pd.DataFrame
        Non-empty summary whose first row contains ``alpha``, ``beta``,
        ``sigma``, ``var_alpha``, ``var_beta``, ``alpha_beta_cov``, and
        ``t_dist_df``.
    learning_data : pd.DataFrame, optional
        Learning period data for residual calculation. Must contain columns
        ``x`` and ``y`` when provided directly.

    Returns
    -------
    Dict[str, Union[np.ndarray, List, float]]
        Residual-analysis mapping. See Notes for the stable top-level and
        nested schemas.

    Raises
    ------
    KeyError
        If a required summary or learning-data column is absent.
    IndexError
        If ``tbr_summary`` has no rows.
    ValueError
        If residual calculation receives empty or insufficient learning data.
    numpy.linalg.LinAlgError
        If the leverage matrix cannot be inverted.

    Notes
    -----
    The returned mapping has these stable keys:

    ``residuals`` : np.ndarray
        Raw residuals :math:`e_i=y_i-(\beta_0+\beta_1x_i)`.
    ``standardized_residuals`` : np.ndarray
        Values :math:`e_i/\sigma`.
    ``studentized_residuals`` : np.ndarray
        Values :math:`e_i/(\sigma\sqrt{1-h_{ii}})`, with the leverage
        denominator floored internally.
    ``outliers`` : List[int]
        Zero-based positions where absolute studentized residual exceeds
        2.5.
    ``outlier_threshold`` : float
        Fixed threshold ``2.5``.
    ``outlier_percentage`` : float
        Flagged observations as a percentage.
    ``residual_stats`` : Dict[str, float]
        Contains the following nested keys.

        ``mean`` : numpy.float64
            Arithmetic mean.
        ``std`` : numpy.float64
            Population standard deviation (``ddof=0``).
        ``min`` : numpy.float64
            Minimum.
        ``max`` : numpy.float64
            Maximum.
        ``q25`` : numpy.float64
            25th percentile.
        ``median`` : numpy.float64
            Median.
        ``q75`` : numpy.float64
            75th percentile.
    ``residual_std`` : float
        Alias of ``residual_stats["std"]``; it is not necessarily the
        fitted summary ``sigma``.
    ``n_observations`` : int
        Number of learning observations.

    Outliers are flagged using studentized residuals with a default absolute
    threshold of 2.5. Treat them as review candidates rather than automatic
    exclusions.

    See the :doc:`model-diagnostics methodology </mathematical_methodology>` for
    the definitions of :math:`e_i`, :math:`\sigma`, and leverage
    :math:`h_{ii}`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import analyze_tbr_residuals, perform_tbr_analysis
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> results = perform_tbr_analysis(
    ...     data=data, time_col="date", control_col="control", test_col="test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ...     level=0.90, threshold=0.0,
    ... )
    >>> tbr_df = results.tbr_dataframe()
    >>> tbr_summary = results.summary()
    >>> learning_data = tbr_df[tbr_df["period"] == 0]
    >>> residuals = analyze_tbr_residuals(tbr_df, tbr_summary, learning_data)
    >>> list(residuals["residual_stats"]) == [
    ...     "mean", "std", "min", "max", "q25", "median", "q75",
    ... ]
    True
    >>> print(f"Outliers detected: {len(residuals['outliers'])}")
    >>> print(f"Residual std: {residuals['residual_std']:.3f}")
    """
    if learning_data is None:
        learning_data = tbr_df[tbr_df["period"] == 0].copy()

    summary_row = tbr_summary.iloc[0]
    model_params = {
        "alpha": float(summary_row["alpha"]),
        "beta": float(summary_row["beta"]),
        "sigma": float(summary_row["sigma"]),
        "var_alpha": float(summary_row["var_alpha"]),
        "var_beta": float(summary_row["var_beta"]),
        "alpha_beta_cov": float(summary_row["alpha_beta_cov"]),
        "degrees_freedom": int(summary_row["t_dist_df"]),
    }

    # Calculate different types of residuals
    residuals = calculate_residuals(learning_data, model_params, "x", "y")
    standardized_residuals = calculate_standardized_residuals(
        learning_data, model_params, "x", "y"
    )
    studentized_residuals = calculate_studentized_residuals(
        learning_data, model_params, "x", "y"
    )

    # Outlier detection
    outlier_threshold = 2.5
    outliers = np.where(np.abs(studentized_residuals) > outlier_threshold)[0]

    # Residual statistics
    residual_stats = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals),
        "q25": np.percentile(residuals, 25),
        "median": np.median(residuals),
        "q75": np.percentile(residuals, 75),
    }

    return {
        "residuals": residuals,
        "standardized_residuals": standardized_residuals,
        "studentized_residuals": studentized_residuals,
        "outliers": outliers.tolist(),
        "outlier_threshold": outlier_threshold,
        "outlier_percentage": len(outliers) / len(learning_data) * 100,
        "residual_stats": residual_stats,
        "residual_std": residual_stats["std"],
        "n_observations": len(residuals),
    }


def assess_tbr_performance(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
) -> Dict[str, Union[float, int, Dict]]:
    """
    Assess prediction quality and data sufficiency for TBR analysis.

    Evaluates prediction accuracy, data-size ratios, and heuristic summary
    metrics for TBR analysis workflows. It does not measure runtime, memory,
    CPU usage, or other computational profiling metrics.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        Daily TBR output containing ``period``. Test rows additionally require
        ``y``, ``pred``, and ``predsd``; learning rows require ``y`` for the
        fit proxy.
    tbr_summary : pd.DataFrame
        Non-empty summary whose first row contains ``t_dist_df`` and ``sigma``.

    Returns
    -------
    Dict[str, Union[float, int, Dict]]
        Performance-assessment mapping. See Notes for the stable top-level and
        nested schemas.

    Raises
    ------
    KeyError
        If a column needed for the available periods or summary is absent.
    IndexError
        If ``tbr_summary`` has no rows.

    Notes
    -----
    The returned mapping has these stable top-level and nested keys:

    ``data_metrics`` : Dict[str, Union[int, float]]
        Contains the following nested keys.

        ``total_observations`` : int
            Number of rows in ``tbr_df``.
        ``learning_observations`` : int
            Number of rows with ``period == 0``.
        ``test_observations`` : int
            Number of rows with ``period == 1``.
        ``learning_test_ratio`` : float
            Learning count divided by ``max(test count, 1)``.
    ``prediction_metrics`` : Dict[str, float]
        With test rows, contains the following nested keys; it is empty
        with no test rows.

        ``mae`` : float
            Mean absolute value of ``y - pred``.
        ``mse`` : float
            Mean squared value of ``y - pred``.
        ``rmse`` : float
            Root mean squared value of ``y - pred``.
        ``mape`` : float
            Mean absolute percentage error, in percent.
        ``mean_error`` : float
            Mean of ``y - pred``.
        ``std_error`` : float
            Population standard deviation of ``y - pred`` (``ddof=0``).
        ``interval_coverage`` : float
            Proportion inside ``pred ± 1.96 * predsd``.
    ``model_complexity`` : Dict[str, Union[int, float]]
        Contains the following nested keys.

        ``degrees_freedom`` : int
            First-row ``t_dist_df`` converted to ``int``.
        ``sigma`` : float
            First-row residual standard deviation.
        ``r_squared_proxy`` : numpy.float64 or float
            ``1 - sigma**2 / var(learning y)`` using population variance,
            or ``0.0`` without learning rows.
    ``efficiency_score`` : float
        Mean of available clipped components: ``max(0, 1 - mape/100)``,
        closeness of coverage to 0.95, and
        ``min(1, learning_observations/50)``. It is ``0.0`` when no
        component is available.
    ``performance_summary`` : Dict[str, str]
        Contains the following package-specific heuristic labels.

        ``data_quality`` : str
            ``"Good"`` for at least 20 learning rows; otherwise
            ``"Limited"``.
        ``prediction_quality`` : str
            ``"Good"`` for MAPE below 10, ``"Moderate"`` below 25, and
            ``"Poor"`` otherwise.
        ``overall_performance`` : str
            ``"Excellent"`` above 0.8, ``"Good"`` above 0.6,
            ``"Moderate"`` above 0.4, and ``"Poor"`` otherwise.

    The score, fit proxy, and categorical labels are package-specific
    heuristics, not validated performance measures or inferential tests.
    Test-period errors include any treatment effect, MAPE is undefined or
    unstable when observed ``y`` is zero or near zero, and the fixed 1.96 band
    is not tied to the configured TBR credibility level.

    See the :doc:`model-diagnostics methodology </mathematical_methodology>` for
    the inferential assumptions that these heuristics do not replace.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import assess_tbr_performance, perform_tbr_analysis
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> results = perform_tbr_analysis(
    ...     data=data, time_col="date", control_col="control", test_col="test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ...     level=0.90, threshold=0.0,
    ... )
    >>> tbr_df = results.tbr_dataframe()
    >>> tbr_summary = results.summary()
    >>> performance = assess_tbr_performance(tbr_df, tbr_summary)
    >>> list(performance) == [
    ...     "data_metrics", "prediction_metrics", "model_complexity",
    ...     "efficiency_score", "performance_summary",
    ... ]
    True
    >>> print(f"Prediction MAPE: {performance['prediction_metrics']['mape']:.2f}%")
    >>> print(f"Heuristic efficiency score: {performance['efficiency_score']:.2f}")
    """
    # Data size metrics
    total_observations = len(tbr_df)
    learning_observations = len(tbr_df[tbr_df["period"] == 0])
    test_observations = len(tbr_df[tbr_df["period"] == 1])

    # Prediction accuracy metrics (for test period)
    test_data = tbr_df[tbr_df["period"] == 1].copy()
    prediction_metrics = {}

    if not test_data.empty:
        prediction_errors = test_data["y"] - test_data["pred"]

        prediction_metrics = {
            "mae": float(np.mean(np.abs(prediction_errors))),
            "mse": float(np.mean(prediction_errors**2)),
            "rmse": float(np.sqrt(np.mean(prediction_errors**2))),
            "mape": float(np.mean(np.abs(prediction_errors / test_data["y"])) * 100),
            "mean_error": float(np.mean(prediction_errors)),
            "std_error": float(np.std(prediction_errors)),
        }

        # Prediction interval coverage
        pred_lower = test_data["pred"] - 1.96 * test_data["predsd"]
        pred_upper = test_data["pred"] + 1.96 * test_data["predsd"]
        coverage = float(
            np.mean((test_data["y"] >= pred_lower) & (test_data["y"] <= pred_upper))
        )
        prediction_metrics["interval_coverage"] = coverage

    # Model complexity metrics
    summary_row = tbr_summary.iloc[0]
    model_complexity = {
        "degrees_freedom": int(summary_row["t_dist_df"]),
        "sigma": float(summary_row["sigma"]),
        "r_squared_proxy": (
            1.0
            - (
                float(summary_row["sigma"]) ** 2
                / np.var(tbr_df[tbr_df["period"] == 0]["y"])
            )
            if learning_observations > 0
            else 0.0
        ),
    }

    # Efficiency score (composite metric)
    efficiency_components = []
    if prediction_metrics:
        # Lower MAPE is better (invert and normalize)
        mape_score = max(0, 1 - prediction_metrics["mape"] / 100)
        efficiency_components.append(mape_score)

        # Good interval coverage (target ~0.95)
        coverage_score = (
            1 - abs(prediction_metrics.get("interval_coverage", 0.95) - 0.95) * 2
        )
        efficiency_components.append(max(0, coverage_score))

    # Model parsimony (prefer simpler models)
    if learning_observations > 0:
        parsimony_score = min(
            1.0, learning_observations / 50
        )  # Normalize by reasonable sample size
        efficiency_components.append(parsimony_score)

    efficiency_score = np.mean(efficiency_components) if efficiency_components else 0.0

    return {
        "data_metrics": {
            "total_observations": total_observations,
            "learning_observations": learning_observations,
            "test_observations": test_observations,
            "learning_test_ratio": learning_observations / max(test_observations, 1),
        },
        "prediction_metrics": prediction_metrics,
        "model_complexity": model_complexity,
        "efficiency_score": float(efficiency_score),
        "performance_summary": {
            "data_quality": "Good" if learning_observations >= 20 else "Limited",
            "prediction_quality": (
                "Good"
                if prediction_metrics.get("mape", 100) < 10
                else "Moderate"
                if prediction_metrics.get("mape", 100) < 25
                else "Poor"
            ),
            "overall_performance": (
                "Excellent"
                if efficiency_score > 0.8
                else (
                    "Good"
                    if efficiency_score > 0.6
                    else "Moderate"
                    if efficiency_score > 0.4
                    else "Poor"
                )
            ),
        },
    }


def create_tbr_diagnostic_report(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    learning_data: Optional[pd.DataFrame] = None,
    include_detailed_analysis: bool = True,
) -> Dict[str, Union[str, Dict, List]]:
    """
    Create a structured diagnostic report for TBR analysis.

    Generates a complete diagnostic report combining all diagnostic functions
    into a structured, interpretable format suitable for analysis review
    and documentation.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        Daily TBR output required by :func:`diagnose_tbr_analysis`.
    tbr_summary : pd.DataFrame
        Non-empty summary required by :func:`diagnose_tbr_analysis`.
    learning_data : pd.DataFrame, optional
        Learning-period data containing ``x`` and ``y``. If omitted, rows with
        ``period == 0`` are used.
    include_detailed_analysis : bool, default True
        Whether to compute performance metrics and retain the complete
        diagnostic payload.

    Returns
    -------
    Dict[str, Union[str, bool, int, List[str], Dict, None]]
        Diagnostic-report mapping. See Notes for the stable top-level and
        nested schemas.

    Raises
    ------
    ValueError
        Propagated when required columns are missing, the summary is empty, or
        learning data are unusable.
    KeyError
        If explicitly supplied ``learning_data`` lacks ``x`` or ``y``.

    Notes
    -----
    The returned mapping has these stable keys:

    ``executive_summary`` : str
        Generated text based only on ``overall_validity`` and warning count.
    ``overall_validity`` : bool
        The warning-count heuristic from ``model_validation``.
    ``warnings_count`` : int
        Number of strings in ``model_validation["warnings"]``.
    ``key_findings`` : List[str]
        Available :math:`R^2`, assumption-status, MAPE, and fixed-band
        coverage observations formatted as text.
    ``recommendations`` : List[str]
        Heuristic suggestions returned by
        :func:`diagnose_tbr_analysis`.
    ``detailed_results`` : Dict[str, Any] or None
        When requested, contains the following exact nested schemas;
        otherwise ``None``.

        ``model_validation`` : Dict[str, Any]
            Exact mapping documented by :func:`validate_tbr_model`.
        ``diagnostic_summary`` : Dict[str, Any]
            Exact mapping documented by :func:`diagnose_tbr_analysis`.
        ``performance_metrics`` : Dict[str, Any]
            Exact mapping documented by :func:`assess_tbr_performance`.
        ``recommendations`` : List[str]
            Same recommendation list exposed at the report top level.

        Otherwise ``None``.
    ``report_timestamp`` : str
        ISO-formatted local timestamp from ``pd.Timestamp.now()``.

    The report is intended for review and communication. Its statuses,
    recommendations, and categorical wording are heuristic. In particular,
    the generated ``"PASSED"`` sentence reflects absence of emitted warnings;
    it does not establish that every modeling assumption holds, validate a
    causal interpretation, or quantify certainty about the treatment effect.

    See the :doc:`model-diagnostics methodology </mathematical_methodology>` for
    the assumptions and formal inferential quantities underlying TBR.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import create_tbr_diagnostic_report, perform_tbr_analysis
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> results = perform_tbr_analysis(
    ...     data=data, time_col="date", control_col="control", test_col="test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ...     level=0.90, threshold=0.0,
    ... )
    >>> tbr_df = results.tbr_dataframe()
    >>> tbr_summary = results.summary()
    >>> learning_data = tbr_df[tbr_df["period"] == 0]
    >>> report = create_tbr_diagnostic_report(tbr_df, tbr_summary, learning_data)
    >>> list(report) == [
    ...     "executive_summary", "overall_validity", "warnings_count",
    ...     "key_findings", "recommendations", "detailed_results",
    ...     "report_timestamp",
    ... ]
    True
    >>> print(report['executive_summary'])
    >>> print("Key findings:")
    >>> for finding in report['key_findings']:
    ...     print(f"  - {finding}")
    """
    # Run comprehensive diagnostics
    diagnostics = diagnose_tbr_analysis(
        tbr_df,
        tbr_summary,
        learning_data,
        include_performance=include_detailed_analysis,
    )

    # Extract key metrics for summary
    model_validation = diagnostics["model_validation"]
    if isinstance(model_validation, dict):
        model_valid = model_validation.get("overall_validity", False)
        warnings_count = len(model_validation.get("warnings", []))
    else:
        model_valid = False
        warnings_count = 0

    # Generate executive summary
    if model_valid:
        executive_summary = "TBR model validation PASSED. The analysis meets statistical requirements and assumptions."
    else:
        executive_summary = f"TBR model validation identified {warnings_count} issue(s) requiring attention."

    # Key findings
    key_findings = []

    # Add goodness of fit findings
    if isinstance(model_validation, dict):
        gof = model_validation.get("goodness_of_fit", {})
        if isinstance(gof, dict) and "r_squared" in gof:
            key_findings.append(
                f"Model explains {gof['r_squared']:.1%} of variance (R² = {gof['r_squared']:.3f})"
            )

        # Add assumption test findings
        assumptions = model_validation.get("assumption_tests", {})
        if isinstance(assumptions, dict) and "all_assumptions_valid" in assumptions:
            if assumptions["all_assumptions_valid"]:
                key_findings.append("All statistical assumptions are satisfied")
            else:
                failed_assumptions = []
                if not assumptions.get("normality_valid", True):
                    failed_assumptions.append("normality")
                if not assumptions.get("homoscedasticity_valid", True):
                    failed_assumptions.append("homoscedasticity")
                if not assumptions.get("independence_valid", True):
                    failed_assumptions.append("independence")
                key_findings.append(
                    f"Assumption violations: {', '.join(failed_assumptions)}"
                )

    # Add performance findings
    if (
        include_detailed_analysis
        and isinstance(diagnostics["performance_metrics"], dict)
        and "prediction_metrics" in diagnostics["performance_metrics"]
    ):
        pred_metrics = diagnostics["performance_metrics"]["prediction_metrics"]
        if isinstance(pred_metrics, dict) and "mape" in pred_metrics:
            key_findings.append(
                f"Prediction accuracy: {pred_metrics['mape']:.1f}% MAPE"
            )
        if isinstance(pred_metrics, dict) and "interval_coverage" in pred_metrics:
            key_findings.append(
                f"Prediction interval coverage: {pred_metrics['interval_coverage']:.1%}"
            )

    # Compile report
    report = {
        "executive_summary": executive_summary,
        "overall_validity": model_valid,
        "warnings_count": warnings_count,
        "key_findings": key_findings,
        "recommendations": diagnostics["recommendations"],
        "detailed_results": diagnostics if include_detailed_analysis else None,
        "report_timestamp": pd.Timestamp.now().isoformat(),
    }

    return report


def _generate_diagnostic_recommendations(
    model_validation: Dict,
    performance_metrics: Dict,
) -> List[str]:
    """
    Generate actionable recommendations based on diagnostic results.

    Internal helper function that analyzes diagnostic results and generates
    specific, actionable recommendations for improving TBR analysis quality.

    Parameters
    ----------
    model_validation : Dict
        Results from validate_tbr_model()
    diagnostic_summary : Dict
        Results from create_diagnostic_summary()
    performance_metrics : Dict
        Results from assess_tbr_performance()

    Returns
    -------
    List[str]
        List of actionable recommendations
    """
    recommendations = []

    # Model validation recommendations
    if not model_validation.get("overall_validity", True):
        recommendations.append(
            "Address model validation warnings before proceeding with analysis"
        )

    # Assumption-based recommendations
    assumptions = model_validation.get("assumption_tests", {})
    if isinstance(assumptions, dict):
        if not assumptions.get("normality_valid", True):
            recommendations.append(
                "Consider data transformation (log, square root) to improve residual normality"
            )
        if not assumptions.get("homoscedasticity_valid", True):
            recommendations.append(
                "Investigate heteroscedasticity - consider weighted regression or variance stabilization"
            )
        if not assumptions.get("independence_valid", True):
            recommendations.append(
                "Check for temporal patterns in data - consider time series methods if autocorrelation persists"
            )

    # Goodness of fit recommendations
    gof = model_validation.get("goodness_of_fit", {})
    if isinstance(gof, dict) and "r_squared" in gof:
        if gof["r_squared"] < 0.3:
            recommendations.append(
                "Low model fit (R² < 0.3) - consider additional predictors or different model specification"
            )
        elif gof["r_squared"] < 0.5:
            recommendations.append(
                "Moderate model fit - investigate potential confounding variables or model improvements"
            )

    # Outlier recommendations
    residual_analysis = model_validation.get("residual_analysis", {})
    if (
        isinstance(residual_analysis, dict)
        and "outlier_percentage" in residual_analysis
    ):
        if residual_analysis["outlier_percentage"] > 10:
            recommendations.append(
                "High percentage of outliers detected - investigate data quality and consider robust regression methods"
            )

    # Performance recommendations
    if performance_metrics and "prediction_metrics" in performance_metrics:
        pred_metrics = performance_metrics["prediction_metrics"]
        if "mape" in pred_metrics and pred_metrics["mape"] > 15:
            recommendations.append(
                "High prediction error (MAPE > 15%) - consider model refinement or additional validation"
            )
        if (
            "interval_coverage" in pred_metrics
            and pred_metrics["interval_coverage"] < 0.85
        ):
            recommendations.append(
                "Poor prediction interval coverage - uncertainty estimates may be unreliable"
            )

    # Data quality recommendations
    if performance_metrics and "data_metrics" in performance_metrics:
        data_metrics = performance_metrics["data_metrics"]
        if data_metrics["learning_observations"] < 20:
            recommendations.append(
                "Limited learning period data - consider collecting more historical data for robust model fitting"
            )
        if data_metrics["learning_test_ratio"] < 2:
            recommendations.append(
                "Short learning period relative to test period - consider longer historical baseline"
            )

    # Default recommendation if no issues found
    if not recommendations:
        recommendations.append(
            "Model diagnostics look good - proceed with confidence in TBR analysis results"
        )

    return recommendations
