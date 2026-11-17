"""
TBR Subinterval Analysis Module.

This module provides specialized functionality for custom time window analysis
in Time-Based Regression (TBR), enabling flexible analysis of treatment effects
over specific subintervals within the test period.

The subinterval analysis approach allows researchers to:
- Analyze treatment effects for specific time ranges
- Compare effects across different time windows
- Apply an explicitly chosen interval-exclusion rule across time windows
- Perform detailed temporal analysis of treatment impact
- Review effect variation across different intervals

Functions
---------
compute_interval_estimate_and_ci : Compute subinterval effect estimate and credible interval
analyze_multiple_subintervals : Analyze multiple time windows simultaneously
create_subinterval_summary : Create comprehensive subinterval analysis summary
validate_subinterval_parameters : Validate subinterval analysis parameters

Examples
--------
>>> import pandas as pd
>>> from tbr.analysis.subinterval import compute_interval_estimate_and_ci
>>> tbr_df = pd.DataFrame(
...     {
...         "period": [1, 1, 1],
...         "y": [110.0, 115.0, 118.0],
...         "x": [100.0, 103.0, 107.0],
...         "pred": [105.0, 108.0, 112.0],
...     }
... )
>>> tbr_summary = pd.DataFrame(
...     {
...         "sigma": [3.0],
...         "t_dist_df": [20],
...         "var_alpha": [4.0],
...         "var_beta": [0.0004],
...         "alpha_beta_cov": [-0.03],
...     }
... )
>>> result = compute_interval_estimate_and_ci(
...     tbr_df, tbr_summary,
...     start_position=1, end_position=2, ci_level=0.80
... )
>>> print(f"Positions 1-2 effect: {result['estimate']:.2f}")
>>> print(f"80% credible interval: [{result['lower']:.2f}, {result['upper']:.2f}]")

Analyze multiple subintervals:

>>> from tbr.analysis.subinterval import analyze_multiple_subintervals
>>> intervals = [(1, 2), (2, 3), (1, 3)]
>>> results = analyze_multiple_subintervals(
...     tbr_df, tbr_summary, intervals, ci_level=0.80
... )
>>> for i, result in enumerate(results):
...     start, end = intervals[i]
...     print(f"Positions {start}-{end}: {result['estimate']:.2f} "
...           f"[{result['lower']:.2f}, {result['upper']:.2f}]")

Create comprehensive summary:

>>> from tbr.analysis.subinterval import create_subinterval_summary
>>> summary = create_subinterval_summary(
...     tbr_df, tbr_summary, intervals=[(1, 2), (2, 3)], ci_level=0.80
... )
>>> print(
...     summary[['interval', 'estimate', 'lower', 'upper', 'interval_excludes_threshold']]
... )
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import core functionality for wrapping
from tbr.core.effects import compute_interval_estimate_and_ci as core_compute_interval


def compute_interval_estimate_and_ci(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_position: int,
    end_position: int,
    ci_level: float,
) -> Dict[str, float]:
    r"""
    Compute a subinterval effect estimate and credible interval.

    The selected interval has inclusive one-based bounds
    :math:`a=\mathrm{start\_position}` and
    :math:`b=\mathrm{end\_position}`. The value
    :math:`T_s=b-a+1` is ``n_observations``, the number of selected rows, each
    representing one test observation. Test rows are selected in their
    existing order.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        TBR output containing ``period``, ``y``, ``x``, and ``pred``. Rows with
        ``period == 1`` must be in test-row/time-point order.
    tbr_summary : pd.DataFrame
        Non-empty summary containing ``sigma``, ``t_dist_df``, ``var_alpha``,
        ``var_beta``, and ``alpha_beta_cov``. The legacy ``t_dist_df`` maps
        one way from ``degrees_freedom`` and stores residual/pretest-regression
        Student-t degrees of freedom. The legacy ``alpha_beta_cov`` maps one
        way from ``cov_alpha_beta`` and stores covariance between the estimated
        intercept and slope. Values are read from the last row.
    start_position : int
        One-based inclusive position :math:`a` of the first selected test
        observation. Must be at least 1.
    end_position : int
        One-based inclusive position :math:`b` of the last selected test
        observation. Must be at least ``start_position`` and no greater than
        the number of test observations.
    ci_level : float
        Credibility level in the open interval ``(0, 1)``.

    Returns
    -------
    Dict[str, float]
        Subinterval result mapping. See Notes for the stable schema.

    Raises
    ------
    ValueError
        If a required input column is absent, ``tbr_summary`` has no rows,
        ``tbr_df`` has no test rows, the positional range is invalid, or
        ``ci_level`` is outside ``(0, 1)``.
    TypeError
        If an input is not a DataFrame, a positional bound is not an integer, or
        ``ci_level`` is not numeric.

    Notes
    -----
    The returned mapping has these stable keys:

    ``estimate`` : floating-point scalar
        :math:`\Delta(a,b)=\sum_{t=a}^{b}(y_t-\hat{y}_t^*)`.
    ``se`` : floating-point scalar
        Standard error :math:`\sqrt{\mathbb{V}[\Delta(a,b)]}`.
    ``precision`` : floating-point scalar
        Credible-interval half-width, :math:`t_{\alpha/2,\nu}` times ``se``.
    ``lower`` : floating-point scalar
        Lower credible bound, ``estimate - precision``.
    ``upper`` : floating-point scalar
        Upper credible bound, ``estimate + precision``.

    The subinterval analysis uses the same mathematical foundation as the full
    TBR analysis, with credible intervals calculated using the t-distribution:

    .. math::
        CI = estimate ± t_{α/2,df} × se

    where the standard error combines model uncertainty and residual noise:

    .. math::
        se = \sqrt{T_s × σ^2 + T_s^2 × v}

    with

    .. math::
        v = \mathbb{V}[\hat{\beta}_0]
        + 2\bar{x}\operatorname{Cov}(\hat{\beta}_0,\hat{\beta}_1)
        + \bar{x}^2\mathbb{V}[\hat{\beta}_1]

    where :math:`\bar{x}` is the mean control value over the subinterval. The
    Here :math:`T_s` is the number of selected test observations. The term
    :math:`T_s^2 × v` is the variance of the summed counterfactual predictions,
    which are correlated because they share the estimated coefficients;
    :math:`T_s × σ^2` is the residual noise.

    In the credible-interval equation, alpha = 1 - ci_level is the interval
    tail probability.

    Mathematical Foundation
    -----------------------
    The subinterval estimate is calculated as:

    .. math::
        estimate = \sum_{i=start}^{end} (y_i - pred_i)

    where y_i is the observed value and pred_i is the counterfactual prediction
    at test-row position i within the subinterval.

    For mathematical background, see the
    :doc:`mathematical methodology guide </mathematical_methodology>`,
    especially the section on subinterval analysis.

    Examples
    --------
    Analyze the first two test-row positions:

    >>> import pandas as pd
    >>> from tbr import compute_interval_estimate_and_ci
    >>> tbr_df = pd.DataFrame(
    ...     {
    ...         "period": [1, 1, 1],
    ...         "y": [110.0, 115.0, 118.0],
    ...         "x": [100.0, 103.0, 107.0],
    ...         "pred": [105.0, 108.0, 112.0],
    ...     }
    ... )
    >>> tbr_summary = pd.DataFrame(
    ...     {
    ...         "sigma": [3.0],
    ...         "t_dist_df": [20],
    ...         "var_alpha": [4.0],
    ...         "var_beta": [0.0004],
    ...         "alpha_beta_cov": [-0.03],
    ...     }
    ... )
    >>> result = compute_interval_estimate_and_ci(
    ...     tbr_df, tbr_summary,
    ...     start_position=1, end_position=2, ci_level=0.80
    ... )
    >>> list(result) == ["estimate", "se", "precision", "lower", "upper"]
    True
    >>> bool(result["estimate"] == 12.0)
    True
    >>> bool(abs(result["precision"] - (result["upper"] - result["lower"]) / 2) < 1e-12)
    True
    >>> print(f"Positions 1-2 effect: {result['estimate']:.2f}")
    >>> print(f"80% credible interval: [{result['lower']:.2f}, {result['upper']:.2f}]")
    >>> print(f"Precision: ±{result['precision']:.2f}")

    Analyze one test-row position:

    >>> position_3_result = compute_interval_estimate_and_ci(
    ...     tbr_df, tbr_summary,
    ...     start_position=3, end_position=3, ci_level=0.95
    ... )
    >>> print(f"Position 3 effect: {position_3_result['estimate']:.2f}")

    Compare different credibility levels:

    >>> result_80 = compute_interval_estimate_and_ci(
    ...     tbr_df, tbr_summary,
    ...     start_position=1, end_position=3, ci_level=0.80
    ... )
    >>> result_95 = compute_interval_estimate_and_ci(
    ...     tbr_df, tbr_summary,
    ...     start_position=1, end_position=3, ci_level=0.95
    ... )
    >>> print(f"80% interval width: {result_80['upper'] - result_80['lower']:.2f}")
    >>> print(f"95% interval width: {result_95['upper'] - result_95['lower']:.2f}")
    """
    validate_subinterval_parameters(
        tbr_df,
        tbr_summary,
        start_position,
        end_position,
        ci_level,
    )
    return core_compute_interval(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_position=start_position,
        end_position=end_position,
        ci_level=ci_level,
    )


def analyze_multiple_subintervals(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    intervals: List[Tuple[int, int]],
    ci_level: float = 0.80,
) -> List[Dict[str, float]]:
    """
    Analyze multiple subintervals simultaneously for comparative analysis.

    This function performs subinterval analysis for multiple time windows within
    the test period, enabling comparative analysis of treatment effects across
    different temporal segments. This is particularly useful for understanding
    how treatment effects evolve over time and identifying periods of strongest
    or weakest impact.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        TBR output with columns ``y``, ``x``, ``pred``, and ``period``. Rows
        with ``period == 1`` are interpreted as test rows or time points in
        their existing order.
    tbr_summary : pd.DataFrame
        TBR summary containing ``sigma``, ``t_dist_df``, ``var_alpha``,
        ``var_beta``, and ``alpha_beta_cov``. The legacy ``t_dist_df`` maps
        one way from ``degrees_freedom`` and stores residual/pretest-regression
        Student-t degrees of freedom. The legacy ``alpha_beta_cov`` maps one
        way from ``cov_alpha_beta`` and stores covariance between the estimated
        intercept and slope.
    intervals : List[Tuple[int, int]]
        Non-empty list of ``(start_position, end_position)`` pairs denoting
        inclusive one-based positions among ordered test observations. Each
        start must be at least 1 and no greater than its end, and each end no
        greater than the number of test observations.
    ci_level : float, default=0.80
        Credible interval level for all subintervals in the open interval
        ``(0, 1)``.

    Returns
    -------
    List[Dict[str, float]]
        Results in the same order as ``intervals``. See Notes for the stable
        mapping schema.

    Raises
    ------
    ValueError
        If ``intervals`` is empty, ``ci_level`` is outside ``(0, 1)``, an
        interval starts below 1, starts after its end, or ends beyond the test
        rows, a required input column is absent, or ``tbr_summary`` has no
        rows.
    TypeError
        Propagated when an input is not a DataFrame or an interval bound is not
        an integer.

    Notes
    -----
    Each returned mapping has these stable keys:

    ``estimate`` : floating-point scalar
        Subinterval cumulative-effect estimate.
    ``se`` : floating-point scalar
        Standard error of the estimate.
    ``precision`` : floating-point scalar
        Credible-interval half-width.
    ``lower`` : floating-point scalar
        Lower credible bound.
    ``upper`` : floating-point scalar
        Upper credible bound.

    This function is equivalent to calling compute_interval_estimate_and_ci()
    for each interval individually, but provides a convenient interface for
    batch analysis and ensures consistent parameter validation.

    The results maintain the same mathematical rigor as individual subinterval
    analyses, with each interval analyzed independently using the full TBR
    statistical framework.

    Examples
    --------
    Compare effects across multiple windows:

    >>> import pandas as pd
    >>> from tbr import analyze_multiple_subintervals
    >>> tbr_df = pd.DataFrame(
    ...     {
    ...         "period": [1, 1, 1],
    ...         "y": [110.0, 115.0, 118.0],
    ...         "x": [100.0, 103.0, 107.0],
    ...         "pred": [105.0, 108.0, 112.0],
    ...     }
    ... )
    >>> tbr_summary = pd.DataFrame(
    ...     {
    ...         "sigma": [3.0],
    ...         "t_dist_df": [20],
    ...         "var_alpha": [4.0],
    ...         "var_beta": [0.0004],
    ...         "alpha_beta_cov": [-0.03],
    ...     }
    ... )
    >>> intervals = [(1, 2), (2, 3)]
    >>> results = analyze_multiple_subintervals(
    ...     tbr_df, tbr_summary, intervals, ci_level=0.80
    ... )
    >>> [result["estimate"] for result in results] == [12.0, 13.0]
    True
    >>> for i, result in enumerate(results, 1):
    ...     print(f"Window {i}: {result['estimate']:.2f} "
    ...           f"[{result['lower']:.2f}, {result['upper']:.2f}]")

    Analyze overlapping intervals:

    >>> intervals = [(1, 2), (1, 3), (2, 3)]  # Overlapping windows
    >>> results = analyze_multiple_subintervals(
    ...     tbr_df, tbr_summary, intervals, ci_level=0.90
    ... )
    >>> for i, (start, end) in enumerate(intervals):
    ...     result = results[i]
    ...     print(f"Positions {start}-{end}: {result['estimate']:.2f}")

    Compare different interval lengths:

    >>> intervals = [(1, 1), (1, 2), (1, 3)]
    >>> results = analyze_multiple_subintervals(
    ...     tbr_df, tbr_summary, intervals
    ... )
    >>> for i, (start, end) in enumerate(intervals):
    ...     n_positions = end - start + 1
    ...     average = results[i]['estimate'] / n_positions
    ...     print(f"{n_positions}-position interval: {average:.2f} per position")
    """
    # Validate inputs
    if not intervals:
        raise ValueError("Intervals list cannot be empty")

    if not (0 < ci_level < 1):
        raise ValueError(f"ci_level must be between 0 and 1, got: {ci_level}")

    # Validate each interval
    for i, (start_position, end_position) in enumerate(intervals):
        if start_position > end_position:
            raise ValueError(
                f"Interval {i}: start_position ({start_position}) cannot be "
                f"greater than end_position ({end_position})"
            )
        if start_position < 1:
            raise ValueError(
                f"Interval {i}: start_position must be >= 1, " f"got: {start_position}"
            )

    # Analyze each interval
    results = []
    for start_position, end_position in intervals:
        result = compute_interval_estimate_and_ci(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_position=start_position,
            end_position=end_position,
            ci_level=ci_level,
        )
        results.append(result)

    return results


def create_subinterval_summary(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    intervals: List[Tuple[int, int]],
    ci_level: float = 0.80,
    threshold: float = 0.0,
) -> pd.DataFrame:
    r"""
    Create a structured summary DataFrame for multiple subinterval analyses.

    This function generates a structured summary of subinterval analyses,
    providing a tabular view of treatment effects across multiple time windows.
    The summary includes effect estimates, credible intervals, and an explicitly
    defined interval-exclusion flag for comparing temporal segments.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        TBR output containing ``period``, ``y``, ``x``, and ``pred``.
        Test rows are interpreted in their existing order.
    tbr_summary : pd.DataFrame
        Non-empty summary containing ``sigma``, legacy degrees-of-freedom
        column ``t_dist_df``, ``var_alpha``, ``var_beta``, and
        ``alpha_beta_cov``. The legacy ``t_dist_df`` maps one way from
        ``degrees_freedom`` and stores residual/pretest-regression Student-t
        degrees of freedom. The legacy ``alpha_beta_cov`` maps one way from
        ``cov_alpha_beta`` and stores covariance between the estimated
        intercept and slope.
    intervals : List[Tuple[int, int]]
        Non-empty list of inclusive one-based
        ``(start_position, end_position)`` pairs among ordered test
        observations in their existing order. Each pair selects
        T_s = end_position - start_position + 1 test observations.
    ci_level : float, default=0.80
        Credible interval level for all analyses.
    threshold : float, default=0.0
        Effect threshold for the ``interval_excludes_threshold`` column. An
        interval is flagged when its credible interval does not include this
        value.

    Returns
    -------
    pd.DataFrame
        One row per input interval, preserving input order. See Notes for the
        stable columns.

    Raises
    ------
    ValueError
        Propagated when ``intervals`` is empty, ``ci_level`` is outside
        ``(0, 1)``, a pair starts below 1, starts after its end, or ends beyond
        the test rows, a required input column is absent, or ``tbr_summary``
        has no rows.
    TypeError
        Propagated when an input is not a DataFrame or an interval bound is not
        an integer.

    Notes
    -----
    The returned DataFrame has these stable columns:

    ``interval`` : object
        Generated runtime label. A single-observation interval is formatted as
        ``"Test-period observation {start_position}"``; a range is formatted
        as ``"Test-period observations {start_position} through
        {end_position} (inclusive)"``. The bounds are one-based observation
        positions.
    ``start_position`` : integer
        Inclusive one-based position of the first selected test observation,
        typically ``int64``.
    ``end_position`` : integer
        Inclusive one-based position of the last selected test observation,
        typically ``int64``.
    ``n_observations`` : integer
        ``n_observations`` is the inclusive count between both endpoints for
        rows that each represent one test observation:
        :math:`T_s=\mathrm{end\_position}-\mathrm{start\_position}+1`.
        It is typically ``int64``.
    ``estimate`` : floating-point
        :math:`\Delta(a,b)`, where
        :math:`a=\mathrm{start\_position}` and
        :math:`b=\mathrm{end\_position}`, typically ``float64``.
    ``se`` : floating-point
        Standard error of the estimate, typically ``float64``.
    ``precision`` : floating-point
        Credible-interval half-width, typically ``float64``.
    ``lower`` : floating-point
        Lower credible bound, typically ``float64``.
    ``upper`` : floating-point
        Upper credible bound, typically ``float64``.
    ``interval_excludes_threshold`` : Boolean
        Whether ``lower > threshold`` or ``upper < threshold``. Equality does
        not count as exclusion.
    ``average_effect_per_observation`` : floating-point
        ``estimate / n_observations``, typically ``float64``.
    ``ci_level`` : floating-point
        Credibility level, typically ``float64``.

    The ``interval_excludes_threshold`` column records whether the credible
    interval lies entirely above or below ``threshold``.
    It is not a hypothesis test or a posterior-probability result.

    ``average_effect_per_observation`` is the total interval effect divided by
    the selected observation count. It is not an elapsed-time rate.

    Examples
    --------
    Create a summary for multiple subintervals:

    >>> import pandas as pd
    >>> from tbr import create_subinterval_summary
    >>> tbr_df = pd.DataFrame(
    ...     {
    ...         "period": [1, 1, 1],
    ...         "y": [110.0, 115.0, 118.0],
    ...         "x": [100.0, 103.0, 107.0],
    ...         "pred": [105.0, 108.0, 112.0],
    ...     }
    ... )
    >>> tbr_summary = pd.DataFrame(
    ...     {
    ...         "sigma": [3.0],
    ...         "t_dist_df": [20],
    ...         "var_alpha": [4.0],
    ...         "var_beta": [0.0004],
    ...         "alpha_beta_cov": [-0.03],
    ...     }
    ... )
    >>> intervals = [(1, 2), (2, 3), (1, 3)]
    >>> summary = create_subinterval_summary(
    ...     tbr_df, tbr_summary, intervals, ci_level=0.80
    ... )
    >>> list(summary.columns) == [
    ...     "interval", "start_position", "end_position", "n_observations",
    ...     "estimate", "se",
    ...     "precision", "lower", "upper", "interval_excludes_threshold",
    ...     "average_effect_per_observation", "ci_level",
    ... ]
    True
    >>> # interval reports inclusive test-observation positions.
    >>> summary["interval"].tolist() == [
    ...     "Test-period observations 1 through 2 (inclusive)",
    ...     "Test-period observations 2 through 3 (inclusive)",
    ...     "Test-period observations 1 through 3 (inclusive)",
    ... ]
    True
    >>> print(summary[['interval', 'estimate', 'interval_excludes_threshold']])

    Analyze with a custom threshold:

    >>> summary = create_subinterval_summary(
    ...     tbr_df, tbr_summary, intervals,
    ...     threshold=10.0  # Credible interval must exclude 10
    ... )
    >>> flagged_intervals = summary[summary['interval_excludes_threshold']]
    >>> print(f"Intervals excluding threshold: {len(flagged_intervals)}")

    Compare the average effect per observation:

    >>> summary = create_subinterval_summary(
    ...     tbr_df, tbr_summary, [(1, 1), (1, 2), (1, 3)]
    ... )
    >>> print(
    ...     summary[
    ...         ["interval", "average_effect_per_observation"]
    ...     ].sort_values("average_effect_per_observation")
    ... )
    """
    # Analyze all intervals
    results = analyze_multiple_subintervals(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        intervals=intervals,
        ci_level=ci_level,
    )

    # Create summary DataFrame
    summary_data = []
    for _i, ((start_position, end_position), result) in enumerate(
        zip(intervals, results)
    ):
        n_observations = end_position - start_position + 1
        if start_position == end_position:
            interval_label = f"Test-period observation {start_position}"
        else:
            interval_label = (
                f"Test-period observations {start_position} through "
                f"{end_position} (inclusive)"
            )

        interval_excludes_threshold = (
            result["lower"] > threshold or result["upper"] < threshold
        )

        summary_data.append(
            {
                "interval": interval_label,
                "start_position": start_position,
                "end_position": end_position,
                "n_observations": n_observations,
                "estimate": result["estimate"],
                "se": result["se"],
                "precision": result["precision"],
                "lower": result["lower"],
                "upper": result["upper"],
                "interval_excludes_threshold": interval_excludes_threshold,
                "average_effect_per_observation": (result["estimate"] / n_observations),
                "ci_level": ci_level,
            }
        )

    return pd.DataFrame(summary_data)


def validate_subinterval_parameters(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_position: int,
    end_position: int,
    ci_level: float,
) -> None:
    """
    Validate parameters for subinterval analysis.

    This function performs comprehensive validation of input parameters for
    subinterval analysis, ensuring that all inputs are valid and consistent
    before performing the analysis. It provides clear error messages for
    invalid inputs.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        TBR output DataFrame to validate.
    tbr_summary : pd.DataFrame
        TBR summary DataFrame to validate.
    start_position : int
        Inclusive one-based start position to validate.
    end_position : int
        Inclusive one-based end position to validate.
    ci_level : float
        Credible interval level to validate.

    Raises
    ------
    ValueError
        If any parameter is invalid, with specific error message describing
        the validation failure.
    TypeError
        If parameters have incorrect types.

    Notes
    -----
    This function is called internally by other subinterval analysis functions
    but can also be used directly for parameter validation before analysis.

    The validation checks include:
    - DataFrame structure and required columns
    - Positional range validity and logical consistency
    - Credibility level bounds
    - Test period data availability

    Examples
    --------
    Validate parameters before analysis:

    >>> import pandas as pd
    >>> from tbr.analysis.subinterval import validate_subinterval_parameters
    >>> tbr_df = pd.DataFrame(
    ...     {
    ...         "period": [1, 1, 1],
    ...         "y": [110.0, 115.0, 118.0],
    ...         "x": [100.0, 103.0, 107.0],
    ...         "pred": [105.0, 108.0, 112.0],
    ...     }
    ... )
    >>> tbr_summary = pd.DataFrame(
    ...     {
    ...         "sigma": [3.0],
    ...         "t_dist_df": [20],
    ...         "var_alpha": [4.0],
    ...         "var_beta": [0.0004],
    ...         "alpha_beta_cov": [-0.03],
    ...     }
    ... )
    >>> validate_subinterval_parameters(
    ...     tbr_df, tbr_summary,
    ...     start_position=1, end_position=2, ci_level=0.80
    ... )  # No error

    An out-of-range test-row position raises ``ValueError``:

    >>> validate_subinterval_parameters(
    ...     tbr_df, tbr_summary,
    ...     start_position=5, end_position=10, ci_level=0.80
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...
    """
    # Validate DataFrame types
    if not isinstance(tbr_df, pd.DataFrame):
        raise TypeError("tbr_df must be a pandas DataFrame")

    if not isinstance(tbr_summary, pd.DataFrame):
        raise TypeError("tbr_summary must be a pandas DataFrame")

    # Validate DataFrame structure
    required_tbr_cols = ["y", "x", "pred", "period"]
    missing_tbr_cols = [col for col in required_tbr_cols if col not in tbr_df.columns]
    if missing_tbr_cols:
        raise ValueError(f"tbr_df missing required columns: {missing_tbr_cols}")

    required_summary_cols = [
        "sigma",
        "t_dist_df",
        "var_alpha",
        "var_beta",
        "alpha_beta_cov",
    ]
    missing_summary_cols = [
        col for col in required_summary_cols if col not in tbr_summary.columns
    ]
    if missing_summary_cols:
        raise ValueError(
            f"tbr_summary missing required columns: {missing_summary_cols}"
        )

    if tbr_summary.empty:
        raise ValueError("tbr_summary cannot be empty")

    # Validate test period data exists
    test_data = tbr_df[tbr_df["period"] == 1]
    if test_data.empty:
        raise ValueError("No test period data found (period == 1)")

    # Validate positional bounds
    if not isinstance(start_position, (int, np.integer)) or not isinstance(
        end_position, (int, np.integer)
    ):
        raise TypeError("start_position and end_position must be integers")

    if start_position < 1:
        raise ValueError(f"start_position must be >= 1, got: {start_position}")

    if end_position < start_position:
        raise ValueError(
            f"end_position ({end_position}) must be >= "
            f"start_position ({start_position})"
        )

    max_test_observations = len(test_data)
    if end_position > max_test_observations:
        raise ValueError(
            f"end_position ({end_position}) exceeds the number of test observations "
            f"({max_test_observations})"
        )

    # Validate credibility level
    if not isinstance(ci_level, (int, float, np.integer, np.floating)):
        raise TypeError("ci_level must be a number")

    if not (0 < ci_level < 1):
        raise ValueError(f"ci_level must be between 0 and 1, got: {ci_level}")
