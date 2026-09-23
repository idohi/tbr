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
...         "pred": [105.0, 108.0, 112.0],
...         "estsd": [2.0, 2.1, 2.2],
...     }
... )
>>> tbr_summary = pd.DataFrame({"sigma": [3.0], "t_dist_df": [20]})
>>> result = compute_interval_estimate_and_ci(
...     tbr_df, tbr_summary, start_day=1, end_day=2, ci_level=0.80
... )
>>> print(f"Days 1-2 effect: {result['estimate']:.2f}")
>>> print(f"80% credible interval: [{result['lower']:.2f}, {result['upper']:.2f}]")

Analyze multiple subintervals:

>>> from tbr.analysis.subinterval import analyze_multiple_subintervals
>>> intervals = [(1, 2), (2, 3), (1, 3)]
>>> results = analyze_multiple_subintervals(
...     tbr_df, tbr_summary, intervals, ci_level=0.80
... )
>>> for i, result in enumerate(results):
...     start, end = intervals[i]
...     print(f"Days {start}-{end}: {result['estimate']:.2f} "
...           f"[{result['lower']:.2f}, {result['upper']:.2f}]")

Create comprehensive summary:

>>> from tbr.analysis.subinterval import create_subinterval_summary
>>> summary = create_subinterval_summary(
...     tbr_df, tbr_summary, intervals=[(1, 2), (2, 3)], ci_level=0.80
... )
>>> print(summary[['interval', 'estimate', 'lower', 'upper', 'significant']])
"""

from typing import Dict, List, Tuple

import pandas as pd

# Import core functionality for wrapping
from tbr.core.effects import compute_interval_estimate_and_ci as core_compute_interval


def compute_interval_estimate_and_ci(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_day: int,
    end_day: int,
    ci_level: float,
) -> Dict[str, float]:
    r"""
    Compute a subinterval effect estimate and credible interval.

    The selected interval has inclusive one-based bounds
    :math:`a=\mathrm{start\_day}` and :math:`b=\mathrm{end\_day}`, with
    :math:`T_s=b-a+1`. Test rows are selected in their existing order.

    Parameters
    ----------
    tbr_df : pd.DataFrame
        Daily TBR output containing ``period``, ``y``, ``pred``, and ``estsd``.
        Rows with ``period == 1`` must be in test-day order.
    tbr_summary : pd.DataFrame
        Non-empty summary containing ``sigma`` and ``t_dist_df``. Values are
        read from the last row.
    start_day : int
        Requested one-based start :math:`a`. This direct helper does not
        validate the value.
    end_day : int
        Requested inclusive one-based end :math:`b`. This direct helper does
        not validate the value.
    ci_level : float
        Credibility level. This direct helper does not validate its range.

    Returns
    -------
    Dict[str, float]
        Subinterval result mapping. See Notes for the stable schema.

    Raises
    ------
    KeyError
        If a required input column is absent.
    IndexError
        If ``tbr_summary`` has no rows.

    Notes
    -----
    The returned mapping has these stable keys:

    ``estimate`` : floating-point scalar
        :math:`\Delta(a,b)=\sum_{t=a}^{b}(y_t-\hat{y}_t^*)`.
    ``precision`` : floating-point scalar
        Credible-interval half-width.
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
        se = \sqrt{\sum_{i=start}^{end} estsd_i^2 + n_{days} × σ^2}

    The posterior variance accounts for both prediction uncertainty (estsd²)
    and residual noise (σ²) over the subinterval period.

    Mathematical Foundation
    -----------------------
    The subinterval estimate is calculated as:

    .. math::
        estimate = \sum_{i=start}^{end} (y_i - pred_i)

    where y_i is the observed value and pred_i is the counterfactual prediction
    for day i within the subinterval.

    For mathematical background, see the
    :doc:`mathematical methodology guide </mathematical_methodology>`,
    especially the section on subinterval analysis.

    Examples
    --------
    Analyze effect for the first two days of a test period:

    >>> import pandas as pd
    >>> from tbr import compute_interval_estimate_and_ci
    >>> tbr_df = pd.DataFrame(
    ...     {
    ...         "period": [1, 1, 1],
    ...         "y": [110.0, 115.0, 118.0],
    ...         "pred": [105.0, 108.0, 112.0],
    ...         "estsd": [2.0, 2.1, 2.2],
    ...     }
    ... )
    >>> tbr_summary = pd.DataFrame({"sigma": [3.0], "t_dist_df": [20]})
    >>> result = compute_interval_estimate_and_ci(
    ...     tbr_df, tbr_summary, start_day=1, end_day=2, ci_level=0.80
    ... )
    >>> list(result) == ["estimate", "precision", "lower", "upper"]
    True
    >>> result["estimate"] == 12.0
    True
    >>> abs(result["precision"] - (result["upper"] - result["lower"]) / 2) < 1e-12
    True
    >>> print(f"Days 1-2 effect: {result['estimate']:.2f}")
    >>> print(f"80% credible interval: [{result['lower']:.2f}, {result['upper']:.2f}]")
    >>> print(f"Precision: ±{result['precision']:.2f}")

    Analyze single day effect:

    >>> day_3_result = compute_interval_estimate_and_ci(
    ...     tbr_df, tbr_summary, start_day=3, end_day=3, ci_level=0.95
    ... )
    >>> print(f"Day 3 effect: {day_3_result['estimate']:.2f}")

    Compare different credibility levels:

    >>> result_80 = compute_interval_estimate_and_ci(
    ...     tbr_df, tbr_summary, start_day=1, end_day=3, ci_level=0.80
    ... )
    >>> result_95 = compute_interval_estimate_and_ci(
    ...     tbr_df, tbr_summary, start_day=1, end_day=3, ci_level=0.95
    ... )
    >>> print(f"80% interval width: {result_80['upper'] - result_80['lower']:.2f}")
    >>> print(f"95% interval width: {result_95['upper'] - result_95['lower']:.2f}")
    """
    return core_compute_interval(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=start_day,
        end_day=end_day,
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
        TBR daily output with columns 'y', 'pred', 'period', 'estsd'.
        Must contain one row per test day with ``period == 1``, ordered
        chronologically.
    tbr_summary : pd.DataFrame
        TBR summary containing 'sigma' and 't_dist_df' parameters.
    intervals : List[Tuple[int, int]]
        Non-empty list of ``(start_day, end_day)`` pairs. Bounds are inclusive
        one-based positions. Each start must be at least 1 and no greater than
        its end. This wrapper does not reject an end beyond the available rows.
    ci_level : float, default=0.80
        Credible interval level for all subintervals.
        Must be between 0 and 1.

    Returns
    -------
    List[Dict[str, float]]
        Results in the same order as ``intervals``. See Notes for the stable
        mapping schema.

    Raises
    ------
    ValueError
        If ``intervals`` is empty, ``ci_level`` is outside ``(0, 1)``, or an
        interval starts below 1 or starts after its end.
    KeyError
        Propagated when a required input column is absent.
    IndexError
        Propagated when ``tbr_summary`` has no rows.

    Notes
    -----
    Each returned mapping has these stable keys:

    ``estimate`` : floating-point scalar
        Subinterval cumulative-effect estimate.
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
    ...         "pred": [105.0, 108.0, 112.0],
    ...         "estsd": [2.0, 2.1, 2.2],
    ...     }
    ... )
    >>> tbr_summary = pd.DataFrame({"sigma": [3.0], "t_dist_df": [20]})
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
    ...     print(f"Days {start}-{end}: {result['estimate']:.2f}")

    Compare different interval lengths:

    >>> intervals = [(1, 1), (1, 2), (1, 3)]
    >>> results = analyze_multiple_subintervals(
    ...     tbr_df, tbr_summary, intervals
    ... )
    >>> for i, (start, end) in enumerate(intervals):
    ...     days = end - start + 1
    ...     avg_daily = results[i]['estimate'] / days
    ...     print(f"{days}-day period: {avg_daily:.2f} per day")
    """
    # Validate inputs
    if not intervals:
        raise ValueError("Intervals list cannot be empty")

    if not (0 < ci_level < 1):
        raise ValueError(f"ci_level must be between 0 and 1, got: {ci_level}")

    # Validate each interval
    for i, (start_day, end_day) in enumerate(intervals):
        if start_day > end_day:
            raise ValueError(
                f"Interval {i}: start_day ({start_day}) cannot be greater than "
                f"end_day ({end_day})"
            )
        if start_day < 1:
            raise ValueError(f"Interval {i}: start_day must be >= 1, got: {start_day}")

    # Analyze each interval
    results = []
    for start_day, end_day in intervals:
        result = compute_interval_estimate_and_ci(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_day=start_day,
            end_day=end_day,
            ci_level=ci_level,
        )
        results.append(result)

    return results


def create_subinterval_summary(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    intervals: List[Tuple[int, int]],
    ci_level: float = 0.80,
    significance_threshold: float = 0.0,
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
        Daily TBR output containing ``period``, ``y``, ``pred``, and ``estsd``.
        Test rows are interpreted in their existing order.
    tbr_summary : pd.DataFrame
        Non-empty summary containing ``sigma`` and legacy degrees-of-freedom
        column ``t_dist_df``.
    intervals : List[Tuple[int, int]]
        Non-empty list of inclusive one-based ``(start_day, end_day)`` pairs.
    ci_level : float, default=0.80
        Credible interval level for all analyses.
    significance_threshold : float, default=0.0
        Threshold for the interval-exclusion flag. An interval is flagged when
        its credible interval does not include this threshold value.

    Returns
    -------
    pd.DataFrame
        One row per input interval, preserving input order. See Notes for the
        stable columns.

    Raises
    ------
    ValueError
        Propagated when ``intervals`` is empty, ``ci_level`` is outside
        ``(0, 1)``, or a pair starts below 1 or starts after its end.
    KeyError
        Propagated when a required input column is absent.
    IndexError
        Propagated when ``tbr_summary`` has no rows.

    Notes
    -----
    The returned DataFrame has these stable columns:

    ``interval`` : object
        Label formatted as ``"Days {start_day}-{end_day}"``.
    ``start_day`` : integer
        Inclusive one-based start, normally ``int64``.
    ``end_day`` : integer
        Inclusive one-based end, normally ``int64``.
    ``days`` : integer
        :math:`T_s=\mathrm{end\_day}-\mathrm{start\_day}+1`, normally
        ``int64``. Structured result objects call this value ``n_days``.
    ``estimate`` : floating-point
        :math:`\Delta(a,b)`, normally ``float64``.
    ``precision`` : floating-point
        Credible-interval half-width, normally ``float64``.
    ``lower`` : floating-point
        Lower credible bound, normally ``float64``.
    ``upper`` : floating-point
        Upper credible bound, normally ``float64``.
    ``significant`` : Boolean
        Whether ``lower > significance_threshold`` or
        ``upper < significance_threshold``. Equality does not count as
        exclusion; this is not a posterior-probability result.
    ``avg_daily_effect`` : floating-point
        ``estimate / days``, normally ``float64``.
    ``ci_level`` : floating-point
        Credibility level, normally ``float64``.

    The ``significant`` column is a legacy interval-exclusion flag. It records
    whether the credible interval lies entirely above or below
    ``significance_threshold``. It is not a hypothesis test or a
    posterior-probability result.

    The average daily effect is calculated as the total interval effect
    divided by the number of days, providing a normalized comparison
    metric across intervals of different lengths.

    Examples
    --------
    Create a summary for multiple subintervals:

    >>> import pandas as pd
    >>> from tbr import create_subinterval_summary
    >>> tbr_df = pd.DataFrame(
    ...     {
    ...         "period": [1, 1, 1],
    ...         "y": [110.0, 115.0, 118.0],
    ...         "pred": [105.0, 108.0, 112.0],
    ...         "estsd": [2.0, 2.1, 2.2],
    ...     }
    ... )
    >>> tbr_summary = pd.DataFrame({"sigma": [3.0], "t_dist_df": [20]})
    >>> intervals = [(1, 2), (2, 3), (1, 3)]
    >>> summary = create_subinterval_summary(
    ...     tbr_df, tbr_summary, intervals, ci_level=0.80
    ... )
    >>> list(summary.columns) == [
    ...     "interval", "start_day", "end_day", "days", "estimate",
    ...     "precision", "lower", "upper", "significant",
    ...     "avg_daily_effect", "ci_level",
    ... ]
    True
    >>> summary["interval"].tolist() == ["Days 1-2", "Days 2-3", "Days 1-3"]
    True
    >>> print(summary[['interval', 'estimate', 'significant']])

    Analyze with a custom interval-exclusion threshold:

    >>> summary = create_subinterval_summary(
    ...     tbr_df, tbr_summary, intervals,
    ...     significance_threshold=10.0  # Credible interval must exclude 10
    ... )
    >>> flagged_intervals = summary[summary['significant']]
    >>> print(f"Intervals excluding threshold: {len(flagged_intervals)}")

    Compare average daily effects:

    >>> summary = create_subinterval_summary(
    ...     tbr_df, tbr_summary, [(1, 1), (1, 2), (1, 3)]
    ... )
    >>> print(summary[['interval', 'avg_daily_effect']].sort_values('avg_daily_effect'))
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
    for _i, ((start_day, end_day), result) in enumerate(zip(intervals, results)):
        days = end_day - start_day + 1

        # Check significance (CI excludes threshold)
        significant = (
            result["lower"] > significance_threshold
            or result["upper"] < significance_threshold
        )

        summary_data.append(
            {
                "interval": f"Days {start_day}-{end_day}",
                "start_day": start_day,
                "end_day": end_day,
                "days": days,
                "estimate": result["estimate"],
                "precision": result["precision"],
                "lower": result["lower"],
                "upper": result["upper"],
                "significant": significant,
                "avg_daily_effect": result["estimate"] / days,
                "ci_level": ci_level,
            }
        )

    return pd.DataFrame(summary_data)


def validate_subinterval_parameters(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_day: int,
    end_day: int,
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
        TBR daily output DataFrame to validate.
    tbr_summary : pd.DataFrame
        TBR summary DataFrame to validate.
    start_day : int
        Start day of subinterval to validate.
    end_day : int
        End day of subinterval to validate.
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
    - Day range validity and logical consistency
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
    ...         "pred": [105.0, 108.0, 112.0],
    ...         "estsd": [2.0, 2.1, 2.2],
    ...     }
    ... )
    >>> tbr_summary = pd.DataFrame({"sigma": [3.0], "t_dist_df": [20]})
    >>> validate_subinterval_parameters(
    ...     tbr_df, tbr_summary, start_day=1, end_day=2, ci_level=0.80
    ... )  # No error

    An out-of-range day raises ``ValueError``:

    >>> validate_subinterval_parameters(
    ...     tbr_df, tbr_summary, start_day=5, end_day=10, ci_level=0.80
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
    required_tbr_cols = ["y", "pred", "period", "estsd"]
    missing_tbr_cols = [col for col in required_tbr_cols if col not in tbr_df.columns]
    if missing_tbr_cols:
        raise ValueError(f"tbr_df missing required columns: {missing_tbr_cols}")

    required_summary_cols = ["sigma", "t_dist_df"]
    missing_summary_cols = [
        col for col in required_summary_cols if col not in tbr_summary.columns
    ]
    if missing_summary_cols:
        raise ValueError(
            f"tbr_summary missing required columns: {missing_summary_cols}"
        )

    # Validate test period data exists
    test_data = tbr_df[tbr_df["period"] == 1]
    if test_data.empty:
        raise ValueError("No test period data found (period == 1)")

    # Validate day parameters
    if not isinstance(start_day, int) or not isinstance(end_day, int):
        raise TypeError("start_day and end_day must be integers")

    if start_day < 1:
        raise ValueError(f"start_day must be >= 1, got: {start_day}")

    if end_day < start_day:
        raise ValueError(f"end_day ({end_day}) must be >= start_day ({start_day})")

    max_test_day = len(test_data)
    if end_day > max_test_day:
        raise ValueError(
            f"end_day ({end_day}) exceeds available test days ({max_test_day})"
        )

    # Validate credibility level
    if not isinstance(ci_level, (int, float)):
        raise TypeError("ci_level must be a number")

    if not (0 < ci_level < 1):
        raise ValueError(f"ci_level must be between 0 and 1, got: {ci_level}")
