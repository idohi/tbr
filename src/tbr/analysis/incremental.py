"""
TBR Incremental Analysis Module.

This module provides specialized functionality for incremental TBR analysis,
enabling test-row-by-test-row progression analysis of treatment effects.

The incremental analysis approach allows researchers to:
- Track treatment effect evolution over time
- Track when a declared posterior-probability or interval criterion is met
- Compare candidate test durations using explicitly chosen decision rules
- Understand effect stability and consistency patterns

Functions
---------
create_incremental_tbr_summaries : Create test-row-by-test-row summaries

Examples
--------
>>> import pandas as pd
>>> from tbr.analysis.incremental import create_incremental_tbr_summaries
>>> tbr_dataframe = pd.DataFrame(
...     {"period": [1, 1, 1], "cumdif": [5.0, 8.0, 11.0], "cumsd": [2.0, 3.0, 4.0]}
... )
>>> incremental = create_incremental_tbr_summaries(
...     tbr_dataframe, alpha=50.2, beta=0.95, sigma=25.3,
...     var_alpha=100.5, var_beta=0.001, cov_alpha_beta=-0.05,
...     degrees_freedom=43, level=0.80, threshold=0.0
... )
>>> print(f"Position 1 effect: {incremental.iloc[0]['estimate']:.2f}")
>>> print(f"Position 3 effect: {incremental.iloc[2]['estimate']:.2f}")

Analyze effect progression:

>>> for position in range(len(incremental)):
...     row = incremental.iloc[position]
...     print(f"Position {row['test_day']}: {row['estimate']:.2f} "
...           f"(prob={row['prob']:.3f})")

Find when posterior probability exceeds a declared threshold:

>>> rows_above_threshold = incremental[incremental['prob'] > 0.8]
>>> if not rows_above_threshold.empty:
...     first_position = rows_above_threshold.iloc[0]['test_day']
...     print(f"Posterior probability exceeds 0.8 from position {first_position}")
"""

import pandas as pd

# Import functional implementation for wrapping
from tbr.functional.tbr_functions import (
    create_incremental_tbr_summaries as functional_create_incremental_tbr_summaries,
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
    r"""
    Create one cumulative TBR summary per ordered test row.

    Test rows are consumed in their input order. Row :math:`T` of the result
    summarizes the first :math:`T` test rows or time points; this helper does
    not apply a stopping rule or define statistical significance.

    Parameters
    ----------
    tbr_dataframe : pd.DataFrame
        TBR output containing ``period``, ``cumdif``, and ``cumsd``. Rows
        satisfying ``period == 1`` must be in test-row/time-point order.
    alpha : float
        Regression intercept :math:`\beta_0`.
    beta : float
        Regression slope :math:`\beta_1`.
    sigma : float
        Positive residual standard deviation :math:`\sigma`.
    var_alpha : float
        Variance :math:`\mathbb{V}[\hat{\beta}_0]`.
    var_beta : float
        Variance :math:`\mathbb{V}[\hat{\beta}_1]`.
    cov_alpha_beta : float
        Covariance
        :math:`\operatorname{Cov}(\hat{\beta}_0,\hat{\beta}_1)`.
    degrees_freedom : int
        Positive regression degrees of freedom :math:`\nu`.
    level : float
        Credibility level in the closed interval ``[0, 1]``.
    threshold : float
        Effect threshold :math:`\theta` for the posterior probability.

    Returns
    -------
    pd.DataFrame
        Incremental summary ordered by ``test_day``. See Notes for the stable
        columns and their order.

    Raises
    ------
    ValueError
        If ``tbr_dataframe`` is empty or lacks a required column; ``level`` is
        outside ``[0, 1]``; ``degrees_freedom`` or ``sigma`` is not positive;
        no row has ``period == 1``; a consumed ``cumdif``, ``cumsd``, or
        ``threshold`` is non-finite; or a consumed ``cumsd`` is negative.

    Notes
    -----
    The returned DataFrame has these stable columns in order:

    ``test_day`` : integer
        Legacy name for the one-based cumulative test-row or time-point
        position, normally ``int64``.
    ``estimate`` : ``float64``
        Cumulative-effect estimate :math:`\hat{\Delta}(T)`, the running
        cumulative sum through the first :math:`T` entries of the ordered
        test-row/time-point sequence, with :math:`T` one-based.
    ``precision`` : ``float64``
        Credible-interval half-width.
    ``lower`` : ``float64``
        Lower credible bound.
    ``upper`` : ``float64``
        Upper credible bound.
    ``se`` : ``float64``
        ``cumsd`` Student-t scale used as the standard error of the cumulative
        effect estimate.
    ``level`` : ``float64``
        Credibility level.
    ``thres`` : ``float64``
        Legacy summary name mapped one way from input ``threshold``; stores the
        effect threshold :math:`\theta` used by the posterior exceedance
        probability.
    ``prob`` : ``float64``
        Posterior probability
        :math:`P(\Delta(T)>\theta\mid\mathrm{data})`.
    ``alpha`` : ``float64``
        Regression intercept :math:`\beta_0`.
    ``beta`` : ``float64``
        Regression slope :math:`\beta_1`.
    ``alpha_beta_cov`` : ``float64``
        Legacy summary name mapped one way from input ``cov_alpha_beta``;
        stores the covariance between the estimated regression intercept and
        slope.
    ``var_alpha`` : ``float64``
        Intercept-estimate variance.
    ``var_beta`` : ``float64``
        Slope-estimate variance.
    ``sigma`` : ``float64``
        Residual standard deviation.
    ``t_dist_df`` : ``float64``
        Legacy summary name mapped one way from input ``degrees_freedom``;
        stores the residual/pretest-regression Student-t degrees of freedom.

    Here :math:`T` is the one-based count of ordered test rows or time points,
    and :math:`\Delta(T)` is the running cumulative sum through the first
    :math:`T` test rows. The row with ``test_day == T`` reports that sum.

    The incremental analysis is particularly useful for:

    - Identifying the first observed test-row position meeting a probability or
      interval criterion
    - Understanding effect stability and consistency
    - Comparing candidate durations under an explicitly chosen criterion
    - Informing the design of future experiments

    Mathematical Foundation
    -----------------------
    The incremental analysis maintains the same mathematical rigor as the
    complete TBR analysis, with each incremental period using:

    .. math::
        CI_t = estimate_t ± t_{α/2,df} × se_t

    Here the subscript :math:`t` indexes the cumulative row, while the
    Student-t critical value is distribution notation. The equation's interval
    tail probability is alpha = 1 - level, distinct from the Python parameter
    ``alpha``, which is the regression intercept.

    The posterior probability for each incremental period uses:

    .. math::
        P(effect_t > threshold) = 1 - F_t((threshold - estimate_t)/se_t, df)

    In this expression, :math:`F_t` is the Student-t cumulative distribution
    function, not the ``test_day`` index.

    For mathematical background, see the
    :doc:`mathematical methodology guide </mathematical_methodology>`,
    especially the sections on cumulative treatment effects, posterior
    variance, and statistical inference.

    Examples
    --------
    Create incremental summaries for test-row-by-test-row analysis:

    >>> import pandas as pd
    >>> from tbr import create_incremental_tbr_summaries
    >>> tbr_dataframe = pd.DataFrame(
    ...     {"period": [1, 1, 1], "cumdif": [5.0, 8.0, 11.0], "cumsd": [2.0, 3.0, 4.0]}
    ... )
    >>> incremental = create_incremental_tbr_summaries(
    ...     tbr_dataframe, alpha=50.2, beta=0.95, sigma=25.3,
    ...     var_alpha=100.5, var_beta=0.001, cov_alpha_beta=-0.05,
    ...     degrees_freedom=43, level=0.80, threshold=0.0
    ... )
    >>> list(incremental["test_day"]) == [1, 2, 3]
    True
    >>> incremental["test_day"].dtype == "int64"
    True
    >>> incremental["estimate"].tolist() == [5.0, 8.0, 11.0]
    True
    >>> print(f"Position 1 effect: {incremental.iloc[0]['estimate']:.2f}")
    >>> print(f"Position 3 effect: {incremental.iloc[2]['estimate']:.2f}")

    Analyze effect progression:

    >>> for position in range(len(incremental)):
    ...     row = incremental.iloc[position]
    ...     print(f"Position {row['test_day']}: {row['estimate']:.2f} "
    ...           f"(prob={row['prob']:.3f})")

    Find when posterior probability exceeds a threshold:

    >>> high_prob_rows = incremental[incremental['prob'] > 0.8]
    >>> if not high_prob_rows.empty:
    ...     first_position = high_prob_rows.iloc[0]['test_day']
    ...     print(f"Posterior probability exceeds 0.8 from position {first_position}")

    Track credible interval width over time:

    >>> interval_width = incremental['upper'] - incremental['lower']
    >>> print(f"Final credible interval width: {interval_width.iloc[-1]:.3f}")

    Apply a declared probability rule to candidate durations:

    >>> # Find the first observed position meeting the probability rule.
    >>> target_prob = 0.9
    >>> matching_rows = incremental[incremental['prob'] >= target_prob]
    >>> if not matching_rows.empty:
    ...     first_position = matching_rows.iloc[0]['test_day']
    ...     print(f"First observed position meeting the 90% rule: {first_position}")
    """
    return functional_create_incremental_tbr_summaries(
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
