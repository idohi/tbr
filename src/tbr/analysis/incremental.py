"""
TBR Incremental Analysis Module.

This module provides specialized functionality for incremental TBR analysis,
enabling day-by-day progression analysis of treatment effects during test periods.

The incremental analysis approach allows researchers to:
- Track treatment effect evolution over time
- Track when a declared posterior-probability or interval criterion is met
- Compare candidate test durations using explicitly chosen decision rules
- Understand effect stability and consistency patterns

Functions
---------
create_incremental_tbr_summaries : Create day-by-day incremental summaries

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
>>> print(f"Day 1 effect: {incremental.iloc[0]['estimate']:.2f}")
>>> print(f"Day 3 effect: {incremental.iloc[2]['estimate']:.2f}")

Analyze effect progression:

>>> for day in range(len(incremental)):
...     row = incremental.iloc[day]
...     print(f"Day {row['test_day']}: {row['estimate']:.2f} "
...           f"(prob={row['prob']:.3f})")

Find when posterior probability exceeds a declared threshold:

>>> significant_days = incremental[incremental['prob'] > 0.8]
>>> if not significant_days.empty:
...     first_sig_day = significant_days.iloc[0]['test_day']
...     print(f"Posterior probability exceeds 0.8 from day {first_sig_day}")
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
    Create one cumulative TBR summary per test day.

    Test rows are consumed in their input order. Row :math:`T` of the result
    summarizes test days 1 through :math:`T`; this helper does not apply a
    stopping rule or define statistical significance.

    Parameters
    ----------
    tbr_dataframe : pd.DataFrame
        Daily TBR output containing ``period``, ``cumdif``, and ``cumsd``.
        Rows satisfying ``period == 1`` must be in test-day order.
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
        or no row has ``period == 1``.

    Notes
    -----
    The returned DataFrame has these stable columns in order:

    ``test_day`` : integer
        One-based cumulative test-day number, normally ``int64``.
    ``estimate`` : ``float64``
        Cumulative-effect estimate :math:`\hat{\Delta}(T)`.
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
        Legacy summary name for input ``threshold``.
    ``prob`` : ``float64``
        Posterior probability
        :math:`P(\Delta(T)>\theta\mid\mathrm{data})`.
    ``alpha`` : ``float64``
        Regression intercept :math:`\beta_0`.
    ``beta`` : ``float64``
        Regression slope :math:`\beta_1`.
    ``alpha_beta_cov`` : ``float64``
        Legacy summary name for input ``cov_alpha_beta``.
    ``var_alpha`` : ``float64``
        Intercept-estimate variance.
    ``var_beta`` : ``float64``
        Slope-estimate variance.
    ``sigma`` : ``float64``
        Residual standard deviation.
    ``t_dist_df`` : ``float64``
        Summary-table representation of input ``degrees_freedom``.

    Each row in the returned DataFrame represents the cumulative effect
    from test day 1 through the specified test day. This allows analysis
    of how treatment effects accumulate and stabilize over time.

    The incremental analysis is particularly useful for:

    - Identifying the first observed day meeting a declared probability or
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

    where t represents the incremental test period (day 1, day 2, etc.).

    The posterior probability for each incremental period uses:

    .. math::
        P(effect_t > threshold) = 1 - F_t((threshold - estimate_t)/se_t, df)

    For mathematical background, see the
    :doc:`mathematical methodology guide </mathematical_methodology>`,
    especially the sections on cumulative treatment effects, posterior
    variance, and statistical inference.

    Examples
    --------
    Create incremental summaries for day-by-day analysis:

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
    >>> print(f"Day 1 effect: {incremental.iloc[0]['estimate']:.2f}")
    >>> print(f"Day 3 effect: {incremental.iloc[2]['estimate']:.2f}")

    Analyze effect progression:

    >>> for day in range(len(incremental)):
    ...     row = incremental.iloc[day]
    ...     print(f"Day {row['test_day']}: {row['estimate']:.2f} "
    ...           f"(prob={row['prob']:.3f})")

    Find when posterior probability exceeds a threshold:

    >>> high_prob_days = incremental[incremental['prob'] > 0.8]
    >>> if not high_prob_days.empty:
    ...     first_day = high_prob_days.iloc[0]['test_day']
    ...     print(f"Posterior probability exceeds 0.8 from day {first_day}")

    Track credible interval width over time:

    >>> interval_width = incremental['upper'] - incremental['lower']
    >>> print(f"Final credible interval width: {interval_width.iloc[-1]:.3f}")

    Apply a declared probability rule to candidate durations:

    >>> # Find the first observed day meeting the declared probability rule.
    >>> target_prob = 0.9
    >>> matching_days = incremental[incremental['prob'] >= target_prob]
    >>> if not matching_days.empty:
    ...     first_matching_day = matching_days.iloc[0]['test_day']
    ...     print(f"First observed day meeting the 90% probability rule: {first_matching_day}")
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
