"""
TBR Analysis Summary Module.

This module provides clean, modular interfaces for creating TBR summary statistics.
It delegates computation to the functional implementation while documenting the
public summary schema and its statistical interpretation.

Functions
---------
create_tbr_summary : Create single-row TBR summary with credible intervals

Examples
--------
>>> import pandas as pd
>>> from tbr.analysis.summary import create_tbr_summary
>>> tbr_dataframe = pd.DataFrame(
...     {"period": [1, 1], "cumdif": [5.0, 8.0], "cumsd": [2.0, 3.0]}
... )
>>> summary = create_tbr_summary(
...     tbr_dataframe, alpha=50, beta=0.95, sigma=25,
...     var_alpha=100, var_beta=0.001, cov_alpha_beta=-0.05,
...     degrees_freedom=43, level=0.80, threshold=0.0
... )
>>> print(f"Effect estimate: {summary['estimate'].iloc[0]:.2f}")
"""

import pandas as pd

# Import functional implementation for wrapping
from tbr.functional.tbr_functions import (
    create_tbr_summary as functional_create_tbr_summary,
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
    r"""
    Create a single-row TBR summary.

    The final test-period ``cumdif`` and ``cumsd`` values determine the
    cumulative-effect estimate and its standard error (Student-t scale). All
    returned statistics and model values are explicitly stored as ``float64``.

    Parameters
    ----------
    tbr_dataframe : pd.DataFrame
        Daily TBR output containing ``period``, ``cumdif``, and ``cumsd``.
        Rows satisfying ``period == 1`` must be in test-day order; the last such
        row supplies the final cumulative values.
    alpha : float
        Regression intercept :math:`\beta_0`. This is unrelated to interval
        tail probability.
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
        Credibility level in the closed interval ``[0, 1]``. Endpoint values
        can produce degenerate or unbounded intervals.
    threshold : float
        Effect threshold :math:`\theta` used for
        :math:`P(\Delta(T)>\theta\mid\mathrm{data})`.

    Returns
    -------
    pd.DataFrame
        Single-row summary. See Notes for the stable columns and their order.

    Raises
    ------
    ValueError
        If ``tbr_dataframe`` is empty or lacks a required column; ``level`` is
        outside ``[0, 1]``; ``degrees_freedom`` or ``sigma`` is not positive;
        or no row has ``period == 1``.

    Notes
    -----
    The returned DataFrame has these stable columns in order:

    ``estimate`` : ``float64``
        Final cumulative-effect estimate :math:`\hat{\Delta}(T)`.
    ``precision`` : ``float64``
        Credible-interval half-width.
    ``lower`` : ``float64``
        Lower credible bound.
    ``upper`` : ``float64``
        Upper credible bound.
    ``se`` : ``float64``
        Final ``cumsd`` value: Student-t scale used as the standard error of the
        cumulative-effect estimate.
    ``level`` : ``float64``
        Credibility level.
    ``thres`` : ``float64``
        Legacy summary name for input ``threshold``.
    ``prob`` : ``float64``
        Posterior threshold-exceedance probability
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

    The credible interval is calculated using the t-distribution:

    .. math::
        CI = estimate ± t_{α/2,df} × se

    where t_{α/2,df} is the critical value from t-distribution with specified
    degrees of freedom.

    The posterior probability uses the t-distribution CDF:

    .. math::
        P(effect > threshold) = 1 - F_t((threshold - estimate)/se, df)

    where F_t is the t-distribution cumulative distribution function.

    For mathematical background, see the
    :doc:`mathematical methodology guide </mathematical_methodology>`,
    especially the sections on treatment effect estimation and statistical
    inference.

    Examples
    --------
    Create summary for a TBR analysis output:

    >>> import pandas as pd
    >>> from tbr import create_tbr_summary
    >>> tbr_dataframe = pd.DataFrame(
    ...     # Each row is one cumulative test day; the final row defines the
    ...     # final summary estimate and standard error.
    ...     {"period": [1, 1], "cumdif": [5.0, 8.0], "cumsd": [2.0, 3.0]}
    ... )
    >>> summary = create_tbr_summary(
    ...     tbr_dataframe, alpha=50.2, beta=0.95, sigma=25.3,
    ...     var_alpha=100.5, var_beta=0.001, cov_alpha_beta=-0.05,
    ...     degrees_freedom=43, level=0.80, threshold=0.0
    ... )
    >>> list(summary.columns) == [
    ...     "estimate", "precision", "lower", "upper", "se", "level",
    ...     "thres", "prob", "alpha", "beta", "alpha_beta_cov",
    ...     "var_alpha", "var_beta", "sigma", "t_dist_df",
    ... ]
    True
    >>> set(summary.dtypes.astype(str)) == {"float64"}
    True
    >>> summary["estimate"].iloc[0] == 8.0
    True
    >>> print(f"Effect estimate: {summary['estimate'].iloc[0]:.2f}")
    >>> print(
    ...     f"80% credible interval: "
    ...     f"[{summary['lower'].iloc[0]:.2f}, {summary['upper'].iloc[0]:.2f}]"
    ... )
    >>> print(f"P(effect > 0): {summary['prob'].iloc[0]:.3f}")

    Access model parameters:

    >>> print(f"Regression slope: {summary['beta'].iloc[0]:.3f}")
    >>> print(f"Residual std: {summary['sigma'].iloc[0]:.2f}")
    """
    return functional_create_tbr_summary(
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
