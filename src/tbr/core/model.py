"""
Object-oriented interface for Time-Based Regression analysis.

This module provides the TBRAnalysis class, which wraps the functional API
with a stateful interface for storing configuration, fitted parameters, and
analysis results.

Examples
--------
>>> from tbr.core.model import TBRAnalysis
>>> import pandas as pd
>>> import numpy as np
>>>
>>> # Create sample data
>>> data = pd.DataFrame({
...     'date': pd.date_range('2023-01-01', periods=90),
...     'control': np.random.normal(1000, 50, 90),
...     'test': np.random.normal(1020, 55, 90)
... })
>>>
>>> # Initialize and fit model
>>> model = TBRAnalysis(level=0.80, threshold=0.0)
>>> model.fit(
...     data=data,
...     time_col='date',
...     control_col='control',
...     test_col='test',
...     pretest_start='2023-01-01',
...     test_start='2023-02-15',
...     test_end='2023-03-01'
... )
>>>
>>> # Access results
>>> final_effect = model.summaries_.iloc[-1]['estimate']
>>> print(f"Treatment Effect: {final_effect:.2f}")

Notes
-----
Configuration parameters are stored in __init__. Analysis is performed via
the fit() method. Fitted results are accessed via underscore-suffixed
attributes (results_, summaries_, params_), which validate fitted state
before access.
"""

from typing import Any, Dict, Optional

import pandas as pd


class TBRAnalysis:
    """
    Time-Based Regression Analysis with stateful interface.

    Wraps the functional TBR API to store configuration, fitted parameters,
    and analysis results.

    Parameters
    ----------
    level : float, default=0.80
        Credibility level for confidence intervals (e.g., 0.80 for 80% CI).
        Must be between 0 and 1 exclusive.
    threshold : float, default=0.0
        Threshold for probability calculation. Typically 0.0 for testing
        positive effects. Can be any finite float value.
    test_end_inclusive : bool, default=False
        Whether to include the test_end boundary in the test period.

        - False (default): Exclusive end boundary (data < test_end)
        - True: Inclusive end boundary (data <= test_end)

    Attributes
    ----------
    level : float
        Credibility level for confidence intervals.
    threshold : float
        Threshold for probability calculation.
    test_end_inclusive : bool
        Whether test_end is inclusive.
    fitted_ : bool
        Whether the model has been fitted.
    results_ : pd.DataFrame
        TBR DataFrame with predictions, effects, and uncertainties.
        Available after calling fit().
    summaries_ : pd.DataFrame
        Incremental summaries with daily progression of effects.
        Available after calling fit().
    params_ : dict
        Regression model parameters (alpha, beta, sigma, variances, etc.).
        Available after calling fit().

    Examples
    --------
    Basic workflow:

    >>> model = TBRAnalysis(level=0.80, threshold=0.0)
    >>> model.fit(data, 'date', 'control', 'test',
    ...           pretest_start='2023-01-01',
    ...           test_start='2023-02-15',
    ...           test_end='2023-03-01')
    >>> print(model.summaries_.iloc[-1])

    Custom configuration:

    >>> model = TBRAnalysis(level=0.95, threshold=5.0, test_end_inclusive=True)
    >>> model.fit(data, 'date', 'control', 'test',
    ...           pretest_start='2023-01-01',
    ...           test_start='2023-02-15',
    ...           test_end='2023-02-15')  # Same-day analysis

    Notes
    -----
    Configuration parameters are stored in __init__. Call fit() to perform
    analysis. Access fitted results via underscore-suffixed attributes
    (results_, summaries_, params_), which validate fitted state before access.

    See Also
    --------
    tbr.functional.perform_tbr_analysis : Functional API for TBR analysis
    """

    def __init__(
        self,
        level: float = 0.80,
        threshold: float = 0.0,
        test_end_inclusive: bool = False,
    ) -> None:
        """
        Initialize TBR analysis with configuration parameters.

        Parameters
        ----------
        level : float, default=0.80
            Credibility level for confidence intervals.
        threshold : float, default=0.0
            Threshold for probability calculation.
        test_end_inclusive : bool, default=False
            Whether to include test_end boundary in analysis.

        Raises
        ------
        ValueError
            If level is not between 0 and 1 exclusive.
        TypeError
            If parameters have incorrect types.
        """
        # Validate configuration parameters
        if not isinstance(level, (int, float)):
            raise TypeError(f"level must be numeric, got {type(level).__name__}")

        if not (0 < level < 1):
            raise ValueError(f"level must be between 0 and 1 exclusive, got {level}")

        if not isinstance(threshold, (int, float)):
            raise TypeError(
                f"threshold must be numeric, got {type(threshold).__name__}"
            )

        if not isinstance(test_end_inclusive, bool):
            raise TypeError(
                f"test_end_inclusive must be bool, got {type(test_end_inclusive).__name__}"
            )

        # Store configuration
        self.level = float(level)
        self.threshold = float(threshold)
        self.test_end_inclusive = test_end_inclusive

        # Initialize state (will be set by fit())
        self._fitted = False
        self._results: Optional[pd.DataFrame] = None
        self._summaries: Optional[pd.DataFrame] = None
        self._params: Optional[Dict[str, Any]] = None
        self._fit_info: Optional[Dict[str, Any]] = None

    @property
    def fitted_(self) -> bool:
        """
        Whether the model has been fitted.

        Returns
        -------
        bool
            True if fit() has been called successfully, False otherwise.
        """
        return self._fitted

    @property
    def results_(self) -> pd.DataFrame:
        """
        TBR DataFrame with predictions, effects, and uncertainties.

        This DataFrame contains the complete time series with all TBR calculations:
        - Original data (time, control, test values)
        - Period indicators (pretest=0, test=1, cooldown=3)
        - Counterfactual predictions (pred, predsd)
        - Effects (dif, cumdif, cumsd, estsd)

        Returns
        -------
        pd.DataFrame
            Complete TBR analysis results DataFrame.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = TBRAnalysis()
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> results = model.results_
        >>> test_period_results = results[results['period'] == 1]
        """
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before accessing results_."
            )
        assert self._results is not None  # Guaranteed by _fitted check
        return self._results

    @property
    def summaries_(self) -> pd.DataFrame:
        """
        Incremental summaries with daily progression of cumulative effects.

        This DataFrame contains day-by-day summaries for the test period:
        - estimate: Cumulative treatment effect
        - precision: 1/variance of the estimate
        - lower, upper: Credible interval bounds
        - se: Standard error of the estimate
        - level: Credibility level used
        - threshold: Threshold used for probability calculation
        - prob: Posterior probability of exceeding threshold
        - Model parameters (alpha, beta, sigma, variances, covariances)

        Returns
        -------
        pd.DataFrame
            Incremental summaries for each day of the test period.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = TBRAnalysis()
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> summaries = model.summaries_
        >>> final_effect = summaries.iloc[-1]['estimate']
        >>> is_significant = summaries.iloc[-1]['lower'] > 0
        """
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before accessing summaries_."
            )
        assert self._summaries is not None  # Guaranteed by _fitted check
        return self._summaries

    @property
    def params_(self) -> Dict[str, Any]:
        """
        Regression model parameters from TBR analysis.

        Returns
        -------
        dict
            Dictionary containing regression parameters:
            - alpha: Intercept coefficient
            - beta: Slope coefficient
            - sigma: Residual standard error
            - var_alpha: Variance of alpha
            - var_beta: Variance of beta
            - cov_alpha_beta: Covariance between alpha and beta
            - degrees_freedom: Degrees of freedom for t-distribution
            - pretest_x_mean: Mean of control in pretest period
            - pretest_sum_x_squared_deviations: Sum of squared deviations

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = TBRAnalysis()
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> params = model.params_
        >>> print(f"Beta coefficient: {params['beta']:.4f}")
        >>> print(f"Residual std error: {params['sigma']:.4f}")
        """
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before accessing params_."
            )
        assert self._params is not None  # Guaranteed by _fitted check
        return self._params

    def __repr__(self) -> str:
        """
        Return string representation of TBRAnalysis instance.

        Returns
        -------
        str
            String representation showing configuration and fitted status.
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        return (
            f"TBRAnalysis(level={self.level}, threshold={self.threshold}, "
            f"test_end_inclusive={self.test_end_inclusive}, {fitted_str})"
        )

    def __str__(self) -> str:
        """
        Return user-friendly string representation.

        Returns
        -------
        str
            Human-readable string with key information.
        """
        if not self._fitted:
            return f"TBRAnalysis (not fitted)\n  level={self.level}\n  threshold={self.threshold}"

        assert self._summaries is not None  # Guaranteed by _fitted check
        n_test_days = len(self._summaries)
        final_effect = self._summaries.iloc[-1]["estimate"]
        final_lower = self._summaries.iloc[-1]["lower"]
        final_upper = self._summaries.iloc[-1]["upper"]

        return (
            f"TBRAnalysis (fitted)\n"
            f"  Configuration:\n"
            f"    level={self.level}\n"
            f"    threshold={self.threshold}\n"
            f"  Results:\n"
            f"    Test period days: {n_test_days}\n"
            f"    Final effect estimate: {final_effect:.2f}\n"
            f"    {int(self.level*100)}% CI: [{final_lower:.2f}, {final_upper:.2f}]"
        )
