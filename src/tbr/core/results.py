"""
Result object structures for TBR analysis outputs.

This module provides structured result containers for TBR analysis methods.
User input data is kept separate from computed outputs, eliminating the
possibility of column name conflicts.

Result objects provide typed attribute access, conversion and export methods,
and time-indexed pandas objects. ``TBRPredictionResult``,
``TBRSummaryResult``, and ``TBRSubintervalResult`` are frozen dataclasses;
``TBRResults`` is the comprehensive result returned by
``tbr.perform_tbr_analysis``. Only ``TBRSummaryResult`` provides
``to_dataframe()``.

Examples
--------
Basic usage with functional API:

>>> from tbr import perform_tbr_analysis
>>> import pandas as pd
>>> import numpy as np
>>>
>>> rng = np.random.default_rng(42)
>>> data = pd.DataFrame({
...     'date': pd.date_range('2023-01-01', periods=90),
...     'control': rng.normal(1000, 50, 90),
...     'test': rng.normal(1020, 55, 90)
... })
>>>
>>> # Run analysis - returns TBRResults object
>>> results = perform_tbr_analysis(
...     data, 'date', 'control', 'test',
...     pretest_start=pd.Timestamp('2023-01-01'),
...     test_start=pd.Timestamp('2023-02-15'),
...     test_end=pd.Timestamp('2023-03-01'),
...     level=0.80, threshold=0.0
... )
>>>
>>> # Access scalar summary
>>> print(f"Effect: {results.estimate:.2f}")
>>> print(f"Credible interval: [{results.conf_int_lower:.2f}, {results.conf_int_upper:.2f}]")
>>>
>>> # Access time series (all indexed by time)
>>> results.effects.head()
>>>
>>> # Get daily summary table
>>> summary_df = results.summary()
>>>
>>> # Get comprehensive TBR dataframe
>>> tbr_df = results.tbr_dataframe()

Object-oriented API:

>>> from tbr import TBRAnalysis
>>> model = TBRAnalysis(level=0.80, threshold=0.0)
>>> model.fit(
...     data=data,
...     time_col='date',
...     control_col='control',
...     test_col='test',
...     pretest_start=pd.Timestamp('2023-01-01'),
...     test_start=pd.Timestamp('2023-02-15'),
...     test_end=pd.Timestamp('2023-03-01'),
... )
>>>
>>> # Access prediction results
>>> predictions = model.predict()
>>> print(predictions.predictions.head())  # DataFrame
>>> print(f"Mean prediction: {predictions.predictions['pred'].mean():.2f}")
>>>
>>> # Access summary results
>>> summary = model.summarize()
>>> print(f"Effect: {summary.estimate:.2f}")
>>> print(f"Credible interval: [{summary.lower:.2f}, {summary.upper:.2f}]")
>>> print(f"Probability: {summary.prob:.3f}")
>>>
>>> # Access subinterval results
>>> week1 = model.analyze_subinterval(1, 7)
>>> print(f"Week 1 effect: {week1.estimate:.2f}")
>>> print(f"Interval half-width: {week1.se:.2f}")

Notes
-----
User input data is kept separate from computed outputs, eliminating any
possibility of column name conflicts. All time series outputs are pd.Series
indexed by the time column values (datetime64[ns], int64, or float64).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class TBRPredictionResult:
    r"""
    Result container for TBR counterfactual predictions.

    Contains predictions with uncertainty estimates and metadata about
    the model used to generate them.

    Attributes
    ----------
    predictions : pd.DataFrame
        Floating-point counterfactual prediction output. See Notes for the
        stable columns.
    n_predictions : int
        Number of rows in ``predictions`` and values in ``control_values``.
    model_params : dict
        Copy of the fitted parameter mapping used for prediction. See Notes
        for the stable keys.
    control_values : np.ndarray
        One-dimensional copy of the control values used for prediction.

    Notes
    -----
    The ``predictions`` DataFrame has these stable columns:

    ``pred`` : float64
        Counterfactual prediction :math:`\hat{y}_t^*`.
    ``predsd`` : float64
        Prediction standard deviation
        :math:`\sqrt{\mathbb{V}[y_t^*]}`, including coefficient and residual
        uncertainty.

    The ``model_params`` mapping has these stable keys:

    ``alpha`` : float
        Regression intercept :math:`\hat{\beta}_0`.
    ``beta`` : float
        Regression slope :math:`\hat{\beta}_1`.
    ``sigma`` : float
        Residual standard deviation :math:`\sigma`.
    ``var_alpha`` : float
        Variance of the intercept estimate.
    ``var_beta`` : float
        Variance of the slope estimate.
    ``cov_alpha_beta`` : float
        Intercept/slope covariance.
    ``degrees_freedom`` : int
        Student's :math:`t` degrees of freedom.
    ``pretest_x_mean`` : float
        Mean pretest control value.
    ``pretest_sum_x_squared_deviations`` : float
        Sum of squared pretest control deviations.

    ``@dataclass(frozen=True)`` provides shallow immutability: attribute
    rebinding is blocked, but the stored DataFrame, dictionary, and ndarray
    remain mutable. Results created by :meth:`tbr.TBRAnalysis.predict` receive
    copies of those inputs at construction. Attribute access and
    :meth:`to_dict` return the stored mutable objects by reference, so callers
    should copy them before mutation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import TBRAnalysis
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> model = TBRAnalysis(level=0.80)
    >>> model.fit(
    ...     data,
    ...     "date",
    ...     "control",
    ...     "test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ... )
    >>> result = model.predict()
    >>> print(result.predictions.head())
    >>> print(f"Generated {result.n_predictions} predictions")
    >>> print(f"Model alpha: {result.model_params['alpha']:.3f}")
    """

    predictions: pd.DataFrame
    n_predictions: int
    model_params: Dict[str, Union[float, int]]
    control_values: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.

        Returns
        -------
        dict
            Prediction-result mapping. See Notes for the stable schema.

        Notes
        -----
        The returned mapping has these stable keys:

        ``predictions`` : pd.DataFrame
            The ``pred`` and ``predsd`` floating-point DataFrame stored on the
            result. ``pred`` contains counterfactual predictions and ``predsd``
            contains prediction standard deviations including coefficient and
            residual uncertainty.
        ``n_predictions`` : int
            Number of predictions.
        ``model_params`` : dict
            Fitted parameter mapping containing:

            ``alpha`` : float
                Regression intercept.
            ``beta`` : float
                Regression slope.
            ``sigma`` : float
                Residual standard deviation.
            ``var_alpha`` : float
                Intercept-estimate variance.
            ``var_beta`` : float
                Slope-estimate variance.
            ``cov_alpha_beta`` : float
                Intercept/slope covariance.
            ``degrees_freedom`` : int
                Regression degrees of freedom.
            ``pretest_x_mean`` : float
                Mean pretest control value.
            ``pretest_sum_x_squared_deviations`` : float
                Sum of squared pretest control deviations.
        ``control_values`` : np.ndarray
            Prediction inputs.

        The contained mutable objects are returned by reference; this method
        does not deep-copy them. This class has no ``to_dataframe()`` method.
        """
        return {
            "predictions": self.predictions,
            "n_predictions": self.n_predictions,
            "model_params": self.model_params,
            "control_values": self.control_values,
        }

    def to_json(self, filepath: str, **kwargs: Any) -> None:
        """
        Export result to JSON file.

        Parameters
        ----------
        filepath : str
            Path to the output JSON file.
        **kwargs : Any
            Additional arguments passed to ``export_to_json()``.

        Examples
        --------
        >>> result.to_json('predictions.json')  # doctest: +SKIP
        """
        from tbr.utils.export import export_to_json

        export_to_json(self, filepath, **kwargs)

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """
        Export predictions DataFrame to CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        **kwargs : Any
            Additional arguments passed to ``export_to_csv()``.

        Examples
        --------
        >>> result.to_csv('predictions.csv', index=False)  # doctest: +SKIP
        """
        from tbr.utils.export import export_to_csv

        export_to_csv(self.predictions, filepath, **kwargs)

    @property
    def mean_pred(self) -> float:
        """
        Mean of predicted values.

        Returns
        -------
        float
            Average of counterfactual predictions
        """
        return float(self.predictions["pred"].mean())

    @property
    def mean_uncertainty(self) -> float:
        """
        Mean counterfactual prediction standard deviation.

        Returns
        -------
        float
            Average of the ``predsd`` values.
        """
        return float(self.predictions["predsd"].mean())

    def __repr__(self) -> str:
        """Generate string representation."""
        return (
            f"TBRPredictionResult(\n"
            f"  n_predictions={self.n_predictions},\n"
            f"  mean_pred={self.mean_pred:.3f},\n"
            f"  mean_uncertainty={self.mean_uncertainty:.3f}\n"
            f")"
        )


@dataclass(frozen=True)
class TBRSummaryResult:
    r"""
    Result container for TBR summary statistics.

    Contains comprehensive summary statistics for TBR analysis including
    effect estimates, credible intervals, and model parameters.

    Attributes
    ----------
    estimate : float
        Cumulative treatment effect estimate
    lower : float
        Lower bound of credible interval
    upper : float
        Upper bound of credible interval
    se : float
        Standard error of the estimate
    prob : float
        Posterior probability of exceeding threshold
    precision : float
        Precision (half-width of credible interval)
    level : float
        Credibility level used
    threshold : float
        Threshold used for probability calculation
    alpha : float
        Regression intercept coefficient
    beta : float
        Regression slope coefficient
    sigma : float
        Residual standard deviation
    var_alpha : float
        Variance of intercept estimate
    var_beta : float
        Variance of slope estimate
    cov_alpha_beta : float
        Covariance between intercept and slope
    degrees_freedom : int
        Student's :math:`t` degrees of freedom.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import TBRAnalysis
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> model = TBRAnalysis(level=0.80)
    >>> model.fit(
    ...     data,
    ...     "date",
    ...     "control",
    ...     "test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ... )
    >>> result = model.summarize()
    >>> print(f"Effect: {result.estimate:.2f}")
    >>> print(f"80% credible interval: [{result.lower:.2f}, {result.upper:.2f}]")
    >>> print(f"Posterior prob > threshold: {result.prob:.3f}")
    """

    estimate: float
    lower: float
    upper: float
    se: float
    prob: float
    precision: float
    level: float
    threshold: float
    alpha: float
    beta: float
    sigma: float
    var_alpha: float
    var_beta: float
    cov_alpha_beta: float
    degrees_freedom: int

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """
        Convert result to dictionary format.

        Returns
        -------
        dict
            Summary-result mapping. See Notes for the stable schema.

        Notes
        -----
        The returned mapping has these stable keys:

        ``estimate`` : float
            Cumulative-effect estimate.
        ``lower`` : float
            Lower credible bound.
        ``upper`` : float
            Upper credible bound.
        ``se`` : float
            Standard error of the estimate.
        ``prob`` : float
            Posterior threshold-exceedance probability.
        ``precision`` : float
            Precision (half-width of credible interval).
        ``level`` : float
            Credibility level.
        ``threshold`` : float
            Effect threshold.
        ``alpha`` : float
            Regression intercept.
        ``beta`` : float
            Regression slope.
        ``sigma`` : float
            Residual standard deviation.
        ``var_alpha`` : float
            Intercept-estimate variance.
        ``var_beta`` : float
            Slope-estimate variance.
        ``cov_alpha_beta`` : float
            Intercept/slope covariance.
        ``degrees_freedom`` : int
            Student's :math:`t` degrees of freedom.
        """
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "se": self.se,
            "prob": self.prob,
            "precision": self.precision,
            "level": self.level,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "beta": self.beta,
            "sigma": self.sigma,
            "var_alpha": self.var_alpha,
            "var_beta": self.var_beta,
            "cov_alpha_beta": self.cov_alpha_beta,
            "degrees_freedom": self.degrees_freedom,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert result to single-row DataFrame format.

        Returns
        -------
        pd.DataFrame
            New single-row DataFrame. See Notes for the stable columns.

        Notes
        -----
        For results produced by :class:`tbr.TBRAnalysis`, the stable columns
        are:

        ``estimate`` : float64
            Cumulative-effect estimate.
        ``lower`` : float64
            Lower credible bound.
        ``upper`` : float64
            Upper credible bound.
        ``se`` : float64
            Standard error of the estimate.
        ``prob`` : float64
            Posterior threshold-exceedance probability.
        ``precision`` : float64
            Precision (half-width of credible interval).
        ``level`` : float64
            Credibility level.
        ``threshold`` : float64
            Effect threshold.
        ``alpha`` : float64
            Regression intercept.
        ``beta`` : float64
            Regression slope.
        ``sigma`` : float64
            Residual standard deviation.
        ``var_alpha`` : float64
            Intercept-estimate variance.
        ``var_beta`` : float64
            Slope-estimate variance.
        ``cov_alpha_beta`` : float64
            Intercept/slope covariance.
        ``degrees_freedom`` : int64
            Student's :math:`t` degrees of freedom.
        """
        return pd.DataFrame([self.to_dict()])

    def to_json(self, filepath: str, **kwargs: Any) -> None:
        """
        Export result to JSON file.

        Parameters
        ----------
        filepath : str
            Path to the output JSON file.
        **kwargs : Any
            Additional arguments passed to ``export_to_json()``.

        Examples
        --------
        >>> summary.to_json('summary.json')  # doctest: +SKIP
        """
        from tbr.utils.export import export_to_json

        export_to_json(self, filepath, **kwargs)

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """
        Export result to CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        **kwargs : Any
            Additional arguments passed to ``export_to_csv()``.

        Examples
        --------
        >>> summary.to_csv('summary.csv', index=False)  # doctest: +SKIP
        """
        from tbr.utils.export import export_to_csv

        export_to_csv(self, filepath, **kwargs)

    def is_significant(self, probability_threshold: float = 0.95) -> bool:
        """
        Check whether the stored posterior probability meets a cutoff.

        This convenience test compares ``prob`` with a posterior-probability
        cutoff. It does not test whether a credible interval excludes zero.

        Parameters
        ----------
        probability_threshold : float, default=0.95
            Cutoff compared directly with ``prob``. The method performs no
            range validation.

        Returns
        -------
        bool
            True if posterior probability is at least ``probability_threshold``.
        """
        return self.prob >= probability_threshold

    def __repr__(self) -> str:
        """Generate string representation."""
        return (
            f"TBRSummaryResult(\n"
            f"  estimate={self.estimate:.3f},\n"
            f"  CI=[{self.lower:.3f}, {self.upper:.3f}] (level={self.level}),\n"
            f"  se={self.se:.3f},\n"
            f"  prob={self.prob:.3f}\n"
            f")"
        )


@dataclass(frozen=True)
class TBRSubintervalResult:
    """
    Result container for TBR subinterval analysis.

    Contains treatment effect estimates for a specific time window within
    the test period, with credible intervals and metadata.

    Attributes
    ----------
    estimate : float
        Treatment effect estimate for the subinterval
    lower : float
        Lower bound of credible interval
    upper : float
        Upper bound of credible interval
    se : float
        Credible-interval half-width stored under the legacy ``se`` field name.
    ci_level : float
        Credibility level used for interval
    start_day : int
        Starting day of subinterval (1-indexed)
    end_day : int
        Ending day of subinterval (1-indexed)
    n_days : int
        Inclusive subinterval length, ``end_day - start_day + 1``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import TBRAnalysis
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(1000, 50, size=44)
    >>> data = pd.DataFrame(
    ...     {
    ...         "date": pd.date_range("2023-01-01", periods=44),
    ...         "control": control,
    ...         "test": 1.05 * control + rng.normal(0, 10, size=44),
    ...     }
    ... )
    >>> model = TBRAnalysis(level=0.80)
    >>> model.fit(
    ...     data,
    ...     "date",
    ...     "control",
    ...     "test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ... )
    >>> result = model.analyze_subinterval(1, 7)
    >>> print(f"Week 1 effect: {result.estimate:.2f}")
    >>> print(f"Credible interval: [{result.lower:.2f}, {result.upper:.2f}]")
    >>> print(f"Days: {result.start_day}-{result.end_day} ({result.n_days} days)")
    >>> if result.contains_zero():
    ...     print("Credible interval includes zero")
    """

    estimate: float
    lower: float
    upper: float
    se: float
    ci_level: float
    start_day: int
    end_day: int
    n_days: int

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """
        Convert result to dictionary format.

        Returns
        -------
        dict
            Subinterval-result mapping. See Notes for the stable schema.

        Notes
        -----
        The returned mapping has these stable keys:

        ``estimate`` : float
            Cumulative effect over the subinterval.
        ``lower`` : float
            Lower credible bound.
        ``upper`` : float
            Upper credible bound.
        ``se`` : float
            Credible-interval half-width stored under the legacy field name.
        ``ci_level`` : float
            Credibility level.
        ``start_day`` : int
            One-based inclusive start day.
        ``end_day`` : int
            One-based inclusive end day.
        ``n_days`` : int
            Inclusive subinterval length.
        """
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "se": self.se,
            "ci_level": self.ci_level,
            "start_day": self.start_day,
            "end_day": self.end_day,
            "n_days": self.n_days,
        }

    def to_json(self, filepath: str, **kwargs: Any) -> None:
        """
        Export result to JSON file.

        Parameters
        ----------
        filepath : str
            Path to the output JSON file.
        **kwargs : Any
            Additional arguments passed to ``export_to_json()``.

        Examples
        --------
        >>> result.to_json('week1_results.json')  # doctest: +SKIP
        """
        from tbr.utils.export import export_to_json

        export_to_json(self, filepath, **kwargs)

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """
        Export result to CSV file as single-row DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        **kwargs : Any
            Additional arguments passed to ``pandas.DataFrame.to_csv()``.

        Examples
        --------
        >>> result.to_csv('week1_results.csv', index=False)  # doctest: +SKIP
        """
        df = pd.DataFrame([self.to_dict()])
        df.to_csv(filepath, **kwargs)

    def contains_zero(self) -> bool:
        """
        Check if credible interval contains zero.

        Returns
        -------
        bool
            ``True`` when ``lower <= 0 <= upper``.
        """
        return self.lower <= 0 <= self.upper

    def is_positive(self) -> bool:
        """
        Check if entire credible interval is positive.

        Returns
        -------
        bool
            ``True`` when ``lower > 0``.
        """
        return self.lower > 0

    def is_negative(self) -> bool:
        """
        Check if entire credible interval is negative.

        Returns
        -------
        bool
            ``True`` when ``upper < 0``.
        """
        return self.upper < 0

    def __repr__(self) -> str:
        """Generate string representation."""
        return (
            f"TBRSubintervalResult(\n"
            f"  days={self.start_day}-{self.end_day} (n={self.n_days}),\n"
            f"  estimate={self.estimate:.3f},\n"
            f"  CI=[{self.lower:.3f}, {self.upper:.3f}] (level={self.ci_level}),\n"
            f"  se={self.se:.3f}\n"
            f")"
        )


class TBRResults:
    """
    Comprehensive results from Time-Based Regression analysis.

    All computed outputs are accessible via properties and methods, with
    complete separation from user input data. This design eliminates any
    possibility of column name conflicts.

    All time series outputs are returned as pandas Series indexed by the time
    column values from the input data, preserving temporal alignment for
    plotting, merging, and further analysis.

    Attributes
    ----------
    control : pd.Series
        Control group values indexed by time.
    test : pd.Series
        Test group values indexed by time.
    fittedvalues : pd.Series
        Fitted values from the pretest-period regression.
    predictions : pd.Series
        Counterfactual predictions for the test period.
    resid : pd.Series
        Pretest residuals, defined as observed minus fitted values.
    effects : pd.Series
        Test-period treatment effects, defined as observed minus predicted values.
    cumulative_effect : pd.Series
        Cumulative treatment effects over the test period.
    prediction_se : pd.Series
        Counterfactual prediction standard deviations for the test period.
    cumulative_se : pd.Series
        Cumulative effect standard errors (test period).
    estimate : float
        Final cumulative treatment effect.
    conf_int_lower : float
        Lower bound of the credible interval for the final estimate.
    conf_int_upper : float
        Upper bound of the credible interval for the final estimate.
    pvalue : float
        Legacy property name containing the posterior probability that the
        cumulative effect exceeds the configured threshold; not a frequentist
        p-value.
    n_pretest : int
        Number of pretest observations.
    n_test : int
        Number of test observations.
    n_test_days : int
        Number of days in the test period.
    alpha : float
        Regression intercept.
    beta : float
        Regression slope.
    sigma : float
        Residual standard deviation.
    model_params : dict
        Copy of the model-parameter mapping. Keys are ``alpha : float``,
        ``beta : float``, ``sigma : float``, ``var_alpha : float``,
        ``var_beta : float``, ``cov_alpha_beta : float``,
        ``degrees_freedom : int``, ``n_pretest : int``, and
        ``pretest_x_mean : float``.

    Notes
    -----
    Instances are returned by :func:`~tbr.perform_tbr_analysis`; users normally
    do not instantiate this class directly.

    Time-series properties return their stored pandas Series by reference;
    callers must use ``.copy()`` before mutation. In contrast, ``summary()``,
    ``tbr_dataframe()``, ``conf_int()``, and ``model_params`` return new
    DataFrame or dictionary objects.

    The ``summary()``, ``tbr_dataframe()``, and ``conf_int()`` methods provide
    tabular exports of the fitted analysis results and are documented below.
    Input data and computed outputs are kept separate, with all results
    accessible via properties. Time alignment is preserved through pandas
    Series indexing, working seamlessly with datetime, integer, or float
    time columns.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tbr import perform_tbr_analysis
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
    ...     data,
    ...     "date",
    ...     "control",
    ...     "test",
    ...     pretest_start=pd.Timestamp("2023-01-01"),
    ...     test_start=pd.Timestamp("2023-01-31"),
    ...     test_end=pd.Timestamp("2023-02-14"),
    ...     level=0.80,
    ...     threshold=0.0,
    ... )
    >>>
    >>> # Access scalar summary
    >>> print(f"Effect: {results.estimate:.2f}")
    >>> print(
    ...     f"80% credible interval: "
    ...     f"[{results.conf_int_lower:.2f}, {results.conf_int_upper:.2f}]"
    ... )
    >>> print(f"Posterior prob > threshold: {results.pvalue:.3f}")
    >>>
    >>> # Access time series (all indexed by time)
    >>> results.effects.describe()
    >>> results.cumulative_effect.plot(title='Cumulative Treatment Effect')  # doctest: +SKIP
    >>>
    >>> # Get daily summary table
    >>> daily_summary = results.summary()
    >>> print(daily_summary.tail())
    >>>
    >>> # Get comprehensive TBR dataframe
    >>> tbr_df = results.tbr_dataframe()

    See Also
    --------
    perform_tbr_analysis : Functional API that returns ``TBRResults``.
    TBRAnalysis : Object-oriented API wrapper.
    """

    def __init__(
        self,
        _data: pd.DataFrame,
        time_col: str,
        control_col: str,
        test_col: str,
        model_params: Dict[str, Union[float, int]],
        periods: Dict[str, pd.DataFrame],
        level: float,
        threshold: float,
    ):
        """Initialize TBRResults with analysis outputs."""
        # Store metadata
        self._time_col = time_col
        self._control_col = control_col
        self._test_col = test_col
        self._model_params = model_params
        self._level = level
        self._threshold = threshold

        # Store period data
        self._baseline_data = periods.get("baseline", pd.DataFrame())
        self._pretest_data = periods["pretest"]
        self._test_data = periods["test"]
        self._cooldown_data = periods.get("cooldown", pd.DataFrame())

        # Compute all results during initialization
        self._compute_results()

    def _compute_results(self) -> None:
        """
        Compute all TBR analysis results.

        This internal method performs the complete TBR analysis pipeline,
        computing fitted values, predictions, effects, and uncertainties
        for all periods. Results are stored as internal Series indexed by time.
        """
        # Import here to avoid circular imports
        from tbr.functional.tbr_functions import (
            calculate_cumulative_standard_deviation,
            calculate_model_variance,
            calculate_sum_x_squared_deviations,
            create_incremental_tbr_summaries,
            generate_counterfactual_predictions,
            safe_int_conversion,
        )

        # Extract time indices for each period
        self._baseline_time = (
            self._baseline_data[self._time_col].values
            if not self._baseline_data.empty
            else np.array([])
        )
        self._pretest_time = self._pretest_data[self._time_col].values
        self._test_time = self._test_data[self._time_col].values
        self._all_time = np.concatenate([self._pretest_time, self._test_time])

        # Extract control and test values
        pretest_control = self._pretest_data[self._control_col].values
        pretest_test = self._pretest_data[self._test_col].values
        test_control = self._test_data[self._control_col].values
        test_test = self._test_data[self._test_col].values

        # Store input data as Series (indexed by time)
        self._control_series = pd.Series(
            np.concatenate([pretest_control, test_control]),
            index=self._all_time,
            name="control",
        )
        self._test_series = pd.Series(
            np.concatenate([pretest_test, test_test]),
            index=self._all_time,
            name="test",
        )

        # Compute pretest statistics
        pretest_x_mean = float(np.mean(pretest_control))
        pretest_sum_x_squared_deviations = calculate_sum_x_squared_deviations(
            pretest_control
        )
        n_pretest = safe_int_conversion(self._model_params["n_pretest"], "n_pretest")

        # Compute fitted values for pretest period
        alpha = cast(float, self._model_params["alpha"])
        beta = cast(float, self._model_params["beta"])
        fitted_vals = alpha + beta * pretest_control
        self._fittedvalues = pd.Series(
            fitted_vals, index=self._pretest_time, name="fittedvalues"
        )

        # Compute fitted value uncertainties (model variance only)
        fitted_variances = calculate_model_variance(
            x_values=pretest_control,
            pretest_x_mean=pretest_x_mean,
            sigma=self._model_params["sigma"],
            n_pretest=n_pretest,
            pretest_sum_x_squared_deviations=pretest_sum_x_squared_deviations,
        )
        self._fitted_se = pd.Series(
            np.sqrt(fitted_variances), index=self._pretest_time, name="fitted_se"
        )

        # Compute residuals for pretest period
        self._resid = pd.Series(
            pretest_test - fitted_vals, index=self._pretest_time, name="resid"
        )

        # Generate counterfactual predictions for test period
        test_predictions = generate_counterfactual_predictions(
            alpha=self._model_params["alpha"],
            beta=self._model_params["beta"],
            sigma=self._model_params["sigma"],
            pretest_x_mean=pretest_x_mean,
            n_pretest=n_pretest,
            pretest_sum_x_squared_deviations=pretest_sum_x_squared_deviations,
            test_period_data=self._test_data,
            control_col=self._control_col,
            time_col=self._time_col,
        )

        # Store predictions and uncertainties
        self._predictions = pd.Series(
            test_predictions["pred"].values, index=self._test_time, name="predictions"
        )
        self._prediction_se = pd.Series(
            test_predictions["predsd"].values,
            index=self._test_time,
            name="prediction_se",
        )

        # Compute treatment effects
        self._effects = pd.Series(
            test_test - test_predictions["pred"].values,
            index=self._test_time,
            name="effects",
        )

        # Compute cumulative effects
        self._cumulative_effect = pd.Series(
            self._effects.cumsum().values,
            index=self._test_time,
            name="cumulative_effect",
        )

        # Compute cumulative standard deviations
        cumsd_values = calculate_cumulative_standard_deviation(
            test_control,
            self._model_params["sigma"],
            self._model_params["var_alpha"],
            self._model_params["var_beta"],
            self._model_params["cov_alpha_beta"],
        )
        self._cumulative_se = pd.Series(
            cumsd_values, index=self._test_time, name="cumulative_se"
        )

        # Create comprehensive DataFrame for summary computation
        # This is internal and uses standard column names
        self._internal_df = self._build_internal_dataframe()

        # Compute incremental daily summaries
        self._summary_df = create_incremental_tbr_summaries(
            tbr_dataframe=self._internal_df,
            alpha=self._model_params["alpha"],
            beta=self._model_params["beta"],
            sigma=self._model_params["sigma"],
            var_alpha=self._model_params["var_alpha"],
            var_beta=self._model_params["var_beta"],
            cov_alpha_beta=self._model_params["cov_alpha_beta"],
            degrees_freedom=safe_int_conversion(
                self._model_params["degrees_freedom"], "degrees_freedom"
            ),
            level=self._level,
            threshold=self._threshold,
        )

    def _build_internal_dataframe(self) -> pd.DataFrame:
        """
        Build internal DataFrame for computations.

        This creates a DataFrame with standard column names for internal use.
        Not exposed to users - they use property-based access instead.
        """
        dataframes_to_combine = []

        # Process baseline period if it exists
        if not self._baseline_data.empty:
            baseline_df = self._baseline_data.copy()
            baseline_df["period"] = -1
            baseline_df["y"] = baseline_df[self._test_col]
            baseline_df["x"] = baseline_df[self._control_col]
            baseline_df["pred"] = np.nan
            baseline_df["predsd"] = np.nan
            baseline_df["dif"] = np.nan
            baseline_df["cumdif"] = np.nan
            baseline_df["cumsd"] = np.nan
            baseline_df["estsd"] = np.nan
            dataframes_to_combine.append(baseline_df)

        # Process pretest period
        pretest_df = self._pretest_data.copy()
        pretest_df["period"] = 0
        pretest_df["y"] = pretest_df[self._test_col]
        pretest_df["x"] = pretest_df[self._control_col]
        pretest_df["pred"] = self._fittedvalues.values
        pretest_df["predsd"] = 0.0
        pretest_df["dif"] = self._resid.values
        pretest_df["cumdif"] = np.nan
        pretest_df["cumsd"] = 0.0
        pretest_df["estsd"] = self._fitted_se.values
        dataframes_to_combine.append(pretest_df)

        # Process test period
        test_df = self._test_data.copy()
        test_df["period"] = 1
        test_df["y"] = test_df[self._test_col]
        test_df["x"] = test_df[self._control_col]
        test_df["pred"] = self._predictions.values
        test_df["predsd"] = self._prediction_se.values
        test_df["dif"] = self._effects.values
        test_df["cumdif"] = self._cumulative_effect.values
        test_df["cumsd"] = self._cumulative_se.values
        test_df["estsd"] = np.nan
        dataframes_to_combine.append(test_df)

        return pd.concat(dataframes_to_combine, ignore_index=True)

    # =========================================================================
    # Properties: Time Series (all indexed by time)
    # =========================================================================

    @property
    def control(self) -> pd.Series:
        """
        Control group values indexed by time.

        Returns
        -------
        pd.Series
            Control group metric values for pretest and test periods,
            with value dtype inherited from the source control column. The
            index is derived from the configured input time-column values and
            retains their datetime, integer, or floating dtype where pandas
            permits. The stored Series is returned by reference; call
            ``.copy()`` before mutation.
        """
        return self._control_series

    @property
    def test(self) -> pd.Series:
        """
        Test group values indexed by time.

        Returns
        -------
        pd.Series
            Test group metric values for pretest and test periods,
            with value dtype inherited from the source treatment/test column.
            The index is derived from the configured input time-column values
            and retains their datetime, integer, or floating dtype where
            pandas permits. The stored Series is returned by reference; call
            ``.copy()`` before mutation.
        """
        return self._test_series

    @property
    def fittedvalues(self) -> pd.Series:
        r"""
        Fitted values from regression model (pretest period only).

        Returns
        -------
        pd.Series
            Floating-point values
            :math:`\hat{y}_t=\hat{\beta}_0+\hat{\beta}_1x_t` for the pretest
            period. The index is derived from pretest input time-column values
            and retains their dtype where pandas permits. The stored Series is
            returned by reference; call ``.copy()`` before mutation.
        """
        return self._fittedvalues

    @property
    def predictions(self) -> pd.Series:
        r"""
        Counterfactual predictions for test period.

        Returns
        -------
        pd.Series
            Floating-point counterfactual predictions
            :math:`\hat{y}_t^*` for the test period. The index is derived from
            test input time-column values and retains their dtype where pandas
            permits. The stored Series is returned by reference; call
            ``.copy()`` before mutation.
        """
        return self._predictions

    @property
    def resid(self) -> pd.Series:
        """
        Residuals for pretest period (observed - fitted).

        Returns
        -------
        pd.Series
            Floating-point residuals from the regression fit for the pretest
            period. The index is derived from pretest input time-column values
            and retains their dtype where pandas permits. The stored Series is
            returned by reference; call ``.copy()`` before mutation.
        """
        return self._resid

    @property
    def effects(self) -> pd.Series:
        r"""
        Treatment effects for test period (observed - predicted).

        Returns
        -------
        pd.Series
            Floating-point pointwise effects
            :math:`\phi_t=y_t-\hat{y}_t^*` for the test period. The index is
            derived from test input time-column values and retains their dtype
            where pandas permits. The stored Series is returned by reference;
            call ``.copy()`` before mutation.
        """
        return self._effects

    @property
    def cumulative_effect(self) -> pd.Series:
        r"""
        Cumulative treatment effects over test period.

        Returns
        -------
        pd.Series
            Floating-point running cumulative effects :math:`\Delta(T)` for
            the test period. The index is derived from test input time-column
            values and retains their dtype where pandas permits. The stored
            Series is returned by reference; call ``.copy()`` before mutation.
        """
        return self._cumulative_effect

    @property
    def prediction_se(self) -> pd.Series:
        r"""
        Counterfactual prediction standard deviations for the test period.

        Returns
        -------
        pd.Series
            Floating-point values of
            :math:`\sqrt{\mathbb{V}[y_t^*]}` including coefficient and
            residual uncertainty for the test period. The index is derived
            from test input time-column values and retains their dtype where
            pandas permits. The stored Series is returned by reference; call
            ``.copy()`` before mutation.
        """
        return self._prediction_se

    @property
    def cumulative_se(self) -> pd.Series:
        r"""
        Cumulative effect standard errors over test period.

        Returns
        -------
        pd.Series
            Standard errors of cumulative effects, indexed by test time. The
            index retains its dtype where pandas permits. The stored Series is
            returned by reference; call ``.copy()`` before mutation.
        """
        return self._cumulative_se

    # =========================================================================
    # Properties: Scalars
    # =========================================================================

    @property
    def estimate(self) -> float:
        """
        Final cumulative treatment effect estimate.

        Returns
        -------
        float
            Total cumulative effect at end of test period.
        """
        return float(self._cumulative_effect.iloc[-1])

    @property
    def conf_int_lower(self) -> float:
        """
        Lower bound of credible interval for final estimate.

        Returns
        -------
        float
            Lower credible bound at the configured level.
        """
        return float(self._summary_df.iloc[-1]["lower"])

    @property
    def conf_int_upper(self) -> float:
        """
        Upper bound of credible interval for final estimate.

        Returns
        -------
        float
            Upper credible bound at the configured level.
        """
        return float(self._summary_df.iloc[-1]["upper"])

    @property
    def pvalue(self) -> float:
        r"""
        Posterior probability of effect exceeding the configured threshold.

        ``pvalue`` is a legacy property name; the value is not a frequentist
        p-value.

        Returns
        -------
        float
            :math:`P(\Delta(T) > \theta\mid\mathrm{data})`.
        """
        return float(self._summary_df.iloc[-1]["prob"])

    @property
    def n_pretest(self) -> int:
        """
        Number of observations in pretest period.

        Returns
        -------
        int
            Count of pretest observations.
        """
        return len(self._pretest_data)

    @property
    def n_test(self) -> int:
        """
        Number of observations in test period.

        Returns
        -------
        int
            Count of test observations.
        """
        return len(self._test_data)

    @property
    def n_test_days(self) -> int:
        """
        Number of days (time points) in test period.

        Returns
        -------
        int
            Length of test period.
        """
        return len(self._test_data)

    # =========================================================================
    # Properties: Model Parameters
    # =========================================================================

    @property
    def alpha(self) -> float:
        """
        Regression intercept coefficient.

        Returns
        -------
        float
            Intercept from regression: test = alpha + beta * control.
        """
        return float(self._model_params["alpha"])

    @property
    def beta(self) -> float:
        """
        Regression slope coefficient.

        Returns
        -------
        float
            Slope from regression: test = alpha + beta * control.
        """
        return float(self._model_params["beta"])

    @property
    def sigma(self) -> float:
        """
        Residual standard deviation from regression.

        Returns
        -------
        float
            Standard deviation of residuals from pretest fit.
        """
        return float(self._model_params["sigma"])

    @property
    def model_params(self) -> Dict[str, Union[float, int]]:
        """
        Complete dictionary of regression model parameters.

        Returns
        -------
        dict
            Copy of the model-parameter mapping. See Notes for the stable
            schema.

        Notes
        -----
        The returned mapping has these stable keys:

        ``alpha`` : float
            Regression intercept.
        ``beta`` : float
            Regression slope.
        ``sigma`` : float
            Residual standard deviation.
        ``var_alpha`` : float
            Intercept-estimate variance.
        ``var_beta`` : float
            Slope-estimate variance.
        ``cov_alpha_beta`` : float
            Intercept/slope covariance.
        ``degrees_freedom`` : int
            Student's :math:`t` degrees of freedom.
        ``n_pretest`` : int
            Number of pretest observations.
        ``pretest_x_mean`` : float
            Mean pretest control value.
        """
        return self._model_params.copy()

    # =========================================================================
    # Methods
    # =========================================================================

    def summary(self) -> pd.DataFrame:
        """
        Generate daily incremental summary statistics.

        Creates a DataFrame with one row per test day, showing cumulative
        effects and credible intervals as they accumulate over time.

        Returns
        -------
        pd.DataFrame
            Copy with one row per cumulative test day. See Notes for the stable
            columns.

        Notes
        -----
        The returned DataFrame has these stable columns:

        ``test_day`` : int64
            One-based cumulative test-day number.
        ``estimate`` : float64
            Cumulative-effect estimate.
        ``precision`` : float64
            Precision (half-width of credible interval).
        ``lower`` : float64
            Lower credible bound.
        ``upper`` : float64
            Upper credible bound.
        ``se`` : float64
            Standard error of the estimate.
        ``level`` : float64
            Credibility level.
        ``thres`` : float64
            Legacy column name for ``threshold``.
        ``prob`` : float64
            Posterior threshold-exceedance probability.
        ``alpha`` : float64
            Regression intercept.
        ``beta`` : float64
            Regression slope.
        ``alpha_beta_cov`` : float64
            Legacy column name for ``cov_alpha_beta``.
        ``var_alpha`` : float64
            Intercept-estimate variance.
        ``var_beta`` : float64
            Slope-estimate variance.
        ``sigma`` : float64
            Residual standard deviation.
        ``t_dist_df`` : float64
            Legacy column name for ``degrees_freedom``.

        This incremental DataFrame is distinct from the structured object
        returned by :meth:`tbr.TBRAnalysis.summarize`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from tbr import perform_tbr_analysis
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
        ...     data,
        ...     "date",
        ...     "control",
        ...     "test",
        ...     pretest_start=pd.Timestamp("2023-01-01"),
        ...     test_start=pd.Timestamp("2023-01-31"),
        ...     test_end=pd.Timestamp("2023-02-14"),
        ...     level=0.80,
        ...     threshold=0.0,
        ... )
        >>> summary = results.summary()
        >>> print(summary.tail())  # Last 5 days
        """
        return self._summary_df.copy()

    def tbr_dataframe(self) -> pd.DataFrame:
        """
        Get the comprehensive TBR analysis dataframe.

        Returns the complete TBR dataframe combining the baseline, pretest,
        and test periods with computed statistics using standard column names.
        Cooldown rows are retained in period metadata but are not included in
        this dataframe.

        Returns
        -------
        pd.DataFrame
            Copy containing original source columns. See Notes for the
            additional stable columns.

        Notes
        -----
        The returned DataFrame contains the original source columns plus:

        ``period`` : int64
            ``-1`` baseline, ``0`` pretest, ``1`` test.
        ``y`` : source-dependent
            Observed treatment/test metric.
        ``x`` : source-dependent
            Observed control metric.
        ``pred`` : float64
            Pretest fitted value or test counterfactual prediction; missing in
            baseline.
        ``predsd`` : float64
            Test counterfactual prediction standard deviation; ``0.0`` in
            pretest and missing in baseline.
        ``dif`` : float64
            Pretest residual or pointwise test effect; missing in baseline.
        ``cumdif`` : float64
            Cumulative test effect; missing outside test.
        ``cumsd`` : float64
            Cumulative-effect standard error (Student-t scale) under a legacy
            column name; ``0.0`` in pretest and missing in baseline.
        ``estsd`` : float64
            Fitted-value standard error; available in pretest and missing in
            baseline and test.

        The user-named time, control, test, and any additional source columns
        retain source-dependent dtypes where pandas permits.

        This method returns a copy to prevent accidental modification of stored
        results. For time-indexed Series access, use property accessors such as
        ``results.cumulative_effect``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from tbr import perform_tbr_analysis
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
        ...     data,
        ...     "date",
        ...     "control",
        ...     "test",
        ...     pretest_start=pd.Timestamp("2023-01-01"),
        ...     test_start=pd.Timestamp("2023-01-31"),
        ...     test_end=pd.Timestamp("2023-02-14"),
        ...     level=0.80,
        ...     threshold=0.0,
        ... )
        >>> tbr_df = results.tbr_dataframe()
        >>> tbr_df.to_csv('tbr_results.csv', index=False)  # doctest: +SKIP
        >>>
        >>> # Filter to test period only
        >>> test_period = tbr_df[tbr_df['period'] == 1]

        See Also
        --------
        summary : Get daily incremental summary statistics
        """
        return self._internal_df.copy()

    def conf_int(self, level: Optional[float] = None) -> pd.DataFrame:
        """
        Get credible interval bounds for final estimate.

        Parameters
        ----------
        level : float, optional
            Credibility level (e.g., 0.80, 0.95). If not specified,
            uses the level from analysis initialization.

        Returns
        -------
        pd.DataFrame
            New single-row credible-bounds DataFrame. See Notes.

        Notes
        -----
        The returned DataFrame has these stable columns:

        ``lower`` : float64
            Lower credible bound.
        ``upper`` : float64
            Upper credible bound.

        With ``level=None``, the method returns bounds at the configured
        analysis level. An explicit different level recomputes the bounds from
        the final estimate and cumulative standard error.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from tbr import perform_tbr_analysis
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
        ...     data,
        ...     "date",
        ...     "control",
        ...     "test",
        ...     pretest_start=pd.Timestamp("2023-01-01"),
        ...     test_start=pd.Timestamp("2023-01-31"),
        ...     test_end=pd.Timestamp("2023-02-14"),
        ...     level=0.80,
        ...     threshold=0.0,
        ... )
        >>> ci = results.conf_int()  # Uses 0.80
        >>> ci_95 = results.conf_int(level=0.95)  # Override to 0.95
        """
        if level is None:
            level = self._level

        # Recompute credible interval if different level requested
        if level != self._level:
            final_effect = self.estimate
            final_se = float(self._cumulative_se.iloc[-1])
            df = self._model_params["degrees_freedom"]
            t_crit = stats.t.ppf((1 + level) / 2, df)
            precision = t_crit * final_se
            lower = final_effect - precision
            upper = final_effect + precision
        else:
            lower = self.conf_int_lower
            upper = self.conf_int_upper

        return pd.DataFrame({"lower": [lower], "upper": [upper]})

    def __repr__(self) -> str:
        """
        Generate string representation.

        Returns
        -------
        str
            Multi-line summary of key results.
        """
        return (
            f"<TBRResults>\n"
            f"Cumulative Effect: {self.estimate:.2f}\n"
            f"{self._level*100:.0f}% CI: "
            f"[{self.conf_int_lower:.2f}, {self.conf_int_upper:.2f}]\n"
            f"Probability > {self._threshold}: {self.pvalue:.3f}\n"
            f"Test Period: {self.n_test_days} observations\n"
            f"Model: α = {self.alpha:.3f}, β = {self.beta:.3f}, σ = {self.sigma:.3f}"
        )

    def __str__(self) -> str:
        """Generate user-friendly string representation."""
        return self.__repr__()
