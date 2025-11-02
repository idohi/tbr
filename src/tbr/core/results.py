"""
Professional result object structures for TBR analysis outputs.

This module provides structured result containers for TBR analysis methods,
following professional scientific PyPI package standards (SciPy, Statsmodels).

Result objects provide:
- Type-safe attribute access
- Rich string representations
- Conversion methods (to_dict, to_dataframe)
- Comprehensive metadata
- Immutability via frozen dataclasses

Examples
--------
>>> from tbr import TBRAnalysis
>>> model = TBRAnalysis(level=0.80)
>>> model.fit(data, 'date', 'control', 'test', ...)
>>>
>>> # Access prediction results
>>> predictions = model.predict()
>>> print(predictions.predictions.head())  # DataFrame
>>> print(f"Mean prediction: {predictions.predictions['pred'].mean():.2f}")
>>>
>>> # Access summary results
>>> summary = model.summarize()
>>> print(f"Effect: {summary.estimate:.2f}")
>>> print(f"CI: [{summary.lower:.2f}, {summary.upper:.2f}]")
>>> print(f"Probability: {summary.prob:.3f}")
>>>
>>> # Access subinterval results
>>> week1 = model.analyze_subinterval(1, 7)
>>> print(f"Week 1 effect: {week1.estimate:.2f} ± {week1.se:.2f}")

Notes
-----
All result objects are frozen dataclasses, providing immutability and type safety.
They follow patterns from scipy.stats and statsmodels for professional consistency.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TBRPredictionResult:
    """
    Result container for TBR counterfactual predictions.

    Contains predictions with uncertainty estimates and metadata about
    the model used to generate them.

    Attributes
    ----------
    predictions : pd.DataFrame
        DataFrame with columns:
        - pred: Predicted counterfactual values
        - predsd: Prediction standard deviations (uncertainty)
    n_predictions : int
        Number of predictions generated
    model_params : Dict[str, float]
        Model parameters used (alpha, beta, sigma, etc.)
    control_values : np.ndarray
        Control values used for predictions

    Examples
    --------
    >>> result = model.predict()
    >>> print(result.predictions.head())
    >>> print(f"Generated {result.n_predictions} predictions")
    >>> print(f"Model alpha: {result.model_params['alpha']:.3f}")
    """

    predictions: pd.DataFrame
    n_predictions: int
    model_params: Dict[str, float]
    control_values: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.

        Returns
        -------
        dict
            Dictionary with all result attributes
        """
        return {
            "predictions": self.predictions,
            "n_predictions": self.n_predictions,
            "model_params": self.model_params,
            "control_values": self.control_values,
        }

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
        Mean prediction uncertainty.

        Returns
        -------
        float
            Average of prediction standard deviations
        """
        return float(self.predictions["predsd"].mean())

    def __repr__(self) -> str:
        """Generate professional string representation."""
        return (
            f"TBRPredictionResult(\n"
            f"  n_predictions={self.n_predictions},\n"
            f"  mean_pred={self.mean_pred:.3f},\n"
            f"  mean_uncertainty={self.mean_uncertainty:.3f}\n"
            f")"
        )


@dataclass(frozen=True)
class TBRSummaryResult:
    """
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
        Degrees of freedom

    Examples
    --------
    >>> result = model.summarize()
    >>> print(f"Effect: {result.estimate:.2f}")
    >>> print(f"95% CI: [{result.lower:.2f}, {result.upper:.2f}]")
    >>> print(f"Significant: {result.prob > 0.95}")
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

    def to_dict(self) -> Dict[str, float]:
        """
        Convert result to dictionary format.

        Returns
        -------
        dict
            Dictionary with all summary statistics
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
            Single-row DataFrame with all summary statistics
        """
        return pd.DataFrame([self.to_dict()])

    def is_significant(self, probability_threshold: float = 0.95) -> bool:
        """
        Check if effect is statistically significant.

        Parameters
        ----------
        probability_threshold : float, default=0.95
            Probability threshold for significance

        Returns
        -------
        bool
            True if posterior probability exceeds threshold
        """
        return self.prob >= probability_threshold

    def __repr__(self) -> str:
        """Generate professional string representation."""
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
        Standard error of the estimate
    ci_level : float
        Credibility level used for interval
    start_day : int
        Starting day of subinterval (1-indexed)
    end_day : int
        Ending day of subinterval (1-indexed)
    n_days : int
        Number of days in the subinterval

    Examples
    --------
    >>> result = model.analyze_subinterval(1, 7)
    >>> print(f"Week 1 effect: {result.estimate:.2f}")
    >>> print(f"CI: [{result.lower:.2f}, {result.upper:.2f}]")
    >>> print(f"Days: {result.start_day}-{result.end_day} ({result.n_days} days)")
    >>> if result.contains_zero():
    ...     print("Effect not significant (CI contains zero)")
    """

    estimate: float
    lower: float
    upper: float
    se: float
    ci_level: float
    start_day: int
    end_day: int
    n_days: int

    def to_dict(self) -> Dict[str, float]:
        """
        Convert result to dictionary format.

        Returns
        -------
        dict
            Dictionary with all subinterval statistics
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

    def contains_zero(self) -> bool:
        """
        Check if credible interval contains zero.

        Returns
        -------
        bool
            True if interval contains zero (effect not significant)
        """
        return self.lower <= 0 <= self.upper

    def is_positive(self) -> bool:
        """
        Check if entire credible interval is positive.

        Returns
        -------
        bool
            True if lower bound > 0 (positive effect with high confidence)
        """
        return self.lower > 0

    def is_negative(self) -> bool:
        """
        Check if entire credible interval is negative.

        Returns
        -------
        bool
            True if upper bound < 0 (negative effect with high confidence)
        """
        return self.upper < 0

    def __repr__(self) -> str:
        """Generate professional string representation."""
        return (
            f"TBRSubintervalResult(\n"
            f"  days={self.start_day}-{self.end_day} (n={self.n_days}),\n"
            f"  estimate={self.estimate:.3f},\n"
            f"  CI=[{self.lower:.3f}, {self.upper:.3f}] (level={self.ci_level}),\n"
            f"  se={self.se:.3f}\n"
            f")"
        )
