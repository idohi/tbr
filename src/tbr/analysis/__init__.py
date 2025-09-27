"""
TBR Analysis Framework.

This module provides comprehensive analysis tools and advanced TBR features
for creating summary statistics, incremental analysis, subinterval analysis,
and model diagnostics. All functions maintain 100% mathematical compatibility
with the proven functional implementation while providing clean, modular interfaces.

The module follows SPEC-1 lazy loading patterns for optimal performance and
integrates seamlessly with the existing validation and core module infrastructure.
The modular design follows patterns established by top scientific PyPI packages
like SciPy, Pandas, and Statsmodels.

Modules
-------
summary : TBR summary statistics generation
incremental : Day-by-day incremental analysis
subinterval : Custom time window analysis
diagnostics : Model validation and assumption checking (future)

Functions
---------
create_tbr_summary : Create single-row TBR summary with credible intervals
create_incremental_tbr_summaries : Create day-by-day incremental summaries
compute_interval_estimate_and_ci : Compute subinterval effect estimate and credible interval
analyze_multiple_subintervals : Analyze multiple time windows simultaneously
create_subinterval_summary : Create comprehensive subinterval analysis summary

Examples
--------
>>> from tbr.analysis import create_tbr_summary
>>> summary = create_tbr_summary(
...     tbr_dataframe, alpha=50, beta=0.95, sigma=25,
...     var_alpha=100, var_beta=0.001, cov_alpha_beta=-0.05,
...     degrees_freedom=43, level=0.80, threshold=0.0
... )
>>> print(f"Effect estimate: {summary['estimate'].iloc[0]:.2f}")

>>> from tbr.analysis import create_incremental_tbr_summaries
>>> incremental = create_incremental_tbr_summaries(
...     tbr_dataframe, alpha=50, beta=0.95, sigma=25,
...     var_alpha=100, var_beta=0.001, cov_alpha_beta=-0.05,
...     degrees_freedom=43, level=0.80, threshold=0.0
... )
>>> print(f"Day 1 effect: {incremental.iloc[0]['estimate']:.2f}")

>>> from tbr.analysis import compute_interval_estimate_and_ci
>>> result = compute_interval_estimate_and_ci(
...     tbr_dataframe, tbr_summary, start_day=5, end_day=10, ci_level=0.80
... )
>>> print(f"Days 5-10 effect: {result['estimate']:.2f}")
"""

# Lazy imports for performance (SPEC-1)
import lazy_loader as lazy

# SPEC-1 Lazy Loading Implementation for analysis module
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["summary", "incremental", "subinterval"],
    submod_attrs={
        "summary": ["create_tbr_summary"],
        "incremental": ["create_incremental_tbr_summaries"],
        "subinterval": [
            "compute_interval_estimate_and_ci",
            "analyze_multiple_subintervals",
            "create_subinterval_summary",
            "validate_subinterval_parameters",
        ],
    },
)
