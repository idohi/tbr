"""
TBR Analysis Framework.

This module provides comprehensive analysis tools and advanced TBR features
for creating summary statistics, incremental analysis, subinterval analysis,
and model diagnostics. All functions maintain 100% mathematical compatibility
with the proven functional implementation while providing clean, modular interfaces.

The module uses lazy loading for optimal performance and integrates seamlessly
with the existing validation and core module infrastructure.

Modules
-------
summary : TBR summary statistics generation
incremental : Day-by-day incremental analysis
subinterval : Custom time window analysis
diagnostics : Model validation and assumption checking
performance : Performance diagnostics and computational efficiency metrics

Functions
---------
create_tbr_summary : Create single-row TBR summary with credible intervals
create_incremental_tbr_summaries : Create day-by-day incremental summaries
compute_interval_estimate_and_ci : Compute subinterval effect estimate and credible interval
analyze_multiple_subintervals : Analyze multiple time windows simultaneously
create_subinterval_summary : Create comprehensive subinterval analysis summary
validate_tbr_model : Comprehensive TBR model validation with assumption checking
diagnose_tbr_analysis : End-to-end diagnostic workflow for TBR analysis
check_tbr_assumptions : Statistical assumption validation for TBR models
analyze_tbr_residuals : TBR-specific residual analysis and outlier detection
assess_tbr_performance : Performance and efficiency diagnostics
create_tbr_diagnostic_report : Comprehensive diagnostic reporting
TBRPerformanceAnalyzer : Specialized performance analyzer for TBR workflows
quick_performance_check : Quick performance analysis of TBR workflows
optimize_tbr_data_size : Find optimal data size for target performance

Examples
--------
>>> import pandas as pd
>>> from tbr.analysis import create_tbr_summary
>>> tbr_dataframe = pd.DataFrame(
...     {"period": [1, 1], "cumdif": [5.0, 8.0], "cumsd": [2.0, 3.0]}
... )
>>> model_terms = dict(
...     alpha=50, beta=0.95, sigma=25,
...     var_alpha=100, var_beta=0.001, cov_alpha_beta=-0.05,
...     degrees_freedom=43, level=0.80, threshold=0.0,
... )
>>> summary = create_tbr_summary(tbr_dataframe, **model_terms)
>>> print(f"Effect estimate: {summary['estimate'].iloc[0]:.2f}")

>>> from tbr.analysis import create_incremental_tbr_summaries
>>> incremental = create_incremental_tbr_summaries(tbr_dataframe, **model_terms)
>>> print(f"Day 1 effect: {incremental.iloc[0]['estimate']:.2f}")

>>> from tbr.analysis import compute_interval_estimate_and_ci
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
"""

# Lazy imports for performance
import lazy_loader as lazy

# Lazy loading implementation - modules load only when accessed
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["summary", "incremental", "subinterval", "diagnostics", "performance"],
    submod_attrs={
        "summary": ["create_tbr_summary"],
        "incremental": ["create_incremental_tbr_summaries"],
        "subinterval": [
            "compute_interval_estimate_and_ci",
            "analyze_multiple_subintervals",
            "create_subinterval_summary",
            "validate_subinterval_parameters",
        ],
        "diagnostics": [
            "validate_tbr_model",
            "diagnose_tbr_analysis",
            "check_tbr_assumptions",
            "analyze_tbr_residuals",
            "assess_tbr_performance",
            "create_tbr_diagnostic_report",
        ],
        "performance": [
            "TBRPerformanceAnalyzer",
            "quick_performance_check",
            "optimize_tbr_data_size",
        ],
    },
)
