"""
TBR Analysis Framework.

This module provides comprehensive analysis tools and advanced TBR features
for creating summary statistics, incremental analysis, subinterval analysis,
and model diagnostics. All functions maintain 100% mathematical compatibility
with the proven functional implementation while providing clean, modular interfaces.

The module follows SPEC-1 lazy loading patterns for optimal performance and
integrates seamlessly with the existing validation and core module infrastructure.

Modules
-------
summary : TBR summary statistics and incremental analysis
subinterval : Custom time window analysis (future)
diagnostics : Model validation and assumption checking (future)

Functions
---------
create_tbr_summary : Create single-row TBR summary with credible intervals
create_incremental_tbr_summaries : Create day-by-day incremental summaries

Examples
--------
>>> from tbr.analysis import create_tbr_summary
>>> summary = create_tbr_summary(
...     tbr_dataframe, alpha=50, beta=0.95, sigma=25,
...     var_alpha=100, var_beta=0.001, cov_alpha_beta=-0.05,
...     degrees_freedom=43, level=0.80, threshold=0.0
... )
>>> print(f"Effect estimate: {summary['estimate'].iloc[0]:.2f}")
"""

# Lazy imports for performance (SPEC-1)
import lazy_loader as lazy

# SPEC-1 Lazy Loading Implementation for analysis module
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["summary"],
    submod_attrs={
        "summary": ["create_tbr_summary", "create_incremental_tbr_summaries"],
    },
)
