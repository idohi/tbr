# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added verified Python 3.12 support to package metadata, Tox, CI, release
  validation, Black targets, and the README. Python 3.12 installs use tested
  lower bounds for NumPy, pandas, SciPy, and statsmodels without precautionary
  upper caps.
- Added an executable Sphinx-Gallery for the three example workflows, with
  captured output and downloadable Python scripts and notebooks.
- Added Matplotlib figures for counterfactual predictions, cumulative and
  subinterval effects, and credible-interval precision.
- `TBRSubintervalResult.precision`: the credible-interval half-width, matching
  `TBRSummaryResult.precision`. It follows `se` in `to_dict()`, `to_json()`,
  and `to_csv()`.
- `compute_interval_estimate_and_ci` and `analyze_multiple_subintervals` return
  an `se` key, and `create_subinterval_summary` returns an `se` column, holding
  the standard error of the subinterval estimate. Both precede `precision`.
- Added public `TBRResidualStatistics`, `TBRResidualAnalysis`, and
  `TBRDiagnosticReport` schemas for the mappings returned by the diagnostic
  API.

### Fixed
- Corrected the public return annotations for `analyze_tbr_residuals` and
  `create_tbr_diagnostic_report` to describe their actual mapping schemas.
- Diagnostic reports now name linearity when it is the only failed assumption,
  instead of emitting an empty `Assumption violations:` finding.
- Subinterval credible intervals (`TBRAnalysis.analyze_subinterval`,
  `compute_interval_estimate_and_ci`, `analyze_multiple_subintervals`,
  `create_subinterval_summary`) omitted model uncertainty and were too narrow.
  The posterior variance is now T_s·σ² + T_s²·v, which accounts for the
  correlation between counterfactual predictions that share the estimated
  coefficients. Intervals are wider than before, and subintervals previously
  flagged as excluding the threshold may no longer be flagged.
- Corrected the subinterval posterior variance formula in
  `docs/mathematical_methodology.md` and in the subinterval docstrings.
- `compute_interval_estimate_and_ci`, `analyze_multiple_subintervals`, and
  `create_subinterval_summary` now validate their inputs with
  `validate_subinterval_parameters`. An `end_day` beyond the last test day
  raises `ValueError` instead of silently using fewer days, and an empty
  `tbr_summary` raises a descriptive `ValueError` instead of `IndexError`.
- The scripts in `examples/` failed because they passed time boundaries as
  strings, where `pd.Timestamp` values are required, and
  `plot_incremental_analysis.py` analyzed a 60th test day that its 59-day test
  period did not have. Two used noise-free sample data and reported zero
  uncertainty, and `plot_method_chaining.py` applied its simulated lift to the
  pretest period too, so its data contained no treatment effect. The examples
  now run, and a test executes each one.
- The examples no longer suggest stopping a test early, claim that the
  cumulative credible interval narrows over time (it widens), call credible
  intervals confidence intervals, or label an interval entirely above zero as
  significant. The credibility level and probability threshold in their output
  now come from the fitted model and its results, and they print the simulated
  true effect for comparison.
- The Quick Start in `README.md` simulated a test group unrelated to the
  control group and with no treatment effect, so its reported effect was
  noise. `docs/api/quickstart.md` gave 2024-02-28 as the last test day instead
  of 2024-02-29. Both now use the simulated data of `plot_basic_analysis.py`.
- `docs/api/patterns.md` and `docs/api/results.md` used simulated effects that
  produced posterior probabilities near 1. They now use the same 1% lift data
  as the basic example. The patterns page also uses exact credibility-level
  labels, reports pre-specified monitoring days instead of the first positive
  interval, and covers the complete test period with its subinterval windows.
- `docs/mathematical_methodology.md` stated that a statistically significant
  F-statistic confirms that the linear relationship is meaningful. It now
  states that it is evidence against a zero slope.
- The RMSE formula in `docs/mathematical_methodology.md` divided by n - 2,
  while the `rmse` reported by `calculate_goodness_of_fit` divides by n. The
  document now gives the formula the package uses.
- Upper-tail probabilities and p-values are computed with the survival
  function (`sf`) instead of `1 - cdf`, which loses precision in the upper tail
  and rounds probabilities below about 1e-16 to 0.
- The `compare_posterior_probabilities` docstring example reported the first
  scenario as the one with the strongest evidence; it now selects the scenario
  ranked 1.
- `calculate_p_value`, `calculate_posterior_probability`, `create_tbr_summary`,
  and `compare_posterior_probabilities` returned a probability or p-value of
  1.0 or 0.5 when an input was NaN, and accepted infinite inputs. They now
  raise `ValueError` for non-finite inputs.
- `perform_threshold_sensitivity_analysis` and
  `calculate_incremental_posterior_probabilities` accepted non-finite inputs
  and could return NaN probabilities. They now raise `ValueError`.
  `calculate_incremental_posterior_probabilities` also rejects negative
  standard errors and handles zero standard errors deterministically.
- `create_tbr_summary` returned a probability of 0.5 when the final cumulative
  standard deviation was 0 or negative. It now returns 1 or 0 when it is 0, as
  `calculate_posterior_probability` does, and raises `ValueError` when it is
  negative.

### Changed
- Static type checking now uses a reproducibly pinned Python 3.8-targeted mypy
  environment and runs in a dedicated CI job.
- **Breaking:** `tbr.core.calculate_posterior_variance` now takes
  `(x_values, sigma, var_alpha, var_beta, cov_alpha_beta)` instead of
  `(estsd_values, n_days, sigma)`, and returns the corrected posterior variance.
- **Breaking:** `compute_interval_estimate_and_ci`,
  `analyze_multiple_subintervals`, and `create_subinterval_summary` now read
  the `x` column and the `var_alpha`, `var_beta`, and `alpha_beta_cov` summary
  fields, and no longer read `estsd`. Hand-built inputs without these columns
  raise a `ValueError` naming the missing columns, and results change for
  hand-built inputs with finite test-period `estsd` values. Outputs from
  `TBRAnalysis` and `perform_tbr_analysis` include these columns and are
  affected only by the variance fix.
- `validate_subinterval_parameters` now requires the `x` column and the
  `var_alpha`, `var_beta`, and `alpha_beta_cov` summary fields, and no longer
  requires `estsd`. Outputs from `TBRAnalysis` and `perform_tbr_analysis`
  already include these columns. It also accepts NumPy integer days and NumPy
  float `ci_level` values.
- **Breaking:** `TBRSubintervalResult.se`, returned by
  `TBRAnalysis.analyze_subinterval`, is now the standard error of the
  subinterval estimate. It previously held the credible-interval half-width.
  Use `TBRSubintervalResult.precision` for the half-width. Code that constructs
  `TBRSubintervalResult` directly must also pass `precision`.
- **Breaking:** `create_subinterval_summary` renames the `significant` column
  to `interval_excludes_threshold` and the `significance_threshold` parameter
  to `threshold`. The flag is unchanged: it is `True` when the credible
  interval lies entirely above or below the threshold. It is not a hypothesis
  test.
- **Breaking:** `TBRSummaryResult.is_significant()` is renamed to
  `meets_probability_threshold()`. Behavior is unchanged: it returns whether
  `prob` is at least `probability_threshold`. It is not a hypothesis test.
- **Breaking:** `calculate_p_value` replaces `two_tailed` with
  `alternative="two-sided"`, `"greater"`, or `"less"`. A one-sided direction
  must now be specified before observing the statistic. The previous
  `two_tailed=False` behavior selected the direction from the statistic's sign,
  doubling the nominal false-positive rate under the symmetric t null.
- `compare_posterior_probabilities` gives tied scenarios the same rank
  (for example, 1, 1, 3); previously, ties were ranked in arbitrary order.
- **Breaking:** The `examples` optional dependency group now installs only
  Matplotlib. Jupyter and Seaborn were removed because the examples do not use
  them.

### Removed
- **Breaking:** `compare_posterior_probabilities` no longer returns
  `relative_strength`. It divided each scenario's posterior probability by
  their sum, which is neither a probability nor a measure of evidence, because
  the scenarios' events are not mutually exclusive.
- **Breaking:** Removed `tbr.core.optimize_threshold_selection`. Its objectives
  selected a boundary of the arbitrary search range rather than a statistically
  meaningful optimum.
- **Breaking:** Removed `tbr.core.validate_posterior_assumptions`. Its
  independence p-value used a fixed scale of 0.5, matching its approximate
  null standard deviation of 2/√n only when n = 16. Its unsupported overall
  labels combined unrelated diagnostic checks into "Valid", "Questionable",
  or "Invalid".

## [0.1.6] - 2026-07-20

### Added
- Comprehensive mathematical methodology document (`docs/mathematical_methodology.md`)
  covering model specification, prediction and uncertainty, posterior inference
  (sections 3–7), assumptions and limitations, a notation reference table mapping
  mathematical symbols to Python variables, and supporting literature including
  the foundational Kerman, Wang & Vaver (2017) TBR paper.
- Sphinx + Read the Docs configuration for rendered documentation site
  at https://tbr.readthedocs.io/, with LaTeX rendering via MathJax,
  full-text search, bibliography support via sphinxcontrib-bibtex,
  type-hint rendering via sphinx-autodoc-typehints, and intersphinx
  cross-references to NumPy, SciPy, pandas, and statsmodels.

### Changed
- README updated with Read the Docs build badge and link to documentation site.

## [0.1.5] - 2026-01-08

### Changed
- Added References section to README citing foundational TBR paper (Kerman, Wang & Vaver, 2017)
- Added CITATION.cff for formal citation support
- Updated README description and Mathematical Foundation section
- Implemented single source of truth for version (reads from pyproject.toml)

## [0.1.4] - 2026-01-07

### Fixed
- Fixed `__version__` in package to match release version

## [0.1.3] - 2026-01-07

### Changed
- README updated to evergreen format (removed version-specific language)

## [0.1.2] - 2026-01-07

### Changed
- First organizationally-approved public release
- No code changes from 0.1.1

## [0.1.1] - 2025-11-08

### Changed
- **License**: Changed from Apache-2.0 to BSD-3-Clause to align with the scientific Python ecosystem (NumPy, pandas, SciPy, statsmodels)
  - Maintains same permissive terms
  - Enhances compatibility and familiarity for scientific community
  - Reduces adoption friction for researchers and organizations

## [0.1.0] - 2025-11-04

### 🎉 First Public Release (Beta)

**TBR v0.1.0** is a feature-complete Python package for Time-Based Regression (TBR) analysis.
This release includes complete functionality and 100% test coverage.
Marked as Beta to gather real-world feedback before v1.0 stable release.

### Added

#### Core Functionality
- **Functional API**: Complete `perform_tbr_analysis()` function with all TBR mathematical implementations
- **OOP API**: `TBRAnalysis` class with `fit()`, `predict()`, and scikit-learn-style parameter management through `get_params()` and `set_params()`
- **Result Objects**: Dataclass-based result containers (`TBRSummaryResult`, `TBRPredictionResult`, `TBRSubintervalResult`)
- **Export Utilities**: JSON and CSV export with full metadata preservation
- **Method Chaining**: Fluent API patterns for streamlined workflows

#### Analysis Capabilities
- **Summary Analysis**: Complete treatment effect estimation with credible intervals
- **Incremental Analysis**: Day-by-day progression of treatment effects
- **Subinterval Analysis**: Custom time window analysis with configurable confidence levels
- **Counterfactual Predictions**: Model-based estimation with uncertainty quantification
- **Diagnostics**: Comprehensive model validation and assumption checking

#### Statistical Features
- **Rigorous Mathematics**: Exact TBR formulas with complete theoretical foundation
- **Variance Quantification**: Proper uncertainty estimation separating model and residual components
- **Credible Intervals**: t-distribution based intervals with proper degrees of freedom
- **Posterior Probabilities**: Bayesian threshold exceedance testing
- **Performance Diagnostics**: Computational efficiency metrics and optimization recommendations

#### Testing & Quality
- **1,227 Tests**: Comprehensive test suite covering all functionality
- **100% Code Coverage**: 2,365 statements, 888 branches fully covered
- **Test Categories**: Unit (726), Integration (143), Mathematical (123), Performance (39)
- **Performance Validation**: OOP API overhead < 50%, linear O(n) scalability confirmed
- **Cross-validation**: OOP vs Functional API equivalence at machine precision (rtol=1e-14)

#### Documentation
- **Quick Start Guide**: Installation and basic usage examples
- **API Reference**: Complete documentation of all classes and methods
- **Result Objects Guide**: Comprehensive documentation of result structures
- **Best Practices**: Common patterns and domain-specific guidance
- **Examples**: Three complete example scripts demonstrating workflows

#### Infrastructure
- **Lazy Loading**: SPEC-1 compliant lazy imports for optimal memory usage
- **Type Hints**: Complete type annotations throughout codebase
- **Pre-commit Hooks**: Black, isort, Ruff, MyPy, pydocstyle, interrogate, vulture
- **CI/CD Pipeline**: GitHub Actions with multi-platform testing (Ubuntu, Windows, macOS)
- **Python Support**: Python 3.8, 3.9, 3.10, 3.11, 3.12

### Changed
- Upgraded from alpha (0.1.0a1) to stable release (0.1.0)
- Improved performance with lazy loading (64-99% faster import times)
- Enhanced error messages with actionable guidance
- Optimized memory efficiency across all operations

### Fixed
- All pre-commit hook issues resolved
- Mathematical correctness validated against theoretical derivations
- Cross-platform compatibility issues resolved
- Numerical stability edge cases handled

### Performance
- **Import Time**: 64-99% faster with lazy loading
- **OOP Overhead**: < 50% vs functional API (often faster)
- **Method Chaining**: < 10% overhead
- **Scalability**: Linear O(n) from 50 to 10,000 samples
- **Memory**: Efficient garbage collection and resource management

### Breaking Changes
None - First stable release

## [0.1.0a1] - 2025-09-16

### Added
- Core TBR functional implementation (`tbr.functional.tbr_functions` module)
- Input validation utilities (17 functions in `tbr.utils.validation`)
- Testing framework with 194 tests across 4 categories (unit, integration, mathematical, performance)
- Cross-platform support (Python 3.8-3.12)
- Package structure following PyPI standards
- Continuous Integration pipeline with GitHub Actions
- Pre-commit hooks for code quality enforcement
- Custom exception classes (`tbr.utils.exceptions`)
- Package configuration for PyPI distribution

---

**Note**: This project is in active development. Version 1.0.0 will mark the first stable release suitable for production use.
