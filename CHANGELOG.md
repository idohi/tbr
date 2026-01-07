# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

**TBR v0.1.0** is a comprehensive, feature-complete Python package for Time-Based Regression (TBR) analysis.
This release includes complete functionality, 100% test coverage, and professional API design suitable for
scientific and commercial applications. Marked as Beta to gather real-world feedback before v1.0 stable release.

### Added

#### Core Functionality
- **Functional API**: Complete `perform_tbr_analysis()` function with all TBR mathematical implementations
- **OOP API**: `TBRAnalysis` class with sklearn-compatible interface (`fit()`, `predict()`, `get_params()`, `set_params()`)
- **Result Objects**: Professional dataclass-based result containers (`TBRSummaryResult`, `TBRPredictionResult`, `TBRSubintervalResult`)
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
