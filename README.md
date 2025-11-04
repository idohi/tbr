# TBR - Time-Based Regression Analysis Package

[![PyPI version](https://badge.fury.io/py/tbr.svg)](https://badge.fury.io/py/tbr)
[![Build Status](https://github.com/idohi/tbr/workflows/CI/badge.svg)](https://github.com/idohi/tbr/actions)
[![Coverage Status](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/idohi/tbr/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://img.shields.io/badge/docs-testing%20guide-blue.svg)](docs/testing/testing.rst)
[![Development Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](https://pypi.org/project/tbr/)

A comprehensive, domain-agnostic Python package for Time-Based Regression (TBR) analysis. Perform rigorous statistical analysis of treatment/control group time series data across any industry - marketing, medical research, economics, and more.

## ✨ v0.1.0 - First Stable Release

**TBR v0.1.0** is production-ready with:
- ✅ Complete TBR functionality (functional + OOP APIs)
- ✅ 1,227 tests with 100% code coverage
- ✅ Professional API design following NumPy/Pandas/Scikit-learn patterns
- ✅ Export utilities (JSON, CSV)
- ✅ Performance validated (linear O(n) scalability)
- ✅ Cross-platform support (Python 3.8-3.12)

**Coming in Future Releases:**
- 📚 Sphinx documentation website (v0.2.0)
- 📊 Visualization tools (v0.3.0)
- 📓 Tutorial notebooks (v0.4.0)
- 🎯 v1.0.0 when API is fully stable

**See Full Details:** [CHANGELOG](CHANGELOG.md) | [Project Plan](ai_helper_notes/PROJECT_PLAN.md)

## 🚀 Features

- **Domain-Agnostic**: Works with any treatment/control group time series data
- **Comprehensive Analysis**: Lift calculation, counterfactual predictions, statistical inference
- **Statistical Rigor**: Credible intervals, significance tests, posterior probability assessments
- **Flexible**: Daily and cumulative analysis, subinterval analysis, incremental analysis
- **Production-Ready**: Type hints, comprehensive testing, professional documentation
- **PyPI Compatible**: Easy installation and distribution

## 📦 Installation

### Quick Install (PyPI)
```bash
# Basic installation (runtime dependencies only)
pip install tbr

# With development tools
pip install tbr[dev]

# With documentation tools
pip install tbr[docs]

# With example/tutorial dependencies
pip install tbr[examples]

# With everything
pip install tbr[dev,docs,examples]
```

### Development Installation

#### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/idohi/tbr.git
cd tbr

# If you have an existing virtual environment, deactivate it first
deactivate  # (optional, but recommended)

# Run the automated setup script - handles everything!
./scripts/setup.sh
```

**What the setup script does:**
- ✅ Checks and installs Python 3.11.9 (via pyenv)
- ✅ Creates/resets virtual environment (removes existing if found)
- ✅ Installs package with all optional dependencies (`pip install -e ".[dev,docs,examples]"`)
- ✅ Sets up pre-commit hooks (including docstring validation)
- ✅ Verifies everything works correctly (including docstring tools)

**Requirements:**
- `pyenv` installed ([installation guide](https://github.com/pyenv/pyenv#installation))

**First-time setup example:**
```bash
# 1. Install pyenv (if not already installed)
# macOS: brew install pyenv
# Linux: curl https://pyenv.run | bash

# 2. Clone and setup
git clone https://github.com/idohi/tbr.git
cd tbr
./scripts/setup.sh

# 3. Activate environment and start developing
source .venv/bin/activate
make test  # Run tests to verify everything works
```

#### Option 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/idohi/tbr.git
cd tbr

# Install Python 3.11.9 (if using pyenv)
pyenv install 3.11.9
pyenv local 3.11.9

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install package with all dependencies
pip install --upgrade pip
pip install -e ".[dev,docs,examples]"
```

## 🛠️ Development Workflow

This project uses a comprehensive development workflow with automated tools:

### Quick Commands
```bash
make help          # Show all available commands
make setup         # Complete environment setup
make test          # Run tests
make lint          # Run linting
make format        # Format code
make docstring     # Check docstring style and coverage
make build         # Build package
make all           # Run complete pipeline
```

### Development Tools
- **Testing**: `pytest` with coverage reporting
- **Code Quality**: `black`, `isort`, `ruff`, `mypy`
- **Docstring Validation**: `pydocstyle` (NumPy convention) + `interrogate` (90% coverage)
- **Pre-commit**: Automated code quality checks including docstring validation
- **Documentation**: `sphinx` with RTD theme
- **Build**: `build` system for PyPI distribution

### Environment Management
- **Python Version**: 3.11.9 (managed with `pyenv`)
- **Dependencies**: `pip-tools` for locked requirements
- **Virtual Environment**: `.venv` for isolation

### 🔧 Troubleshooting Setup

#### Reset Environment
If you need to reset your development environment:
```bash
deactivate              # Exit current virtual environment
./scripts/setup.sh      # Script automatically removes old .venv and creates fresh one
```

#### Manual Environment Reset
```bash
deactivate              # Exit current virtual environment
rm -rf .venv           # Remove virtual environment
./scripts/setup.sh      # Run setup script
```

#### Common Issues
- **`pyenv: command not found`**: Install pyenv first ([guide](https://github.com/pyenv/pyenv#installation))
- **Permission denied**: Make script executable with `chmod +x scripts/setup.sh`
- **Python version issues**: The script will install Python 3.11.9 automatically
- **Dependency conflicts**: The setup script creates a clean environment each time

## 📚 Quick Start (Alpha API)

> **Note**: The high-level `TBRAnalysis` class is coming in future releases. Currently available: functional API.

```python
import pandas as pd
import numpy as np
from tbr.functional import perform_tbr_analysis

# Example: Create time series data (date, control, test columns)
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'date': dates,
    'control': np.random.normal(100, 10, 100),  # Control group metric
    'test': np.random.normal(105, 10, 100)      # Treatment group metric
})

# Run TBR analysis
tbr_results, summary_results = perform_tbr_analysis(
    data=data,
    time_col='date',
    control_col='control',
    test_col='test',
    pretest_start='2023-01-01',  # Start of baseline period
    test_start='2023-02-15',     # Start of treatment period
    test_end='2023-04-10',       # End of analysis period
    level=0.80,                  # Confidence level for intervals
    threshold=0.0                # Threshold for significance testing
)

# View results
print("TBR Analysis Results:")
print(summary_results)

# The results contain:
# - Incremental lift estimates with credible intervals
# - Cumulative effects over time
# - Statistical significance tests
# - Posterior probability assessments
```

### What the Analysis Provides

- **Counterfactual Predictions**: What would have happened without treatment
- **Lift Calculations**: Treatment effect with statistical uncertainty
- **Credible Intervals**: Bayesian confidence bounds using t-distribution
- **Significance Testing**: Posterior probability of positive/negative effects
- **Time Series Output**: Daily and cumulative analysis over treatment period

## 📖 Documentation

- **API Reference**: [Full documentation](https://tbr.readthedocs.io/)
- **Examples**: See `examples/` directory
- **Notebooks**: Interactive tutorials in `notebooks/`
- **Mathematical Details**: See `references/tbr_parameter_derivations.md`

## 🧪 Testing

Our testing framework follows scientific PyPI package standards with comprehensive test categories. For detailed testing methodology, see our [Testing Guide](docs/testing/testing.rst).

### Quick Start
```bash
# Run all tests (recommended)
make test

# Run with coverage report
make test-cov

# Run specific test categories
pytest tests/unit/ -v          # Unit tests
pytest tests/integration/ -v   # Integration tests
pytest tests/mathematical/ -v  # Mathematical validation
pytest tests/performance/ -v   # Performance tests
```

### Test Categories
- **Unit Tests**: Individual function/class testing with comprehensive coverage
- **Integration Tests**: Package structure and imports validation
- **Mathematical Tests**: Known-value validation and algorithm verification
- **Performance Tests**: Speed, scalability, and memory efficiency

### CI-Local Parity Testing
```bash
# Run exact same tests as GitHub Actions CI (recommended for debugging)
make ci-local

# This runs the complete CI pipeline locally:
# 1. Install dependencies (pip install --upgrade pip && pip install -e .[dev])
# 2. Unit tests with coverage (pytest tests/unit/ -v --cov=src/tbr --cov-report=xml --cov-report=term-missing)
# 3. Integration tests (pytest tests/integration/ -v)
# 4. Mathematical validation tests (pytest tests/mathematical/ -v)
# 5. Performance tests (pytest tests/performance/ -v)
```

### Cross-Platform Testing
```bash
# Test across multiple Python versions (like CI)
make test-tox

# Test current Python version only
make test-tox-py
```

### Individual Test Execution
```bash
# Using Make commands (recommended)
make test-single TEST=tests/unit/test_validation.py::TestTimeColumnValidation::test_valid_datetime_column
make test-pattern PATTERN=datetime

# Direct pytest commands
pytest tests/unit/test_validation.py::TestTimeColumnValidation::test_valid_datetime_column -v  # Single test function
pytest tests/unit/test_validation.py::TestTimeColumnValidation -v                              # Single test class
pytest tests/unit/test_validation.py -v                                                        # All tests in file

# Run tests matching patterns (flexible - works with any test names)
pytest -k "validation" -v               # All tests with "validation" in name
pytest -k "datetime" -v                 # All tests with "datetime" in name
pytest -k "not performance" -v          # Exclude performance tests

# Run any specific test with detailed output
pytest tests/unit/test_core_functions.py::TestSumSquaredDeviations::test_basic_calculation -v -s
```

### Advanced Testing
```bash
# Run by test markers
pytest -m "mathematical" -v    # Mathematical validation only
pytest -m "performance" -v     # Performance tests only

# Coverage with HTML report
pytest tests/ --cov=src/tbr --cov-report=html
open htmlcov/index.html

# Show slowest tests
pytest tests/ --durations=10

# Debug failing tests with detailed output
pytest tests/unit/test_validation.py::TestTimeColumnValidation::test_valid_datetime_column -v -s --tb=long
```

### Debug CI Failures Locally
When GitHub Actions CI fails, use `make ci-local` to reproduce the exact same environment and commands locally for faster debugging and fixing.

## 🔧 Project Structure

```
tbr/
├── src/tbr/                 # Main package
│   ├── core/               # Core TBR functionality
│   ├── functional/         # Functional implementation
│   ├── analysis/           # Analysis tools
│   ├── utils/              # Utilities
│   └── visualization/      # Plotting tools
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── mathematical/      # Mathematical validation
│   └── performance/       # Performance tests
├── docs/                   # Documentation
├── examples/              # Usage examples
├── notebooks/             # Jupyter tutorials
├── scripts/               # Development scripts
└── references/            # Mathematical references
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Run `./scripts/setup.sh` for complete environment setup
3. Create a feature branch
4. Make your changes with tests
5. Run `make all` to validate
6. Submit a pull request

### Code Quality
- All code must pass `make check` (linting, type checking, formatting, docstring validation)
- Tests required for new features
- Documentation for public APIs (NumPy docstring convention required)
- Docstring coverage must be ≥90% (enforced by pre-commit hooks)
- Follow existing code style

## 📊 Mathematical Foundation

TBR analysis is based on rigorous statistical methods:

- **Ordinary Least Squares (OLS)** regression
- **Counterfactual prediction** with uncertainty quantification
- **Bayesian inference** for credible intervals
- **Variance decomposition** (model uncertainty + residual noise)
- **Statistical significance** testing

For detailed mathematical derivations, see `references/tbr_parameter_derivations.md`.

## 🔄 Version Compatibility

- **Python**: 3.8+ (tested on 3.8, 3.9, 3.10, 3.11, 3.12)
- **pandas**: 2.0+
- **numpy**: 1.24+
- **scipy**: 1.10+
- **statsmodels**: 0.14+

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Google's GeoexperimentsResearch R package
- Built with modern Python best practices
- Designed for the data science community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/idohi/tbr/issues)
- **Discussions**: [GitHub Discussions](https://github.com/idohi/tbr/discussions)
- **Documentation**: [Read the Docs](https://tbr.readthedocs.io/)

---

**Made with ❤️ for the data science community**
