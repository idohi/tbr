# TBR - Time-Based Regression Analysis Package

[![PyPI version](https://badge.fury.io/py/tbr.svg)](https://badge.fury.io/py/tbr)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, domain-agnostic Python package for Time-Based Regression (TBR) analysis. Perform rigorous statistical analysis of treatment/control group time series data across any industry - marketing, medical research, economics, and more.

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
- ✅ Sets up pre-commit hooks
- ✅ Verifies everything works correctly

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
make build         # Build package
make all           # Run complete pipeline
```

### Development Tools
- **Testing**: `pytest` with coverage reporting
- **Code Quality**: `black`, `isort`, `ruff`, `mypy`
- **Pre-commit**: Automated code quality checks
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

## 📚 Quick Start

```python
import pandas as pd
from tbr import TBRAnalysis

# Load your time series data
data = pd.read_csv('your_data.csv')

# Initialize TBR analysis
tbr = TBRAnalysis(
    data=data,
    date_column='date',
    metric_column='value',
    treatment_column='is_treatment'
)

# Run analysis
results = tbr.analyze(
    pre_period=('2023-01-01', '2023-06-30'),
    post_period=('2023-07-01', '2023-12-31')
)

# Get summary
print(results.summary())

# Plot results
results.plot()
```

## 📖 Documentation

- **API Reference**: [Full documentation](https://tbr.readthedocs.io/)
- **Examples**: See `examples/` directory
- **Notebooks**: Interactive tutorials in `notebooks/`
- **Mathematical Details**: See `references/tbr_parameter_derivations.md`

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_core.py -v

# Run mathematical validation tests
pytest tests/mathematical/ -v
```

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
- All code must pass `make check` (linting, type checking, formatting)
- Tests required for new features
- Documentation for public APIs
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
