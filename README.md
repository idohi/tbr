# TBR - Time-Based Regression Analysis Package

[![PyPI version](https://badge.fury.io/py/tbr.svg)](https://badge.fury.io/py/tbr)
[![Build Status](https://github.com/idohi/tbr/workflows/CI/badge.svg)](https://github.com/idohi/tbr/actions)
[![Documentation Status](https://readthedocs.org/projects/tbr/badge/?version=latest)](https://tbr.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/idohi/tbr/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Development Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://pypi.org/project/tbr/)

A comprehensive Python package for Time-Based Regression (TBR) analysis. TBR combines regression techniques with time series analysis to estimate treatment effects in before-after studies. Applications span marketing effectiveness, medical research, A/B testing, policy evaluation, and more.

## Status: Beta

**TBR** is feature-complete and ready for production use with:
- Complete TBR functionality (functional + OOP APIs)
- 1,200+ tests with 100% code coverage
- Intuitive, type-safe API interfaces
- Export utilities (JSON, CSV)
- Performance validated (linear O(n) scalability)
- Cross-platform support (Python 3.8-3.12)

**Why Beta?** While comprehensively tested, we're gathering real-world feedback before declaring v1.0 stable. We encourage production use and welcome your feedback!

## Features

- **Domain-Agnostic**: Works with any treatment/control group time series data
- **Comprehensive Analysis**: Lift calculation, counterfactual predictions, statistical inference
- **Statistical Rigor**: Credible intervals, uncertainty decomposition, and posterior probability assessments
- **Flexible**: Temporal and cumulative analysis, subinterval analysis, incremental analysis
- **Well-Tested**: Type hints, 100% code coverage, comprehensive test suite
- **Easy to Use**: Simple, intuitive API for both quick analysis and advanced workflows

## Installation

```bash
pip install tbr
```

Optional dependencies:
```bash
pip install tbr[dev]       # Development tools
pip install tbr[docs]      # Documentation tools
pip install tbr[examples]  # Example dependencies
```

## Quick Start

```python
import numpy as np
import pandas as pd
from tbr import TBRAnalysis

# Simulated daily data: the test group tracks the control group, with a 1% lift
# from 2024-02-15
rng = np.random.default_rng(114)
dates = pd.date_range("2024-01-01", periods=90, freq="D")
control = rng.normal(1000, 50, len(dates))
test = control + rng.normal(0, 25, len(dates))
in_test_period = dates >= pd.Timestamp("2024-02-15")
test[in_test_period] = test[in_test_period] * 1.01
data = pd.DataFrame({"date": dates, "control": control, "test": test})

# Initialize and fit model
model = TBRAnalysis(level=0.80)
model.fit(
    data=data,
    time_col="date",
    control_col="control",
    test_col="test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-02-15"),
    test_end=pd.Timestamp("2024-03-31"),  # Exclusive: last test day is 2024-03-30
)

# Get results
summary = model.summarize()
print(f"Cumulative effect: {summary.estimate:.2f}")
print(
    f"{summary.level * 100:g}% credible interval: "
    f"[{summary.lower:.2f}, {summary.upper:.2f}]"
)
print(f"Posterior probability (effect > {summary.threshold:g}): {summary.prob:.3f}")

# Additional capabilities
predictions = model.predict()
subinterval = model.analyze_subinterval(start_day=1, end_day=10)
incremental = model.summarize_incremental()
summary.to_json("results.json")
```

## Key Capabilities

- **Counterfactual Predictions**: Estimates what would have happened without treatment
- **Lift Calculations**: Treatment effect with statistical uncertainty quantification
- **Credible Intervals**: Posterior intervals using the Student's t-distribution
- **Posterior Probability**: Probability that an effect exceeds a chosen threshold
- **Flexible Analysis**: Subinterval analysis, incremental tracking, and custom credibility levels

## Mathematical Foundation

TBR implements statistical methods for estimating causal effects in before-after study designs:

- **Regression Modeling**: Establishes relationship between control and test groups
- **Counterfactual Prediction**: Estimates what would have occurred without intervention
- **Bayesian Inference**: Credible intervals with uncertainty quantification
- **Uncertainty Decomposition**: Separates residual and model contributions

The [mathematical methodology](https://tbr.readthedocs.io/en/latest/mathematical_methodology.html)
defines the package's canonical notation and maps mathematical quantities to
Python parameters, result fields, and DataFrame columns.

## Documentation

- **Examples**: See `examples/` directory in the repository
- **Full Documentation**: [tbr.readthedocs.io](https://tbr.readthedocs.io/)
- **Mathematical Methodology**: [TBR Mathematical Methodology](https://tbr.readthedocs.io/en/latest/mathematical_methodology.html)

## Version Compatibility

- **Python**: 3.8+ (tested on 3.8, 3.9, 3.10, 3.11, 3.12)
- **pandas**: 2.0+
- **numpy**: 1.24+
- **scipy**: 1.10+
- **statsmodels**: 0.14+

## References

The mathematical framework and notation in this package are based on:

> Kerman, J., Wang, P., & Vaver, J. (2017). *Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework*. Technical Report, Google, Inc. [PDF](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45950.pdf)

This package provides a domain-agnostic Python implementation of the time-based regression methodology for any before-after intervention study.

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](https://github.com/idohi/tbr/blob/main/LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/idohi/tbr/issues)
