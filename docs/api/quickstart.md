# TBR Quickstart

This guide runs a complete Time-Based Regression (TBR) analysis with the
object-oriented API. For the derivation and the complete notation-to-code
mapping, see the [mathematical methodology](../mathematical_methodology.md).

## Installation

```bash
pip install tbr
```

## Prepare reproducible data

TBR requires one time column and numeric control and test metric columns. The
time column may use a pandas datetime dtype, integers, or floats; its three
boundary arguments must use the corresponding type.

```python
import numpy as np
import pandas as pd

from tbr import TBRAnalysis

rng = np.random.default_rng(42)
dates = pd.date_range("2024-01-01", periods=60)
control = rng.normal(1000.0, 35.0, len(dates))
noise = rng.normal(0.0, 8.0, len(dates))
treatment_lift = np.where(dates >= pd.Timestamp("2024-02-15"), 12.0, 0.0)

data = pd.DataFrame(
    {
        "date": dates,
        "control": control,
        "test": 20.0 + 1.02 * control + noise + treatment_lift,
    }
)
```

Here `control` is $x_t$ and `test` is $y_t$. The pretest regression estimates
the intercept $\beta_0$ as Python `alpha`, the slope $\beta_1$ as `beta`, and
the residual standard deviation $\sigma$ as `sigma`.

## Fit and summarize

By default, `test_end` is exclusive. This example therefore analyzes
2024-02-15 through 2024-02-28.

```python
model = TBRAnalysis(level=0.80, threshold=0.0)
model.fit(
    data=data,
    time_col="date",
    control_col="control",
    test_col="test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-02-15"),
    test_end=pd.Timestamp("2024-03-01"),
)

summary = model.summarize()
print(f"Cumulative effect: {summary.estimate:.2f}")
print(f"80% credible interval: [{summary.lower:.2f}, {summary.upper:.2f}]")
print(f"Posterior P(effect > {summary.threshold:g}): {summary.prob:.3f}")
```

The final cumulative effect $\hat{\Delta}(T)$ is `estimate`.
`se` is the standard error. `precision` is the credible-interval half-width,
while `lower` and `upper` are the credible-interval bounds.
`prob` is the posterior probability
$P(\Delta(T)>\theta\mid\mathrm{data})$, where `threshold` is $\theta$.

```python
print(f"Standard error: {summary.se:.2f}")
print(f"Precision: {summary.precision:.2f}")
assert np.isclose(summary.precision, (summary.upper - summary.lower) / 2)
```

## Inspect predictions and daily results

```python
prediction_result = model.predict()
prediction_frame = prediction_result.predictions
daily_frame = model.results_
incremental_frame = model.summarize_incremental()

print(prediction_frame[["pred", "predsd"]].head())
print(incremental_frame[["test_day", "estimate", "lower", "upper", "prob"]].tail())
```

`pred` contains test-period counterfactual predictions $\hat{y}_t^*$.
`predsd` contains their prediction standard deviations, including model and
residual uncertainty. In `daily_frame`, `dif` is the pointwise effect
$\phi_t$, `cumdif` is $\Delta(T)$, and legacy column `cumsd` is the
cumulative-effect standard error (Student-t scale).

## One-call alternatives

`fit_summarize()` returns the same `TBRSummaryResult` shape as `summarize()`:

```python
one_call_summary = TBRAnalysis(level=0.80).fit_summarize(
    data,
    "date",
    "control",
    "test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-02-15"),
    test_end=pd.Timestamp("2024-03-01"),
)
print(one_call_summary.estimate)
```

The functional API instead returns a comprehensive `TBRResults` object. Its
scalar aliases are `estimate`, `conf_int_lower`, `conf_int_upper`, and the
legacy property `pvalue`, which is a posterior probability rather than a
frequentist p-value.

```python
from tbr import perform_tbr_analysis

results = perform_tbr_analysis(
    data=data,
    time_col="date",
    control_col="control",
    test_col="test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-02-15"),
    test_end=pd.Timestamp("2024-03-01"),
    level=0.80,
    threshold=0.0,
)
print(results.estimate, results.conf_int_lower, results.conf_int_upper)
print(results.pvalue)
```

Do not confuse `results.summary()`, which is a day-by-day DataFrame, with
`model.summarize()`, which returns one structured `TBRSummaryResult`.

## Boundary and interpretation notes

- `pretest_start` and `test_start` are inclusive.
- `test_end` is exclusive unless `test_end_inclusive=True`.
- `TBRSummaryResult.is_significant()` checks whether `prob` meets a posterior
  probability threshold (0.95 by default). It does not test interval exclusion.
- An interval entirely above zero (`summary.lower > 0`) is a separate
  interval-based criterion.
- Conclusions are conditional on the fitted model, its assumptions, and the
  observed data; inspect diagnostics and domain assumptions before drawing
  causal conclusions.

## Next steps

- [API reference](api_reference.rst)
- [Common patterns](patterns.md)
- [Result objects and schemas](results.md)
- [Mathematical methodology](../mathematical_methodology.md)
