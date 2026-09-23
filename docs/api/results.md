# TBR Result Objects

TBR exposes one comprehensive functional result class and three frozen
dataclasses. This page documents their narrative schemas; see the
[API reference](api_reference.rst) for signatures and the
[mathematical methodology](../mathematical_methodology.md) for the complete
notation-to-code mapping.

## Which result is returned?

| Call | Return type | Scope |
|---|---|---|
| `perform_tbr_analysis()` | `TBRResults` | Complete functional analysis |
| `TBRResults.summary()` | `pd.DataFrame` | Cumulative statistics for every test day |
| `TBRAnalysis.summarize()` | `TBRSummaryResult` | Structured final cumulative summary |
| `TBRAnalysis.fit_summarize()` | `TBRSummaryResult` | Fit and return the final summary |
| `TBRAnalysis.predict()` | `TBRPredictionResult` | Counterfactual predictions |
| `TBRAnalysis.fit_predict()` | `TBRPredictionResult` | Fit and return predictions |
| `TBRAnalysis.analyze_subinterval()` | `TBRSubintervalResult` | One inclusive test-day interval |

In particular, `TBRResults.summary()` is not an alias for
`TBRAnalysis.summarize()`: the former returns a multi-row DataFrame, while the
latter returns one `TBRSummaryResult`.

## Shared executable setup

```python
import numpy as np
import pandas as pd

from tbr import TBRAnalysis, perform_tbr_analysis
from tbr.core.results import (
    TBRPredictionResult,
    TBRResults,
    TBRSubintervalResult,
    TBRSummaryResult,
)

rng = np.random.default_rng(11)
dates = pd.date_range("2024-01-01", periods=50)
control = rng.normal(100.0, 8.0, len(dates))
test = (
    4.0
    + 1.1 * control
    + rng.normal(0.0, 2.0, len(dates))
    + np.where(dates >= pd.Timestamp("2024-02-05"), 3.0, 0.0)
)
data = pd.DataFrame({"date": dates, "control": control, "test": test})
boundaries = {
    "pretest_start": pd.Timestamp("2024-01-01"),
    "test_start": pd.Timestamp("2024-02-05"),
    "test_end": pd.Timestamp("2024-02-20"),
}

model = TBRAnalysis(level=0.90, threshold=0.0).fit(
    data, "date", "control", "test", **boundaries
)
summary = model.summarize()
prediction_result = model.predict()
subinterval = model.analyze_subinterval(1, 7)
results = perform_tbr_analysis(
    data,
    "date",
    "control",
    "test",
    level=0.90,
    threshold=0.0,
    **boundaries,
)
```

## `TBRResults`

`TBRResults` is returned by `perform_tbr_analysis()`. It provides scalar
statistics, time-indexed Series, model parameters, a comprehensive daily
DataFrame, and a cumulative summary DataFrame.

### Properties

| Property | Type | Meaning |
|---|---|---|
| `control` | `pd.Series` | Control values $x_t$, indexed by pretest/test time |
| `test` | `pd.Series` | Observed test values $y_t$, indexed by pretest/test time |
| `fittedvalues` | `pd.Series` | Pretest fitted values $\hat{y}_t$ |
| `predictions` | `pd.Series` | Test counterfactual predictions $\hat{y}_t^*$ |
| `resid` | `pd.Series` | Pretest residuals |
| `effects` | `pd.Series` | Pointwise test effects $\phi_t$ |
| `cumulative_effect` | `pd.Series` | Running cumulative effect $\Delta(T)$ |
| `prediction_se` | `pd.Series` | Counterfactual prediction standard deviation |
| `cumulative_se` | `pd.Series` | Cumulative effect standard errors |
| `estimate` | `float` | Final cumulative-effect estimate $\hat{\Delta}(T)$ |
| `conf_int_lower` | `float` | Final lower credible bound |
| `conf_int_upper` | `float` | Final upper credible bound |
| `pvalue` | `float` | Legacy name for posterior threshold-exceedance probability |
| `n_pretest` | `int` | Number of pretest observations |
| `n_test` | `int` | Number of test observations |
| `n_test_days` | `int` | Test-period length; currently equal to `n_test` |
| `alpha` | `float` | Regression intercept $\beta_0$ |
| `beta` | `float` | Regression slope $\beta_1$ |
| `sigma` | `float` | Residual standard deviation $\sigma$ |
| `model_params` | `dict[str, float \| int]` | Copy of the fitted parameter mapping below |

`pvalue` is not a frequentist p-value. It contains
$P(\Delta(T)>\theta\mid\mathrm{data})$ for the configured threshold $\theta$.

### `model_params` keys

| Key | Type | Meaning |
|---|---|---|
| `alpha` | `float` | Regression intercept $\beta_0$ |
| `beta` | `float` | Regression slope $\beta_1$ |
| `sigma` | `float` | Residual standard deviation $\sigma$ |
| `var_alpha` | `float` | Variance of the intercept estimate |
| `var_beta` | `float` | Variance of the slope estimate |
| `cov_alpha_beta` | `float` | Intercept/slope covariance |
| `degrees_freedom` | `int` | Regression/Student's-$t$ degrees of freedom $\nu$ |
| `n_pretest` | `int` | Number of pretest observations $n$ |
| `pretest_x_mean` | `float` | Mean pretest control value $\bar{x}$ |

### `summary()` DataFrame

`results.summary()` returns one row per cumulative test day. `test_day` is
integer; every other column is `float64`, including `t_dist_df`.

| Column | Type | Meaning |
|---|---|---|
| `test_day` | integer | One-based cumulative test-day number |
| `estimate` | `float64` | Cumulative-effect estimate $\hat{\Delta}(T)$ |
| `precision` | `float64` | Credible-interval half-width |
| `lower` | `float64` | Lower credible bound |
| `upper` | `float64` | Upper credible bound |
| `se` | `float64` | Standard error |
| `level` | `float64` | Credibility level |
| `thres` | `float64` | Legacy column name for effect `threshold` $\theta$ |
| `prob` | `float64` | Posterior probability that the effect exceeds `thres` |
| `alpha` | `float64` | Regression intercept $\beta_0$ |
| `beta` | `float64` | Regression slope $\beta_1$ |
| `alpha_beta_cov` | `float64` | Legacy column name for `cov_alpha_beta` |
| `var_alpha` | `float64` | Variance of the intercept estimate |
| `var_beta` | `float64` | Variance of the slope estimate |
| `sigma` | `float64` | Residual standard deviation |
| `t_dist_df` | `float64` | Legacy column name for `degrees_freedom` |

The explicit aliases at this boundary are:
`threshold` → `thres`, `cov_alpha_beta` → `alpha_beta_cov`, and
`degrees_freedom` → `t_dist_df`.

### `tbr_dataframe()` DataFrame

This method returns a copy containing original source columns plus normalized
TBR columns. Availability differs by period (`-1` baseline, `0` pretest,
`1` test).

| Column | Type | Meaning and availability |
|---|---|---|
| original source columns | source-dependent | Preserved input columns where pandas concatenation permits |
| `period` | integer | `-1` baseline, `0` pretest, `1` test |
| `y` | source-dependent | Normalized observed test metric |
| `x` | source-dependent | Normalized control metric |
| `pred` | floating-point | Pretest fitted value or test counterfactual; `NaN` in baseline |
| `predsd` | floating-point | Test prediction SD; `0.0` in pretest and `NaN` in baseline |
| `dif` | floating-point | Pretest residual or pointwise test effect; `NaN` in baseline |
| `cumdif` | floating-point | Cumulative test effect; `NaN` outside test |
| `cumsd` | floating-point | Cumulative-effect standard error (Student-t scale) under a legacy column name; `0.0` in pretest and `NaN` in baseline |
| `estsd` | floating-point | Fitted-value SE in pretest; `NaN` in baseline and test |

```python
daily_summary = results.summary()
tbr_frame = results.tbr_dataframe()

assert daily_summary.shape[0] == results.n_test_days
assert tbr_frame.loc[tbr_frame["period"] == 1, "estsd"].isna().all()
```

### `conf_int(level=None)` DataFrame

`results.conf_int()` returns the final cumulative-effect credible interval as a
single-row DataFrame. The optional `level` argument is a credibility level. If
it is omitted or `None`, the method uses the level configured by
`perform_tbr_analysis()`. If a different value is supplied, the method
recalculates the bounds from the final cumulative-effect estimate, standard
error, and fitted degrees of freedom.

| Column | Type | Meaning |
|---|---|---|
| `lower` | `float64` | Lower credible bound at the requested or configured credibility level |
| `upper` | `float64` | Upper credible bound at the requested or configured credibility level |

```python
configured_interval = results.conf_int()
interval_95 = results.conf_int(level=0.95)

assert list(configured_interval.columns) == ["lower", "upper"]
assert configured_interval.dtypes.to_dict() == {
    "lower": np.dtype("float64"),
    "upper": np.dtype("float64"),
}
assert interval_95.loc[0, "lower"] <= results.estimate <= interval_95.loc[0, "upper"]
```

## `TBRSummaryResult`

This frozen dataclass represents the final cumulative test-period summary.

| Attribute | Type | Meaning |
|---|---|---|
| `estimate` | `float` | Cumulative-effect estimate $\hat{\Delta}(T)$ |
| `lower` | `float` | Lower credible bound |
| `upper` | `float` | Upper credible bound |
| `se` | `float` | Standard error |
| `prob` | `float` | Posterior threshold-exceedance probability |
| `precision` | `float` | Credible-interval half-width |
| `level` | `float` | Credibility level |
| `threshold` | `float` | Effect threshold $\theta$ |
| `alpha` | `float` | Regression intercept $\beta_0$ |
| `beta` | `float` | Regression slope $\beta_1$ |
| `sigma` | `float` | Residual standard deviation |
| `var_alpha` | `float` | Variance of the intercept estimate |
| `var_beta` | `float` | Variance of the slope estimate |
| `cov_alpha_beta` | `float` | Intercept/slope covariance |
| `degrees_freedom` | `int` | Student's-$t$ degrees of freedom $\nu$ |

`is_significant(probability_threshold=0.95)` returns whether `prob` is at least
that threshold. It does not inspect the credible interval.

```python
probability_threshold_met = summary.is_significant()
interval_excludes_zero = summary.lower > 0 or summary.upper < 0
summary_dict = summary.to_dict()
summary_frame = summary.to_dataframe()

assert np.isclose(summary.precision, (summary.upper - summary.lower) / 2)
assert probability_threshold_met == (summary.prob >= 0.95)
```

`to_dict()` and `to_dataframe()` use the attribute names above, including
`cov_alpha_beta`, `threshold`, and `degrees_freedom`.

## `TBRPredictionResult`

This frozen dataclass contains predictions and the parameters used to produce
them.

| Attribute/property | Type | Meaning |
|---|---|---|
| `predictions` | `pd.DataFrame` | Prediction table described below |
| `n_predictions` | `int` | Number of prediction rows |
| `model_params` | `dict[str, float \| int]` | Fitted parameter mapping described below |
| `control_values` | `np.ndarray` | Control values used for prediction |
| `mean_pred` | `float` | Mean of `pred` |
| `mean_uncertainty` | `float` | Mean of `predsd` |

### Prediction columns

| Column | Type | Meaning |
|---|---|---|
| `pred` | floating-point | Counterfactual prediction $\hat{y}_t^*$ |
| `predsd` | floating-point | Counterfactual prediction standard deviation, including coefficient and residual uncertainty |

### Prediction `model_params` keys

| Key | Type | Meaning |
|---|---|---|
| `alpha` | `float` | Regression intercept $\beta_0$ |
| `beta` | `float` | Regression slope $\beta_1$ |
| `sigma` | `float` | Residual standard deviation |
| `var_alpha` | `float` | Variance of the intercept estimate |
| `var_beta` | `float` | Variance of the slope estimate |
| `cov_alpha_beta` | `float` | Intercept/slope covariance |
| `degrees_freedom` | `int` | Degrees of freedom $\nu$ |
| `pretest_x_mean` | `float` | Mean pretest control value $\bar{x}$ |
| `pretest_sum_x_squared_deviations` | `float` | Pretest $S_{xx}$ |

```python
assert isinstance(prediction_result, TBRPredictionResult)
assert list(prediction_result.predictions.columns) == ["pred", "predsd"]
assert prediction_result.n_predictions == len(prediction_result.predictions)
```

## `TBRSubintervalResult`

This frozen dataclass describes inclusive one-based test days $a$ through $b$.

| Attribute | Type | Meaning |
|---|---|---|
| `estimate` | `float` | Subinterval effect $\Delta(a,b)$ |
| `lower` | `float` | Lower credible bound |
| `upper` | `float` | Upper credible bound |
| `se` | `float` | Credible-interval half-width under a legacy field name |
| `ci_level` | `float` | Credibility level |
| `start_day` | `int` | Inclusive one-based start $a$ |
| `end_day` | `int` | Inclusive one-based end $b$ |
| `n_days` | `int` | Interval length $T_s=b-a+1$ |

```python
assert isinstance(subinterval, TBRSubintervalResult)
assert subinterval.n_days == subinterval.end_day - subinterval.start_day + 1
assert np.isclose(subinterval.se, (subinterval.upper - subinterval.lower) / 2)

contains_zero = subinterval.contains_zero()
entirely_positive = subinterval.is_positive()
entirely_negative = subinterval.is_negative()
```

These methods are interval criteria only; they do not compute a posterior
probability threshold.

## Conversion and export

The dataclasses expose `to_dict()`, `to_json()`, and `to_csv()`.
`TBRSummaryResult` additionally exposes `to_dataframe()`;
`TBRPredictionResult.predictions` is already a DataFrame.

```python
in_memory_report = {
    "summary": summary.to_dict(),
    "prediction": prediction_result.to_dict(),
    "subinterval": subinterval.to_dict(),
}
```

The following calls are illustrative because they write files:

```python
summary.to_json("summary.json")
summary.to_csv("summary.csv", index=False)
prediction_result.to_csv("predictions.csv", index=False)
subinterval.to_json("subinterval.json")
```

## See also

- [Quickstart](quickstart.md)
- [Common patterns](patterns.md)
- [API reference](api_reference.rst)
- [Mathematical methodology](../mathematical_methodology.md)
