# Common Patterns and Best Practices

These patterns build on the [quickstart](quickstart.md). They emphasize valid
API calls and careful interpretation; the
[mathematical methodology](../mathematical_methodology.md) is authoritative for
notation and derivations.

## Shared example setup

The executable snippets below use this reproducible fitted model:

```python
import numpy as np
import pandas as pd

from tbr import TBRAnalysis

rng = np.random.default_rng(7)
dates = pd.date_range("2024-01-01", periods=75)
control = rng.normal(500.0, 20.0, len(dates))
test_start = pd.Timestamp("2024-03-01")
test = (
    15.0
    + 0.98 * control
    + rng.normal(0.0, 5.0, len(dates))
    + np.where(dates >= test_start, 6.0, 0.0)
)
data = pd.DataFrame({"date": dates, "control": control, "test": test})

model = TBRAnalysis(level=0.90, threshold=0.0)
model.fit(
    data,
    "date",
    "control",
    "test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=test_start,
    test_end=pd.Timestamp("2024-03-16"),
)
```

## Data preparation

### Use supported time types

Datetime columns must use a pandas datetime dtype, and datetime boundaries must
be `pd.Timestamp` values. Integer and float time columns use matching scalar
boundaries.

```python
datetime_data = data.assign(date=pd.to_datetime(data["date"])).sort_values("date")

integer_data = data.assign(day=np.arange(len(data), dtype=np.int64))
integer_model = TBRAnalysis().fit(
    integer_data,
    "day",
    "control",
    "test",
    pretest_start=0,
    test_start=60,
    test_end=75,
)

float_data = data.assign(time=np.arange(len(data), dtype=float))
float_model = TBRAnalysis().fit(
    float_data,
    "time",
    "control",
    "test",
    pretest_start=0.0,
    test_start=60.0,
    test_end=75.0,
)
```

The required columns must be non-null, metric columns must be numeric, and the
time, control, and test column names must be distinct. Sort and validate source
data before fitting:

```python
prepared = data.sort_values("date").reset_index(drop=True)
required = ["date", "control", "test"]
if prepared[required].isna().any().any():
    raise ValueError("TBR input contains missing required values")
if not prepared["date"].is_monotonic_increasing:
    raise ValueError("Time values must be ordered")
```

For file-based workflows, loading and aggregation are context-specific. For
example, after loading a CSV into `raw_data`, daily aggregation could be:

```python
# Illustrative: raw_data must already contain these columns.
raw_data = data.rename(
    columns={"control": "control_metric", "test": "test_metric"}
)
daily = (
    raw_data.groupby("date", as_index=False)[["control_metric", "test_metric"]]
    .sum()
    .sort_values("date")
)
```

## Configuration and boundaries

`level` is the credibility level $1-\alpha$, and `threshold` is the effect
threshold $\theta$ used in $P(\Delta(T)>\theta\mid\mathrm{data})$.
Mathematical $\alpha$ denotes tail probability; Python `alpha` is unrelated
and stores the regression intercept $\beta_0$.

```python
screening_model = TBRAnalysis(level=0.80, threshold=0.0)
decision_model = TBRAnalysis(level=0.95, threshold=1000.0)
```

Choose these values from the decision context before examining results. A
higher credibility level produces a wider interval for the same fitted model.
A nonzero threshold asks whether the cumulative effect exceeds a practically
meaningful value; it does not change the effect estimate.

By default, `test_end` is exclusive. Set `test_end_inclusive=True` only when
the boundary itself should be analyzed:

```python
same_day = TBRAnalysis(test_end_inclusive=True).fit(
    data,
    "date",
    "control",
    "test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-03-01"),
    test_end=pd.Timestamp("2024-03-01"),
)
```

## Compare credibility levels

Refit after changing configuration because `set_params()` clears fitted state.

```python
boundaries = {
    "pretest_start": pd.Timestamp("2024-01-01"),
    "test_start": pd.Timestamp("2024-03-01"),
    "test_end": pd.Timestamp("2024-03-16"),
}

comparison_rows = []
for level in (0.80, 0.90, 0.95):
    summary = TBRAnalysis(level=level).fit_summarize(
        data, "date", "control", "test", **boundaries
    )
    comparison_rows.append(
        {
            "level": level,
            "estimate": summary.estimate,
            "lower": summary.lower,
            "upper": summary.upper,
            "precision": summary.precision,
        }
    )

comparison = pd.DataFrame(comparison_rows)
```

## Separate probability and interval criteria

`TBRSummaryResult.is_significant()` is a convenience check on posterior
probability. Its argument is a probability threshold and defaults to 0.95.
Interval exclusion is a different criterion.

```python
summary = model.summarize()

probability_threshold_met = summary.is_significant(0.95)
interval_excludes_zero = summary.lower > 0 or summary.upper < 0
interval_entirely_positive = summary.lower > 0

print(f"Posterior probability threshold met: {probability_threshold_met}")
print(f"Credible interval excludes zero: {interval_excludes_zero}")
print(f"Credible interval entirely positive: {interval_entirely_positive}")
```

`summary.prob` always refers to the configured `summary.threshold`. Avoid
describing either criterion as proof of an effect. Report the estimate,
credibility level, interval, posterior probability, and threshold together.

```python
print(
    f"Cumulative effect {summary.estimate:.2f}; "
    f"{summary.level:.0%} credible interval "
    f"[{summary.lower:.2f}, {summary.upper:.2f}]; "
    f"P(effect > {summary.threshold:g} | data)={summary.prob:.3f}"
)
```

## Monitor cumulative results

`summarize_incremental()` returns one row for each cumulative test day. The
`se` column is the standard error of the estimate; `precision` is the
credible-interval half-width.

```python
incremental = model.summarize_incremental()
first_positive = incremental.loc[incremental["lower"] > 0, "test_day"]
if not first_positive.empty:
    print(f"Interval first entirely positive on day {int(first_positive.iloc[0])}")
```

Repeatedly examining the same evolving experiment can change operating
characteristics. Treat retrospective stopping-day searches and multiple
subintervals as exploratory unless the monitoring and multiplicity rules were
specified in advance.

## Analyze subintervals

Subinterval bounds are inclusive and one-based: `start_day=1` is the first test
row. The returned `TBRSubintervalResult.se` stores the credible-interval
half-width under a legacy field name.

```python
weekly_rows = []
for start_day in (1, 8):
    end_day = min(start_day + 6, len(model.summaries_))
    interval = model.analyze_subinterval(start_day, end_day)
    weekly_rows.append(
        {
            "start_day": interval.start_day,
            "end_day": interval.end_day,
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "interval_half_width": interval.se,
        }
    )

weekly = pd.DataFrame(weekly_rows)
```

Comparing point estimates across intervals is descriptive. It is not itself a
posterior test of the difference between intervals.

## Work with predictions and detailed outputs

```python
prediction_result = model.predict()
predictions = prediction_result.predictions
test_rows = model.results_.loc[model.results_["period"] == 1]

assert list(predictions.columns) == ["pred", "predsd"]
assert test_rows["estsd"].isna().all()
```

`predsd` is the counterfactual prediction standard deviation. It differs from
legacy column `cumsd`, the cumulative-effect standard error (Student-t scale),
and from `precision`.

## Export results

Structured result objects provide JSON/CSV helpers. The examples below are
illustrative because they write files:

```python
# File-writing examples:
summary.to_json("summary.json")
summary.to_csv("summary.csv", index=False)
prediction_result.to_csv("predictions.csv", index=False)
model.analyze_subinterval(1, 7).to_json("week_1.json")
```

For in-memory composition, use dictionaries and DataFrames:

```python
report = {
    "summary": summary.to_dict(),
    "predictions": prediction_result.to_dict(),
    "first_week": model.analyze_subinterval(1, 7).to_dict(),
}
summary_frame = summary.to_dataframe()
```

## See also

- [Quickstart](quickstart.md)
- [Result objects and schemas](results.md)
- [API reference](api_reference.rst)
- [Mathematical methodology](../mathematical_methodology.md)
