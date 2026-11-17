# Migrating from 0.1.x to 0.2.0

Version 0.2.0 is a breaking pre-1.0 release. It replaces generic
day-specific names with terminology that applies to daily, hourly, weekly, and
other regularly observed data. Statistical calculations and positional
selection behavior are unchanged.

## Subinterval analysis

Subinterval bounds remain inclusive, one-based positions among test-period
observations. Only their names change:

- `start_day` becomes `start_position`;
- `end_day` becomes `end_position`;
- `n_days` and the summary column `days` become `n_observations`; and
- `avg_daily_effect` becomes `average_effect_per_observation`.

The callable signatures change from:

```python
compute_interval_estimate_and_ci(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_day: int,
    end_day: int,
    ci_level: float,
) -> Dict[str, float]

TBRAnalysis.analyze_subinterval(
    start_day: int,
    end_day: int,
    ci_level: Optional[float] = None,
) -> TBRSubintervalResult
```

to:

```python
compute_interval_estimate_and_ci(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_position: int,
    end_position: int,
    ci_level: float,
) -> Dict[str, float]

TBRAnalysis.analyze_subinterval(
    start_position: int,
    end_position: int,
    ci_level: Optional[float] = None,
) -> TBRSubintervalResult
```

The compute signature applies to every supported route:

- `tbr.compute_interval_estimate_and_ci`;
- `tbr.analysis.compute_interval_estimate_and_ci`;
- `tbr.analysis.subinterval.compute_interval_estimate_and_ci`;
- `tbr.core.compute_interval_estimate_and_ci`;
- `tbr.core.prediction.compute_interval_estimate_and_ci`;
- `tbr.core.effects.compute_interval_estimate_and_ci`; and
- `tbr.functional.tbr_functions.compute_interval_estimate_and_ci`.

The direct validator likewise becomes
`tbr.analysis.validate_subinterval_parameters(tbr_df, tbr_summary,
start_position, end_position, ci_level) -> None`.

Before:

```python
result = model.analyze_subinterval(start_day=1, end_day=7)
print(result.start_day, result.end_day, result.n_days)
```

After:

```python
result = model.analyze_subinterval(start_position=1, end_position=7)
print(
    result.start_position,
    result.end_position,
    result.n_observations,
)
```

The same keyword change applies to every supported
`compute_interval_estimate_and_ci` route and to
`tbr.analysis.validate_subinterval_parameters`. The `intervals` argument of
`analyze_multiple_subintervals` and `create_subinterval_summary` remains a
list of integer pairs; each pair now has the documented meaning
`(start_position, end_position)`.

`TBRSubintervalResult` now serializes fields in this order:

```text
estimate, lower, upper, se, precision, ci_level,
start_position, end_position, n_observations
```

`create_subinterval_summary()` now returns columns in this order:

```text
interval, start_position, end_position, n_observations, estimate, se,
precision, lower, upper, interval_excludes_threshold,
average_effect_per_observation, ci_level
```

No temporary aliases or duplicate columns are provided. Calls using removed
keywords raise the normal Python `TypeError`, and removed result attributes
raise `AttributeError`.

Trusted `TBRSubintervalResult` pickle files created by 0.1.x remain loadable.
Their old field names are migrated to canonical names during deserialization;
newly saved objects contain only canonical fields. As with all Python pickle
files, load them only from trusted sources.
