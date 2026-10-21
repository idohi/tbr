"""
Incremental TBR Analysis
========================

Examine how the estimated cumulative treatment effect evolves over a synthetic
test period and compare non-overlapping subintervals.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tbr import TBRAnalysis

# Random number generator with a fixed seed, for reproducibility
rng = np.random.default_rng(122)

n_pretest_days = 60
n_test_days = 60
window_length = 15  # Length of each subinterval window, in test days

# Generate sample data with ramping effect
dates = pd.date_range("2024-01-01", periods=n_pretest_days + n_test_days)
control = rng.normal(1000, 50, len(dates))

# Test group: tracks control with noise, plus a ramping effect in the test period
test = control + rng.normal(0, 25, len(dates))
# Known only because the data is simulated; unknown in real experiments
no_effect = test.copy()
# 2024-01-01 to 2024-02-29: pretest (no effect)
# 2024-03-01 to 2024-04-29: test period with ramping effect (0% → 2%)
for days_into_test in range(n_test_days):
    i = n_pretest_days + days_into_test
    ramp_factor = 1.0 + (0.02 * days_into_test / (n_test_days - 1))  # Ramp to 2%
    test[i] = test[i] * ramp_factor
true_daily_effect = (test - no_effect)[n_pretest_days:]
true_cumulative_effect = np.cumsum(true_daily_effect)

data = pd.DataFrame({"date": dates, "control": control, "test": test})

print("=" * 80)
print("INCREMENTAL ANALYSIS EXAMPLE")
print("=" * 80)
print(
    f"Scenario: Treatment effect ramps from 0% to 2% over {n_test_days}-day "
    "test period"
)
print()

# =============================================================================
# 1. Fit the Model
# =============================================================================

model = TBRAnalysis(level=0.80)
model.fit(
    data=data,
    time_col="date",
    control_col="control",
    test_col="test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-03-01"),  # First test day
    # test_end is exclusive by default, so the last test day is 2024-04-29
    test_end=pd.Timestamp("2024-04-30"),
)

print(f"Pretest days: {(model.results_['period'] == 0).sum()}")
print(f"Test days: {(model.results_['period'] == 1).sum()}")
print()

# =============================================================================
# 2. Get Incremental Summaries
# =============================================================================

print("=" * 80)
print("INCREMENTAL SUMMARIES")
print("=" * 80)

incremental = model.summarize_incremental()

# Show progression at key points
print(f"Cumulative effect progression ({model.level * 100:g}% credible intervals):")
print()

checkpoints = [1, 15, 30, 45, 60]
for day in checkpoints:
    row = incremental.iloc[day - 1]
    print(
        f"Test day {day:2d}: Effect = {row['estimate']:8.2f}, "
        f"SE = {row['se']:6.2f}, "
        f"CI = [{row['lower']:8.2f}, {row['upper']:8.2f}], "
        f"P(>{model.threshold:g}) = {row['prob']:.3f}"
    )

print()

# %%
# Cumulative effect over the test period
# ---------------------------------------
#
# This retrospective plot summarizes the completed test period. Repeatedly
# checking whether the interval excludes zero is not a valid sequential
# stopping rule.
cumulative_level = incremental["level"].iloc[0]
cumulative_fig, cumulative_ax = plt.subplots(figsize=(9, 5))
cumulative_ax.plot(
    incremental["test_day"],
    incremental["estimate"],
    color="tab:blue",
    linewidth=1.8,
    label="Estimated cumulative effect",
)
cumulative_ax.fill_between(
    incremental["test_day"],
    incremental["lower"],
    incremental["upper"],
    color="tab:blue",
    alpha=0.2,
    label=f"{cumulative_level * 100:g}% credible interval",
)
cumulative_ax.plot(
    incremental["test_day"],
    true_cumulative_effect,
    color="tab:orange",
    linestyle="--",
    linewidth=1.8,
    label="Simulated true cumulative effect",
)
cumulative_ax.axhline(
    0, color="black", linewidth=0.9, linestyle=":", label="Zero effect"
)
cumulative_ax.set(
    title="Cumulative effect over the completed test period",
    xlabel="Test day",
    ylabel="Cumulative effect",
)
cumulative_ax.legend()
cumulative_fig.tight_layout()

# %%
# Subinterval analysis in 15-day windows
# ---------------------------------------

print("=" * 80)
print(f"SUBINTERVAL ANALYSIS IN {window_length}-DAY WINDOWS")
print("=" * 80)

# Each window covers only 15 test days, so its interval is wide relative to the
# effect. Compare the estimates with the simulated true effect: differences
# between windows can be noise, and an 80% interval is expected to miss the
# true effect about 1 time in 5.
windows = []
for start_day in range(1, n_test_days + 1, window_length):
    end_day = start_day + window_length - 1
    result = model.analyze_subinterval(start_day, end_day)
    windows.append(
        {
            "test_days": f"{start_day}-{end_day}",
            "true_effect": true_daily_effect[start_day - 1 : end_day].sum(),
            "effect": result.estimate,
            "lower": result.lower,
            "upper": result.upper,
            "interval_positive": result.is_positive(),
        }
    )
subinterval_level = result.ci_level

windows_df = pd.DataFrame(windows)
print(
    f"\nEffects in {window_length}-day windows "
    f"({model.level * 100:g}% credible intervals):"
)
print(windows_df.round(2).to_string(index=False))
print()

# %%
# Effects in non-overlapping subintervals
# ----------------------------------------
#
# Each point estimate covers a separate 15-day period. The asymmetric error
# bars show that interval's credible limits; the crosses show the simulated
# true effects, which are available only because these data were generated.
window_positions = np.arange(len(windows_df))
lower_errors = windows_df["effect"] - windows_df["lower"]
upper_errors = windows_df["upper"] - windows_df["effect"]
subinterval_fig, subinterval_ax = plt.subplots(figsize=(8, 5))
subinterval_ax.errorbar(
    window_positions,
    windows_df["effect"],
    yerr=np.vstack([lower_errors, upper_errors]),
    color="tab:blue",
    marker="o",
    linestyle="none",
    capsize=4,
    label=f"Estimate and {subinterval_level * 100:g}% credible interval",
)
subinterval_ax.scatter(
    window_positions,
    windows_df["true_effect"],
    color="tab:orange",
    marker="x",
    s=60,
    label="Simulated true effect",
)
subinterval_ax.axhline(
    0, color="black", linewidth=0.9, linestyle=":", label="Zero effect"
)
subinterval_ax.set(
    title=f"Effects in non-overlapping {window_length}-day periods",
    xlabel="Test days",
    ylabel="Subinterval effect",
    xticks=window_positions,
    xticklabels=windows_df["test_days"],
)
subinterval_ax.legend()
subinterval_fig.tight_layout()

# %%
# Period comparison
# -----------------

print("=" * 80)
print("PERIOD COMPARISON")
print("=" * 80)

# Compare the early and late halves of the test period
half = n_test_days // 2
periods = [
    ("Early", 1, half),
    ("Late", half + 1, n_test_days),
]

print("\nComparing different phases:")
period_results = {}
for name, start, end in periods:
    result = model.analyze_subinterval(start, end)
    period_results[name] = result
    print(f"\n{name} (test days {start}-{end}):")
    print(f"  Effect: {result.estimate:.2f}")
    print(
        f"  {result.ci_level * 100:g}% credible interval: "
        f"[{result.lower:.2f}, {result.upper:.2f}]"
    )
    print(f"  Interval entirely positive: {result.is_positive()}")

print()

# %%
# Cumulative growth analysis
# --------------------------

print("=" * 80)
print("CUMULATIVE GROWTH ANALYSIS")
print("=" * 80)

# Show how cumulative effect grows
print(f"\nCumulative effect growth every {window_length} test days:")
for day in range(window_length, n_test_days + 1, window_length):
    row = incremental.iloc[day - 1]
    # Calculate average daily effect
    avg_daily = row["estimate"] / day
    print(
        f"Test day {day:2d}: Total = {row['estimate']:8.2f}, "
        f"Avg/day = {avg_daily:6.2f}"
    )

print()

# %%
# Credible-interval width over time
# ---------------------------------

print("=" * 80)
print("CREDIBLE INTERVAL WIDTH OVER TIME")
print("=" * 80)

# The interval is for the cumulative effect. Its posterior variance,
# T * sigma^2 + T^2 * v, includes terms that scale with the number of test
# days T. See:
# https://tbr.readthedocs.io/en/stable/mathematical_methodology.html#posterior-variance-of-the-cumulative-effect
print(
    "\nCumulative credible interval width and precision "
    "(half-width; smaller is more precise):"
)
for day in checkpoints:
    row = incremental.iloc[day - 1]
    ci_width = row["upper"] - row["lower"]
    print(
        f"Test day {day:2d}: Width = {ci_width:8.2f}, "
        f"Precision = {row['precision']:8.2f}"
    )

print()

# %%
# Export incremental results
# --------------------------

print("=" * 80)
print("EXPORT INCREMENTAL RESULTS")
print("=" * 80)

# Export incremental summaries
incremental.to_csv("incremental_summaries.csv", index=False)
print("✓ Incremental summaries exported to 'incremental_summaries.csv'")

# Export the 15-day window analysis
windows_df.to_csv("window_analysis.csv", index=False)
print(f"✓ {window_length}-day window analysis exported to 'window_analysis.csv'")

print()
print("=" * 80)
print("INCREMENTAL ANALYSIS COMPLETE")
print("=" * 80)
print()
early = period_results["Early"]
late = period_results["Late"]
first_day = incremental.iloc[0]
last_day = incremental.iloc[-1]
print("Summary:")
print(
    f"• Estimated effect: {early.estimate:.2f} in test days 1-{half}, "
    f"{late.estimate:.2f} in test days {half + 1}-{n_test_days}"
)
print(
    f"• Cumulative credible interval width: "
    f"{first_day['upper'] - first_day['lower']:.2f} on test day 1, "
    f"{last_day['upper'] - last_day['lower']:.2f} on test day {len(incremental)}"
)
