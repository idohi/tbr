"""
Basic TBR Analysis
==================

Run a complete time-based regression workflow using synthetic daily data with
a known simulated treatment effect.
"""

import numpy as np
import pandas as pd

from tbr import TBRAnalysis

# Random number generator with a fixed seed, for reproducibility
rng = np.random.default_rng(114)

# =============================================================================
# 1. Generate Sample Data
# =============================================================================

n_pretest_days = 45
n_test_days = 45
n_days = n_pretest_days + n_test_days

# Create daily data
dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

# Control group: baseline behavior
control = rng.normal(1000, 50, n_days)

# Test group: tracks control with noise, then 1% lift in test period
test = control + rng.normal(0, 25, n_days)
# Known only because the data is simulated; unknown in real experiments
true_effect = 0.01 * test[n_pretest_days:].sum()
test[n_pretest_days:] = test[n_pretest_days:] * 1.01  # 1% lift

# Create DataFrame
data = pd.DataFrame({"date": dates, "control_metric": control, "test_metric": test})

print("=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print(f"Total days: {len(data)}")
print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print("\nFirst 5 rows:")
print(data.head())
print()

# =============================================================================
# 2. Initialize TBR Model
# =============================================================================

# 80% credible intervals; posterior probability that the effect exceeds 0
model = TBRAnalysis(level=0.80, threshold=0.0)

print("=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)
print(f"Credibility level: {model.level * 100:g}%")
print(f"Threshold: {model.threshold}")
print(f"Test end inclusive: {model.test_end_inclusive}")
print()

# =============================================================================
# 3. Fit the Model
# =============================================================================

# Define time periods
pretest_start = pd.Timestamp("2024-01-01")  # First pretest day
test_start = pd.Timestamp("2024-02-15")  # First test day (intervention starts)
# test_end is exclusive by default, so the last test day is 2024-03-30
test_end = pd.Timestamp("2024-03-31")

print("=" * 80)
print("FITTING MODEL")
print("=" * 80)
one_day = pd.Timedelta(days=1)
print(f"Pretest period: {pretest_start.date()} to {(test_start - one_day).date()}")
print(f"Test period: {test_start.date()} to {(test_end - one_day).date()}")
print()

# Fit model
model.fit(
    data=data,
    time_col="date",
    control_col="control_metric",
    test_col="test_metric",
    pretest_start=pretest_start,
    test_start=test_start,
    test_end=test_end,
)

print(f"Model fitted: {model.fitted_}")
print(f"Pretest days: {(model.results_['period'] == 0).sum()}")
print(f"Test days: {(model.results_['period'] == 1).sum()}")
print()

# =============================================================================
# 4. Get Summary Results
# =============================================================================

summary = model.summarize()

print("=" * 80)
print("ANALYSIS RESULTS")
print("=" * 80)
print(f"Cumulative treatment effect: {summary.estimate:.2f}")
print(f"Simulated true effect: {true_effect:.2f}")
print(f"Standard error: {summary.se:.2f}")
print(
    f"{summary.level * 100:g}% credible interval: "
    f"[{summary.lower:.2f}, {summary.upper:.2f}]"
)
print(f"Posterior probability (effect > {summary.threshold:g}): {summary.prob:.3f}")
# The 0.95 cutoff is separate from the credibility level of the interval
print(f"Posterior probability >= 0.95: {summary.meets_probability_threshold(0.95)}")
print()

# Regression parameters
print("Regression Model:")
print(f"  Intercept (α): {summary.alpha:.2f}")
print(f"  Slope (β): {summary.beta:.4f}")
print(f"  Residual Std Error (σ): {summary.sigma:.2f}")
print()

# =============================================================================
# 5. Analyze Subintervals
# =============================================================================

print("=" * 80)
print("SUBINTERVAL ANALYSIS")
print("=" * 80)

# Analyze consecutive 15-day windows of the test period
window_length = 15
for start_day in range(1, n_test_days + 1, window_length):
    end_day = start_day + window_length - 1
    window = model.analyze_subinterval(start_day=start_day, end_day=end_day)
    print(f"Test days {start_day}-{end_day}:")
    print(f"  Effect: {window.estimate:.2f}")
    print(
        f"  {window.ci_level * 100:g}% credible interval: "
        f"[{window.lower:.2f}, {window.upper:.2f}]"
    )
    print(f"  Interval entirely positive: {window.is_positive()}")
    print()

# =============================================================================
# 6. Generate Predictions
# =============================================================================

print("=" * 80)
print("COUNTERFACTUAL PREDICTIONS")
print("=" * 80)

# Get predictions for test period
predictions = model.predict()

print(f"Number of predictions: {predictions.n_predictions}")
print(f"Mean prediction: {predictions.mean_pred:.2f}")
print(f"Mean prediction standard deviation: {predictions.mean_uncertainty:.2f}")
print()
print("First 5 predictions:")
print(predictions.predictions.head())
print()

# =============================================================================
# 7. Access Detailed Results
# =============================================================================

print("=" * 80)
print("DETAILED RESULTS")
print("=" * 80)

# Get full TBR DataFrame
results_df = model.results_

# Show test period results
test_period = results_df[results_df["period"] == 1]
print(f"Test period data shape: {test_period.shape}")
print()
print("Test period summary (first 5 days):")
print(
    test_period[
        ["date", "control_metric", "test_metric", "pred", "dif", "cumdif"]
    ].head()
)
print()

# =============================================================================
# 8. Export Results
# =============================================================================

print("=" * 80)
print("EXPORT OPTIONS")
print("=" * 80)

# Export summary to JSON
summary.to_json("tbr_summary.json")
print("✓ Summary exported to 'tbr_summary.json'")

# Export summary to CSV
summary.to_csv("tbr_summary.csv")
print("✓ Summary exported to 'tbr_summary.csv'")

# Export predictions
predictions.to_json("tbr_predictions.json")
print("✓ Predictions exported to 'tbr_predictions.json'")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
