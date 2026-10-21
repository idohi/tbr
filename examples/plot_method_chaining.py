"""
Method Chaining
===============

Use the fluent ``TBRAnalysis`` API for configuration, fitting, prediction,
copying, and result export.
"""

import numpy as np
import pandas as pd

from tbr import TBRAnalysis

# Random number generator with a fixed seed, for reproducibility
rng = np.random.default_rng(114)

# Generate sample data
n_pretest_days = 45
n_test_days = 45
n_days = n_pretest_days + n_test_days

dates = pd.date_range("2024-01-01", periods=n_days)
control = rng.normal(1000, 50, n_days)
test = control + rng.normal(0, 25, n_days)
test[n_pretest_days:] = test[n_pretest_days:] * 1.01  # 1% lift in the test period

data = pd.DataFrame({"date": dates, "control": control, "test": test})

print("=" * 80)
print("METHOD CHAINING EXAMPLES")
print("=" * 80)
print()

# =============================================================================
# 1. One-Liner Analysis
# =============================================================================

print("1. ONE-LINER ANALYSIS")
print("-" * 40)

# Complete analysis in one line
summary = TBRAnalysis().fit_summarize(
    data,
    "date",
    "control",
    "test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-02-15"),
    test_end=pd.Timestamp("2024-03-31"),
)

print(f"Cumulative effect: {summary.estimate:.2f}")
print(
    f"{summary.level * 100:g}% credible interval: "
    f"[{summary.lower:.2f}, {summary.upper:.2f}]"
)
print(f"Posterior probability (effect > {summary.threshold:g}): {summary.prob:.3f}")
# The 0.95 cutoff is separate from the credibility level of the interval
print(f"Posterior probability >= 0.95: {summary.meets_probability_threshold(0.95)}")
print()

# =============================================================================
# 2. Configuration and Fit Chaining
# =============================================================================

print("2. CONFIGURATION AND FIT CHAINING")
print("-" * 40)

# Chain configuration, fitting, and summary
# prob is the posterior probability that the cumulative effect exceeds the
# threshold (100 here, for illustration)
configured_summary = (
    TBRAnalysis()
    .set_params(level=0.90, threshold=100.0)
    .fit(
        data,
        "date",
        "control",
        "test",
        pretest_start=pd.Timestamp("2024-01-01"),
        test_start=pd.Timestamp("2024-02-15"),
        test_end=pd.Timestamp("2024-03-31"),
    )
    .summarize()
)

print(f"Cumulative effect: {configured_summary.estimate:.2f}")
print(
    f"{configured_summary.level * 100:g}% credible interval: "
    f"[{configured_summary.lower:.2f}, {configured_summary.upper:.2f}]"
)
print(
    f"Posterior probability (effect > {configured_summary.threshold:g}): "
    f"{configured_summary.prob:.3f}"
)
print()

# =============================================================================
# 3. Copy, Configure, and Analyze
# =============================================================================

print("3. COPY, CONFIGURE, AND ANALYZE")
print("-" * 40)

# Base model
base_model = TBRAnalysis(level=0.80, threshold=0.0)

# Create variant with different configuration
variant_summary = (
    base_model.copy()
    .set_params(level=0.95)
    .fit_summarize(
        data,
        "date",
        "control",
        "test",
        pretest_start=pd.Timestamp("2024-01-01"),
        test_start=pd.Timestamp("2024-02-15"),
        test_end=pd.Timestamp("2024-03-31"),
    )
)

print(f"Variant cumulative effect: {variant_summary.estimate:.2f}")
print(
    f"{variant_summary.level * 100:g}% credible interval: "
    f"[{variant_summary.lower:.2f}, {variant_summary.upper:.2f}]"
)
print()

# =============================================================================
# 4. Fit and Predict Chain
# =============================================================================

print("4. FIT AND PREDICT CHAIN")
print("-" * 40)

# Fit and predict in one statement
predictions = (
    TBRAnalysis()
    .fit(
        data,
        "date",
        "control",
        "test",
        pretest_start=pd.Timestamp("2024-01-01"),
        test_start=pd.Timestamp("2024-02-15"),
        test_end=pd.Timestamp("2024-03-31"),
    )
    .predict()
)

print(f"Mean prediction: {predictions.mean_pred:.2f}")
print(f"Mean prediction standard deviation: {predictions.mean_uncertainty:.2f}")
print()

# =============================================================================
# 5. Comprehensive Chaining Workflow
# =============================================================================

print("5. COMPREHENSIVE CHAINING WORKFLOW")
print("-" * 40)

# Complete workflow with chaining
model = (
    TBRAnalysis()
    .set_params(level=0.90)
    .fit(
        data,
        "date",
        "control",
        "test",
        pretest_start=pd.Timestamp("2024-01-01"),
        test_start=pd.Timestamp("2024-02-15"),
        test_end=pd.Timestamp("2024-03-31"),
    )
)

# Now access different results
summary = model.summarize()
predictions = model.predict()
# analyze_subinterval() uses the model's level unless ci_level is given
first_window = model.analyze_subinterval(1, 15)

print(
    f"Cumulative effect: {summary.estimate:.2f}, "
    f"{summary.level * 100:g}% credible interval: "
    f"[{summary.lower:.2f}, {summary.upper:.2f}]"
)
print(
    f"Test days 1-15 effect: {first_window.estimate:.2f}, "
    f"{first_window.ci_level * 100:g}% credible interval: "
    f"[{first_window.lower:.2f}, {first_window.upper:.2f}]"
)
print(f"Mean prediction: {predictions.mean_pred:.2f}")
print()

# =============================================================================
# 6. Comparing Multiple Configurations
# =============================================================================

print("6. COMPARING MULTIPLE CONFIGURATIONS")
print("-" * 40)

# Compare different credibility levels
# The level changes only the credible interval: the estimate is the same at
# every level, and a higher level gives a wider interval.
levels = [0.80, 0.90, 0.95, 0.99]
results = []

for level in levels:
    level_summary = (
        TBRAnalysis()
        .set_params(level=level)
        .fit_summarize(
            data,
            "date",
            "control",
            "test",
            pretest_start=pd.Timestamp("2024-01-01"),
            test_start=pd.Timestamp("2024-02-15"),
            test_end=pd.Timestamp("2024-03-31"),
        )
    )
    results.append(
        {
            "level": level,
            "estimate": level_summary.estimate,
            "lower": level_summary.lower,
            "upper": level_summary.upper,
        }
    )

comparison_df = pd.DataFrame(results)
print(comparison_df.round(2).to_string(index=False))
print()

# =============================================================================
# 7. Parameter Inspection
# =============================================================================

print("7. PARAMETER INSPECTION")
print("-" * 40)

model = TBRAnalysis(level=0.90, threshold=10.0)
print("Initial parameters:", model.get_params())

# Update and inspect
model.set_params(level=0.95)
print("After update:", model.get_params())
print()

# =============================================================================
# 8. Sklearn-Style Workflow
# =============================================================================

print("8. SKLEARN-STYLE WORKFLOW")
print("-" * 40)

# Use sklearn-compatible get_params/set_params
estimator = TBRAnalysis()
params = estimator.get_params()
print(f"Default params: {params}")

# Clone with different params
estimator_clone = estimator.copy()
estimator_clone.set_params(level=0.95, threshold=5.0)

print(f"Clone params: {estimator_clone.get_params()}")
print(f"Original params (unchanged): {estimator.get_params()}")
print()

# =============================================================================
# 9. Property Access Chaining
# =============================================================================

print("9. PROPERTY ACCESS CHAINING")
print("-" * 40)

# Chain fit and access convenience properties
model = TBRAnalysis().fit(
    data,
    "date",
    "control",
    "test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-02-15"),
    test_end=pd.Timestamp("2024-03-31"),
)

# Access different properties
# final_effect and final_summary cover the full test period
final = model.final_summary  # Same result as model.summarize()
print(f"Final cumulative effect: {model.final_effect:.2f}")
print(
    f"{final.level * 100:g}% credible interval: "
    f"[{final.lower:.2f}, {final.upper:.2f}]"
)
print(f"Fitted: {model.fitted_}")
print(f"Test days: {len(model.summaries_)}")  # summaries_ has one row per test day
print()

# =============================================================================
# 10. One-Liner with Export
# =============================================================================

print("10. ONE-LINER WITH EXPORT")
print("-" * 40)

# Analyze and export in minimal code
summary = TBRAnalysis().fit_summarize(
    data,
    "date",
    "control",
    "test",
    pretest_start=pd.Timestamp("2024-01-01"),
    test_start=pd.Timestamp("2024-02-15"),
    test_end=pd.Timestamp("2024-03-31"),
)
summary.to_json("quick_analysis.json")

print(f"Cumulative effect: {summary.estimate:.2f} (saved to quick_analysis.json)")
print()

print("=" * 80)
print("METHOD CHAINING COMPLETE")
print("=" * 80)
print()
print("Key Takeaways:")
print("• Use fit_summarize() for one-liner analysis")
print("• Chain set_params() → fit() → summarize() for custom configs")
print("• Use copy() to create independent model variants")
print("• Access final_effect and final_summary for quick results")
print("• Compatible with sklearn-style get_params/set_params")
