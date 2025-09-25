"""
Comprehensive mathematical validation tests for TBR core inference module.

This module provides rigorous mathematical validation for the core inference functions
implemented in Task 4.3. Tests ensure mathematical correctness, cross-validation
with scipy.stats implementations, and proper statistical properties.

Test Categories
---------------
1. T-statistic calculation mathematical validation
2. P-value computation mathematical validation
3. Posterior probability calculation mathematical validation
4. Credible interval estimation mathematical validation
5. Critical value calculation mathematical validation
6. Cross-validation with scipy.stats implementations
7. Statistical property verification
8. Edge case mathematical behavior
9. Numerical precision validation

Mathematical Validation
-----------------------
All tests validate against known statistical formulas and cross-check with
scipy.stats implementations to ensure mathematical correctness. Tests verify:

- Mathematical formula correctness against statistical theory
- Cross-validation with scipy.stats for identical results
- Statistical property preservation and relationships
- Numerical stability and precision
- Edge case behavior and boundary conditions
- Integration consistency across inference functions
"""

import numpy as np
from scipy import stats

from tbr.core.inference import (
    calculate_credible_interval,
    calculate_critical_value,
    calculate_p_value,
    calculate_posterior_probability,
    calculate_t_statistic,
)


class TestTStatisticMathematical:
    """Mathematical validation tests for t-statistic calculation."""

    def test_mathematical_formula_correctness(self):
        """Test that t-statistic formula is implemented correctly."""
        # Test cases with known expected values
        test_cases = [
            # (estimate, standard_error, null_value, expected_t)
            (10.0, 2.0, 0.0, 5.0),  # (10 - 0) / 2 = 5
            (15.5, 3.1, 12.0, 1.1290322580645162),  # (15.5 - 12) / 3.1 = 1.129032258...
            (
                -8.2,
                1.5,
                -5.0,
                -2.1333333333333333,
            ),  # (-8.2 - (-5)) / 1.5 = -2.133333333...
            (0.0, 4.0, 0.0, 0.0),  # (0 - 0) / 4 = 0
            (25.7, 0.1, 25.0, 7.0),  # (25.7 - 25) / 0.1 = 7
        ]

        for estimate, se, null_val, expected in test_cases:
            result = calculate_t_statistic(estimate, se, null_val)
            assert (
                abs(result - expected) < 1e-10
            ), f"T-statistic calculation failed: got {result}, expected {expected}"

    def test_mathematical_properties(self):
        """Test mathematical properties of t-statistic."""
        estimate = 12.5
        se = 3.2
        null_value = 8.0

        t_stat = calculate_t_statistic(estimate, se, null_value)

        # Property 1: Sign should match sign of (estimate - null_value)
        expected_sign = np.sign(estimate - null_value)
        actual_sign = np.sign(t_stat)
        assert (
            actual_sign == expected_sign
        ), "T-statistic sign should match (estimate - null_value) sign"

        # Property 2: Magnitude should increase as difference increases
        larger_diff_t = calculate_t_statistic(estimate + 5.0, se, null_value)
        assert abs(larger_diff_t) > abs(
            t_stat
        ), "Larger difference should yield larger |t-statistic|"

        # Property 3: Magnitude should decrease as standard error increases
        larger_se_t = calculate_t_statistic(estimate, se * 2, null_value)
        assert abs(larger_se_t) < abs(
            t_stat
        ), "Larger SE should yield smaller |t-statistic|"

    def test_edge_cases_mathematical(self):
        """Test mathematical behavior in edge cases."""
        # Edge case 1: estimate equals null_value
        t_stat = calculate_t_statistic(10.0, 2.0, 10.0)
        assert (
            abs(t_stat) < 1e-15
        ), "T-statistic should be zero when estimate equals null value"

        # Edge case 2: very small standard error
        t_stat = calculate_t_statistic(1.0, 1e-10, 0.0)
        assert t_stat > 1e9, "Very small SE should yield very large t-statistic"

        # Edge case 3: negative estimates
        t_stat = calculate_t_statistic(-15.0, 3.0, 0.0)
        expected = -15.0 / 3.0
        assert (
            abs(t_stat - expected) < 1e-12
        ), "Negative estimates should be handled correctly"

    def test_numerical_precision_extreme_values(self):
        """Test numerical precision with extreme parameter values."""
        # Test with very small values
        t_stat = calculate_t_statistic(1e-8, 1e-9, 0.0)
        expected = 1e-8 / 1e-9
        assert (
            abs(t_stat - expected) < 1e-12
        ), "Should handle very small values precisely"

        # Test with large values
        t_stat = calculate_t_statistic(1e6, 1e5, 5e5)
        expected = (1e6 - 5e5) / 1e5
        assert abs(t_stat - expected) < 1e-10, "Should handle large values precisely"


class TestPValueMathematical:
    """Mathematical validation tests for p-value calculation."""

    def test_cross_validation_with_scipy(self):
        """Cross-validate p-value calculation with scipy.stats."""
        test_cases = [
            # (t_statistic, degrees_freedom)
            (2.5, 30),
            (-1.8, 45),
            (0.0, 20),
            (4.2, 100),
            (-3.1, 15),
            (0.5, 200),
        ]

        for t_stat, df in test_cases:
            # Calculate using our implementation
            our_p_value = calculate_p_value(t_stat, df)

            # Calculate using scipy (two-tailed)
            scipy_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            # Should be identical to machine precision
            np.testing.assert_allclose(our_p_value, scipy_p_value, rtol=1e-14)

    def test_mathematical_properties_p_value(self):
        """Test mathematical properties of p-values."""
        df = 40

        # Property 1: P-values should be between 0 and 1
        for t_stat in [-5.0, -2.0, 0.0, 1.5, 4.0]:
            p_val = calculate_p_value(t_stat, df)
            assert 0.0 <= p_val <= 1.0, f"P-value must be in [0,1], got {p_val}"

        # Property 2: P-value should be symmetric around t=0
        t_stat = 2.3
        p_pos = calculate_p_value(t_stat, df)
        p_neg = calculate_p_value(-t_stat, df)
        assert abs(p_pos - p_neg) < 1e-14, "P-value should be symmetric around t=0"

        # Property 3: Larger |t| should yield smaller p-values
        p_small = calculate_p_value(1.0, df)
        p_large = calculate_p_value(3.0, df)
        assert p_large < p_small, "Larger |t-statistic| should yield smaller p-value"

        # Property 4: P-value at t=0 should be 1.0
        p_zero = calculate_p_value(0.0, df)
        assert abs(p_zero - 1.0) < 1e-14, "P-value at t=0 should be exactly 1.0"

    def test_degrees_freedom_effect(self):
        """Test effect of degrees of freedom on p-values."""
        t_stat = 2.0

        # Higher degrees of freedom should generally yield smaller p-values for same t
        p_low_df = calculate_p_value(t_stat, 5)
        p_high_df = calculate_p_value(t_stat, 100)

        # This relationship holds for moderate t-values
        assert p_high_df <= p_low_df, "Higher df should yield smaller or equal p-value"


class TestPosteriorProbabilityMathematical:
    """Mathematical validation tests for posterior probability calculation."""

    def test_cross_validation_with_scipy(self):
        """Cross-validate posterior probability with scipy.stats."""
        test_cases = [
            # (estimate, standard_error, degrees_freedom, threshold)
            (10.0, 3.0, 30, 0.0),
            (15.5, 4.2, 45, 10.0),
            (-5.0, 2.1, 25, -8.0),
            (0.0, 1.0, 50, 0.0),
            (25.7, 6.8, 80, 20.0),
        ]

        for est, se, df, thresh in test_cases:
            # Calculate using our implementation
            our_prob = calculate_posterior_probability(est, se, df, thresh)

            # Calculate using scipy
            if se > 0:
                t_stat = (thresh - est) / se
                scipy_prob = 1 - stats.t.cdf(t_stat, df)
            else:
                scipy_prob = 1.0 if est > thresh else 0.0

            # Should be identical to machine precision
            np.testing.assert_allclose(our_prob, scipy_prob, rtol=1e-14)

    def test_mathematical_properties_posterior(self):
        """Test mathematical properties of posterior probabilities."""
        est = 12.0
        se = 4.0
        df = 35

        # Property 1: Probabilities should be between 0 and 1
        for thresh in [-10.0, 0.0, 5.0, 12.0, 20.0, 30.0]:
            prob = calculate_posterior_probability(est, se, df, thresh)
            assert 0.0 <= prob <= 1.0, f"Probability must be in [0,1], got {prob}"

        # Property 2: Probability should decrease as threshold increases
        prob_low = calculate_posterior_probability(est, se, df, 5.0)
        prob_high = calculate_posterior_probability(est, se, df, 15.0)
        assert prob_high < prob_low, "Higher threshold should yield lower probability"

        # Property 3: Probability should be 0.5 when threshold equals estimate
        prob_equal = calculate_posterior_probability(est, se, df, est)
        assert (
            abs(prob_equal - 0.5) < 1e-14
        ), "Probability should be 0.5 when threshold equals estimate"

        # Property 4: Probability approaches 1 as threshold approaches -infinity
        prob_very_low = calculate_posterior_probability(est, se, df, est - 10 * se)
        assert (
            prob_very_low > 0.99
        ), "Very low threshold should yield probability close to 1"

    def test_relationship_with_p_values(self):
        """Test mathematical relationship between posterior probabilities and p-values."""
        est = 18.5
        se = 4.2
        df = 38
        threshold = 0.0

        # Calculate posterior probability for threshold = 0
        prob = calculate_posterior_probability(est, se, df, threshold)

        # Calculate corresponding t-statistic and p-value
        t_stat = calculate_t_statistic(est, se, threshold)
        p_val = calculate_p_value(t_stat, df)

        # Mathematical relationship: for positive t-statistic, p_val ≈ 2 * (1 - prob)
        if t_stat > 0:
            expected_p_val = 2 * (1 - prob)
            assert (
                abs(p_val - expected_p_val) < 1e-12
            ), "P-value and posterior probability should satisfy mathematical relationship"

    def test_edge_case_zero_standard_error(self):
        """Test edge case with zero standard error."""
        # When SE = 0, probability should be deterministic
        prob_above = calculate_posterior_probability(10.0, 0.0, 30, 5.0)
        assert (
            prob_above == 1.0
        ), "Probability should be 1 when estimate > threshold and SE = 0"

        prob_below = calculate_posterior_probability(10.0, 0.0, 30, 15.0)
        assert (
            prob_below == 0.0
        ), "Probability should be 0 when estimate < threshold and SE = 0"

        prob_equal = calculate_posterior_probability(10.0, 0.0, 30, 10.0)
        assert (
            prob_equal == 0.0
        ), "Probability should be 0 when estimate = threshold and SE = 0"


class TestCredibleIntervalMathematical:
    """Mathematical validation tests for credible interval calculation."""

    def test_cross_validation_with_scipy(self):
        """Cross-validate credible interval calculation with scipy.stats."""
        test_cases = [
            # (estimate, standard_error, degrees_freedom, confidence_level)
            (15.0, 3.0, 30, 0.95),
            (22.5, 4.8, 45, 0.90),
            (-8.2, 2.1, 25, 0.80),
            (0.0, 1.5, 50, 0.99),
            (100.7, 12.3, 80, 0.95),
        ]

        for est, se, df, conf in test_cases:
            # Calculate using our implementation
            our_ci = calculate_credible_interval(est, se, df, conf)

            # Calculate using scipy
            alpha = 1 - conf
            critical_val = stats.t.ppf(1 - alpha / 2, df)
            margin = critical_val * se

            expected_ci = {
                "lower": est - margin,
                "upper": est + margin,
                "margin_of_error": margin,
                "critical_value": critical_val,
            }

            # Should be identical to machine precision
            for key in expected_ci:
                np.testing.assert_allclose(our_ci[key], expected_ci[key], rtol=1e-14)

    def test_mathematical_properties_credible_interval(self):
        """Test mathematical properties of credible intervals."""
        est = 18.7
        se = 5.2
        df = 42
        conf = 0.95

        ci = calculate_credible_interval(est, se, df, conf)

        # Property 1: Estimate should be in the center of the interval
        center = (ci["lower"] + ci["upper"]) / 2
        assert abs(center - est) < 1e-14, "Estimate should be at center of interval"

        # Property 2: Margin of error should be half the interval width
        width = ci["upper"] - ci["lower"]
        expected_margin = width / 2
        assert (
            abs(ci["margin_of_error"] - expected_margin) < 1e-14
        ), "Margin of error should be half interval width"

        # Property 3: Lower bound should be less than upper bound
        assert ci["lower"] < ci["upper"], "Lower bound must be less than upper bound"

        # Property 4: Higher confidence should yield wider intervals
        ci_narrow = calculate_credible_interval(est, se, df, 0.80)
        ci_wide = calculate_credible_interval(est, se, df, 0.99)

        width_narrow = ci_narrow["upper"] - ci_narrow["lower"]
        width_wide = ci_wide["upper"] - ci_wide["lower"]

        assert (
            width_wide > width_narrow
        ), "Higher confidence should yield wider interval"

    def test_confidence_level_relationships(self):
        """Test relationships between different confidence levels."""
        est = 25.3
        se = 4.1
        df = 60

        # Calculate intervals for different confidence levels
        ci_80 = calculate_credible_interval(est, se, df, 0.80)
        ci_95 = calculate_credible_interval(est, se, df, 0.95)
        ci_99 = calculate_credible_interval(est, se, df, 0.99)

        # Property: 80% interval should be contained within 95% interval
        assert (
            ci_80["lower"] >= ci_95["lower"]
        ), "80% lower bound should be >= 95% lower bound"
        assert (
            ci_80["upper"] <= ci_95["upper"]
        ), "80% upper bound should be <= 95% upper bound"

        # Property: 95% interval should be contained within 99% interval
        assert (
            ci_95["lower"] >= ci_99["lower"]
        ), "95% lower bound should be >= 99% lower bound"
        assert (
            ci_95["upper"] <= ci_99["upper"]
        ), "95% upper bound should be <= 99% upper bound"

    def test_degrees_freedom_effect_on_intervals(self):
        """Test effect of degrees of freedom on interval width."""
        est = 12.0
        se = 3.0
        conf = 0.95

        # Lower degrees of freedom should yield wider intervals
        ci_low_df = calculate_credible_interval(est, se, 5, conf)
        ci_high_df = calculate_credible_interval(est, se, 100, conf)

        width_low = ci_low_df["upper"] - ci_low_df["lower"]
        width_high = ci_high_df["upper"] - ci_high_df["lower"]

        assert (
            width_low > width_high
        ), "Lower degrees of freedom should yield wider intervals"


class TestCriticalValueMathematical:
    """Mathematical validation tests for critical value calculation."""

    def test_cross_validation_with_scipy(self):
        """Cross-validate critical value calculation with scipy.stats."""
        test_cases = [
            # (degrees_freedom, confidence_level, two_tailed)
            (30, 0.95, True),
            (45, 0.90, True),
            (25, 0.80, True),
            (50, 0.99, True),
            (30, 0.95, False),
            (45, 0.90, False),
        ]

        for df, conf, two_tailed in test_cases:
            # Calculate using our implementation
            our_critical = calculate_critical_value(df, conf, two_tailed)

            # Calculate using scipy
            alpha = 1 - conf
            if two_tailed:
                scipy_critical = stats.t.ppf(1 - alpha / 2, df)
            else:
                scipy_critical = stats.t.ppf(1 - alpha, df)

            # Should be identical to machine precision
            np.testing.assert_allclose(our_critical, scipy_critical, rtol=1e-14)

    def test_mathematical_properties_critical_value(self):
        """Test mathematical properties of critical values."""
        df = 40
        conf = 0.95

        # Property 1: Critical values should be positive
        crit_two = calculate_critical_value(df, conf, True)
        crit_one = calculate_critical_value(df, conf, False)

        assert crit_two > 0, "Two-tailed critical value should be positive"
        assert crit_one > 0, "One-tailed critical value should be positive"

        # Property 2: One-tailed critical value should be smaller than two-tailed
        assert crit_one < crit_two, "One-tailed critical value should be smaller"

        # Property 3: Higher confidence should yield larger critical values
        crit_low = calculate_critical_value(df, 0.80, True)
        crit_high = calculate_critical_value(df, 0.99, True)

        assert (
            crit_high > crit_low
        ), "Higher confidence should yield larger critical value"

    def test_degrees_freedom_effect_on_critical_values(self):
        """Test effect of degrees of freedom on critical values."""
        conf = 0.95
        two_tailed = True

        # Lower degrees of freedom should yield larger critical values
        crit_low_df = calculate_critical_value(5, conf, two_tailed)
        crit_high_df = calculate_critical_value(100, conf, two_tailed)

        assert (
            crit_low_df > crit_high_df
        ), "Lower df should yield larger critical values"

        # As df approaches infinity, critical value approaches normal distribution
        crit_very_high_df = calculate_critical_value(10000, conf, two_tailed)
        normal_critical = stats.norm.ppf(1 - (1 - conf) / 2)  # Normal approximation

        assert (
            abs(crit_very_high_df - normal_critical) < 0.01
        ), "Very high df should approximate normal distribution"


class TestInferenceModuleIntegrationMathematical:
    """Integration tests for mathematical consistency across inference module."""

    def test_t_statistic_p_value_consistency(self):
        """Test mathematical consistency between t-statistic and p-value functions."""
        est = 15.2
        se = 4.8
        null_val = 0.0
        df = 35

        # Calculate t-statistic
        t_stat = calculate_t_statistic(est, se, null_val)

        # Calculate p-value
        p_val = calculate_p_value(t_stat, df)

        # Cross-validate with scipy
        scipy_t = (est - null_val) / se
        scipy_p = 2 * (1 - stats.t.cdf(abs(scipy_t), df))

        assert (
            abs(t_stat - scipy_t) < 1e-14
        ), "T-statistic should match scipy calculation"
        assert abs(p_val - scipy_p) < 1e-14, "P-value should match scipy calculation"

    def test_credible_interval_critical_value_consistency(self):
        """Test consistency between credible interval and critical value functions."""
        est = 22.3
        se = 6.1
        df = 50
        conf = 0.90

        # Calculate credible interval
        ci = calculate_credible_interval(est, se, df, conf)

        # Calculate critical value separately
        crit_val = calculate_critical_value(df, conf, two_tailed=True)

        # They should be consistent
        expected_margin = crit_val * se
        assert (
            abs(ci["margin_of_error"] - expected_margin) < 1e-14
        ), "Credible interval margin should match critical value * SE"
        assert (
            abs(ci["critical_value"] - crit_val) < 1e-14
        ), "Critical values should be identical"

    def test_posterior_probability_credible_interval_relationship(self):
        """Test relationship between posterior probability and credible intervals."""
        est = 18.5
        se = 4.2
        df = 38
        conf = 0.95

        # Calculate 95% credible interval
        ci = calculate_credible_interval(est, se, df, conf)

        # Posterior probability that effect > lower bound should be > 0.975
        prob_lower = calculate_posterior_probability(est, se, df, ci["lower"])
        assert prob_lower > 0.975, "P(θ > lower_bound) should be > 97.5% for 95% CI"

        # Posterior probability that effect > upper bound should be < 0.025
        prob_upper = calculate_posterior_probability(est, se, df, ci["upper"])
        assert prob_upper < 0.025, "P(θ > upper_bound) should be < 2.5% for 95% CI"

    def test_numerical_stability_across_functions(self):
        """Test numerical stability across all inference functions."""
        # Test with challenging numerical conditions
        est = 1e-6
        se = 1e-8
        df = 1000

        # All functions should handle small values without numerical issues
        t_stat = calculate_t_statistic(est, se, 0.0)
        p_val = calculate_p_value(t_stat, df)
        prob = calculate_posterior_probability(est, se, df, 0.0)
        ci = calculate_credible_interval(est, se, df, 0.95)
        crit = calculate_critical_value(df, 0.95, True)

        # Results should be finite and reasonable
        assert np.isfinite(t_stat), "T-statistic must be finite"
        assert np.isfinite(p_val), "P-value must be finite"
        assert np.isfinite(prob), "Posterior probability must be finite"
        assert all(np.isfinite(v) for v in ci.values()), "All CI values must be finite"
        assert np.isfinite(crit), "Critical value must be finite"

        # Probabilities should be in valid range
        assert 0.0 <= p_val <= 1.0, "P-value must be in [0,1]"
        assert 0.0 <= prob <= 1.0, "Posterior probability must be in [0,1]"
