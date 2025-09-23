"""
Unit tests for TBR core inference module.

This module contains comprehensive tests for statistical inference functions
including t-statistics, p-values, credible intervals, and posterior probabilities.
Tests cover mathematical accuracy, edge cases, input validation, and integration
with the broader TBR framework.

Test Categories
---------------
1. T-statistic calculation tests
2. P-value computation tests
3. Posterior probability calculation tests
4. Credible interval estimation tests
5. Critical value calculation tests
6. Input validation and error handling
7. Mathematical property verification
8. Edge case handling
9. Integration with TBR workflow

Mathematical Validation
-----------------------
All tests verify mathematical correctness against known statistical formulas
and cross-validate with scipy.stats implementations where applicable.
"""

import numpy as np
import pytest
from scipy import stats

from tbr.core.inference import (
    calculate_credible_interval,
    calculate_critical_value,
    calculate_p_value,
    calculate_posterior_probability,
    calculate_t_statistic,
)


class TestCalculateTStatistic:
    """Test t-statistic calculation function."""

    def test_basic_t_statistic_calculation(self):
        """Test basic t-statistic calculation with known values."""
        # Test case: estimate=10, se=2, null=0 should give t=5
        t_stat = calculate_t_statistic(
            estimate=10.0, standard_error=2.0, null_value=0.0
        )
        assert abs(t_stat - 5.0) < 1e-10

        # Test case: estimate=15, se=3, null=10 should give t=5/3
        t_stat = calculate_t_statistic(
            estimate=15.0, standard_error=3.0, null_value=10.0
        )
        expected = 5.0 / 3.0
        assert abs(t_stat - expected) < 1e-10

    def test_negative_t_statistic(self):
        """Test t-statistic calculation with negative results."""
        # Estimate below null value should give negative t-statistic
        t_stat = calculate_t_statistic(estimate=5.0, standard_error=2.0, null_value=8.0)
        assert t_stat == -1.5

    def test_zero_difference(self):
        """Test t-statistic when estimate equals null value."""
        t_stat = calculate_t_statistic(estimate=5.0, standard_error=2.0, null_value=5.0)
        assert t_stat == 0.0

    def test_default_null_value(self):
        """Test t-statistic calculation with default null value of 0."""
        t_stat = calculate_t_statistic(estimate=6.0, standard_error=3.0)
        assert t_stat == 2.0

    def test_numpy_input_types(self):
        """Test t-statistic calculation with numpy input types."""
        estimate = np.float64(12.0)
        se = np.float32(4.0)
        null = np.int32(0)

        t_stat = calculate_t_statistic(
            estimate=estimate, standard_error=se, null_value=null
        )
        assert abs(t_stat - 3.0) < 1e-10
        assert isinstance(t_stat, float)

    def test_input_validation(self):
        """Test input validation for t-statistic calculation."""
        # Test non-positive standard error
        with pytest.raises(ValueError, match="standard_error must be positive"):
            calculate_t_statistic(estimate=5.0, standard_error=0.0)

        with pytest.raises(ValueError, match="standard_error must be positive"):
            calculate_t_statistic(estimate=5.0, standard_error=-1.0)

        # Test non-numeric inputs
        with pytest.raises(TypeError, match="estimate must be numeric"):
            calculate_t_statistic(estimate="5.0", standard_error=2.0)

        with pytest.raises(TypeError, match="standard_error must be numeric"):
            calculate_t_statistic(estimate=5.0, standard_error="2.0")

        with pytest.raises(TypeError, match="null_value must be numeric"):
            calculate_t_statistic(estimate=5.0, standard_error=2.0, null_value="0.0")


class TestCalculatePValue:
    """Test p-value calculation function."""

    def test_two_tailed_p_value(self):
        """Test two-tailed p-value calculation."""
        # Test with known t-statistic and degrees of freedom
        t_stat = 2.0
        df = 20

        p_val = calculate_p_value(
            t_statistic=t_stat, degrees_freedom=df, two_tailed=True
        )

        # Cross-validate with scipy
        expected = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        assert abs(p_val - expected) < 1e-10

    def test_one_tailed_p_value_positive(self):
        """Test one-tailed p-value for positive t-statistic."""
        t_stat = 1.5
        df = 25

        p_val = calculate_p_value(
            t_statistic=t_stat, degrees_freedom=df, two_tailed=False
        )

        # Cross-validate with scipy
        expected = 1 - stats.t.cdf(t_stat, df=df)
        assert abs(p_val - expected) < 1e-10

    def test_one_tailed_p_value_negative(self):
        """Test one-tailed p-value for negative t-statistic."""
        t_stat = -1.5
        df = 25

        p_val = calculate_p_value(
            t_statistic=t_stat, degrees_freedom=df, two_tailed=False
        )

        # Cross-validate with scipy
        expected = stats.t.cdf(t_stat, df=df)
        assert abs(p_val - expected) < 1e-10

    def test_extreme_t_statistics(self):
        """Test p-value calculation with extreme t-statistics."""
        df = 30

        # Very large positive t-statistic should give p-value near 0
        p_val = calculate_p_value(t_statistic=10.0, degrees_freedom=df)
        assert 0.0 <= p_val <= 0.001

        # Very large negative t-statistic should also give p-value near 0 (two-tailed)
        p_val = calculate_p_value(t_statistic=-10.0, degrees_freedom=df)
        assert 0.0 <= p_val <= 0.001

    def test_zero_t_statistic(self):
        """Test p-value calculation with t-statistic of zero."""
        p_val = calculate_p_value(t_statistic=0.0, degrees_freedom=20, two_tailed=True)
        assert abs(p_val - 1.0) < 1e-10

        p_val = calculate_p_value(t_statistic=0.0, degrees_freedom=20, two_tailed=False)
        assert abs(p_val - 0.5) < 1e-10

    def test_numpy_input_types(self):
        """Test p-value calculation with numpy input types."""
        t_stat = np.float64(2.5)
        df = np.int32(15)

        p_val = calculate_p_value(t_statistic=t_stat, degrees_freedom=df)
        assert isinstance(p_val, float)
        assert 0.0 <= p_val <= 1.0

    def test_input_validation(self):
        """Test input validation for p-value calculation."""
        # Test non-positive degrees of freedom
        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            calculate_p_value(t_statistic=2.0, degrees_freedom=0)

        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            calculate_p_value(t_statistic=2.0, degrees_freedom=-5)

        # Test non-numeric inputs
        with pytest.raises(TypeError, match="t_statistic must be numeric"):
            calculate_p_value(t_statistic="2.0", degrees_freedom=20)

        with pytest.raises(TypeError, match="degrees_freedom must be integer"):
            calculate_p_value(t_statistic=2.0, degrees_freedom=20.5)

        with pytest.raises(TypeError, match="two_tailed must be boolean"):
            calculate_p_value(t_statistic=2.0, degrees_freedom=20, two_tailed="True")


class TestCalculatePosteriorProbability:
    """Test posterior probability calculation function."""

    def test_basic_posterior_probability(self):
        """Test basic posterior probability calculation."""
        # Estimate well above threshold should give high probability
        prob = calculate_posterior_probability(
            estimate=10.0, standard_error=2.0, degrees_freedom=30, threshold=0.0
        )
        assert prob > 0.9

        # Estimate well below threshold should give low probability
        prob = calculate_posterior_probability(
            estimate=-5.0, standard_error=2.0, degrees_freedom=30, threshold=0.0
        )
        assert prob < 0.1

    def test_estimate_equals_threshold(self):
        """Test posterior probability when estimate equals threshold."""
        prob = calculate_posterior_probability(
            estimate=5.0, standard_error=2.0, degrees_freedom=30, threshold=5.0
        )
        # Should be approximately 0.5 due to symmetry
        assert abs(prob - 0.5) < 0.01

    def test_zero_standard_error(self):
        """Test posterior probability with zero standard error (deterministic case)."""
        # Estimate above threshold with zero SE should give probability 1
        prob = calculate_posterior_probability(
            estimate=5.0, standard_error=0.0, degrees_freedom=30, threshold=0.0
        )
        assert prob == 1.0

        # Estimate below threshold with zero SE should give probability 0
        prob = calculate_posterior_probability(
            estimate=-1.0, standard_error=0.0, degrees_freedom=30, threshold=0.0
        )
        assert prob == 0.0

    def test_cross_validation_with_scipy(self):
        """Cross-validate posterior probability with scipy implementation."""
        estimate = 8.5
        se = 3.2
        df = 45
        threshold = 2.0

        prob = calculate_posterior_probability(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            threshold=threshold,
        )

        # Cross-validate: P(θ > threshold) = 1 - F_t((threshold - estimate) / se)
        t_stat = (threshold - estimate) / se
        expected = 1 - stats.t.cdf(t_stat, df=df)

        assert abs(prob - expected) < 1e-10

    def test_different_thresholds(self):
        """Test posterior probability with different threshold values."""
        estimate = 10.0
        se = 4.0
        df = 25

        # Higher threshold should give lower probability
        prob_low_threshold = calculate_posterior_probability(
            estimate=estimate, standard_error=se, degrees_freedom=df, threshold=0.0
        )
        prob_high_threshold = calculate_posterior_probability(
            estimate=estimate, standard_error=se, degrees_freedom=df, threshold=15.0
        )

        assert prob_low_threshold > prob_high_threshold

    def test_input_validation(self):
        """Test input validation for posterior probability calculation."""
        # Test negative standard error
        with pytest.raises(ValueError, match="standard_error must be non-negative"):
            calculate_posterior_probability(
                estimate=5.0, standard_error=-1.0, degrees_freedom=30
            )

        # Test non-positive degrees of freedom
        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            calculate_posterior_probability(
                estimate=5.0, standard_error=2.0, degrees_freedom=0
            )

        # Test non-numeric inputs
        with pytest.raises(TypeError, match="estimate must be numeric"):
            calculate_posterior_probability(
                estimate="5.0", standard_error=2.0, degrees_freedom=30
            )

        with pytest.raises(TypeError, match="standard_error must be numeric"):
            calculate_posterior_probability(
                estimate=5.0, standard_error="2.0", degrees_freedom=30
            )

        with pytest.raises(TypeError, match="degrees_freedom must be integer"):
            calculate_posterior_probability(
                estimate=5.0, standard_error=2.0, degrees_freedom="30"
            )

        with pytest.raises(TypeError, match="threshold must be numeric"):
            calculate_posterior_probability(
                estimate=5.0, standard_error=2.0, degrees_freedom=30, threshold="0.0"
            )


class TestCalculateCredibleInterval:
    """Test credible interval calculation function."""

    def test_basic_credible_interval(self):
        """Test basic credible interval calculation."""
        estimate = 10.0
        se = 2.0
        df = 30
        confidence = 0.95

        ci = calculate_credible_interval(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            confidence_level=confidence,
        )

        # Check structure
        assert isinstance(ci, dict)
        assert all(
            key in ci for key in ["lower", "upper", "margin_of_error", "critical_value"]
        )

        # Check interval contains estimate
        assert ci["lower"] < estimate < ci["upper"]

        # Check symmetry
        assert abs((estimate - ci["lower"]) - (ci["upper"] - estimate)) < 1e-10

        # Check margin of error
        assert abs(ci["margin_of_error"] - (ci["upper"] - estimate)) < 1e-10

    def test_cross_validation_with_scipy(self):
        """Cross-validate credible interval with scipy implementation."""
        estimate = 15.5
        se = 3.8
        df = 40
        confidence = 0.90

        ci = calculate_credible_interval(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            confidence_level=confidence,
        )

        # Cross-validate critical value
        alpha = 1 - confidence
        expected_critical = stats.t.ppf(1 - alpha / 2, df=df)
        assert abs(ci["critical_value"] - expected_critical) < 1e-10

        # Cross-validate interval bounds
        expected_margin = expected_critical * se
        assert abs(ci["margin_of_error"] - expected_margin) < 1e-10
        assert abs(ci["lower"] - (estimate - expected_margin)) < 1e-10
        assert abs(ci["upper"] - (estimate + expected_margin)) < 1e-10

    def test_different_confidence_levels(self):
        """Test credible intervals with different confidence levels."""
        estimate = 12.0
        se = 2.5
        df = 35

        ci_80 = calculate_credible_interval(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            confidence_level=0.80,
        )
        ci_95 = calculate_credible_interval(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            confidence_level=0.95,
        )
        ci_99 = calculate_credible_interval(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            confidence_level=0.99,
        )

        # Higher confidence should give wider intervals
        assert (
            ci_80["margin_of_error"]
            < ci_95["margin_of_error"]
            < ci_99["margin_of_error"]
        )
        assert (ci_99["upper"] - ci_99["lower"]) > (ci_95["upper"] - ci_95["lower"])
        assert (ci_95["upper"] - ci_95["lower"]) > (ci_80["upper"] - ci_80["lower"])

    def test_return_types(self):
        """Test that all returned values are proper float types."""
        ci = calculate_credible_interval(
            estimate=np.float64(8.0),
            standard_error=np.float32(1.5),
            degrees_freedom=np.int32(25),
            confidence_level=0.95,
        )

        assert isinstance(ci["lower"], float)
        assert isinstance(ci["upper"], float)
        assert isinstance(ci["margin_of_error"], float)
        assert isinstance(ci["critical_value"], float)

    def test_input_validation(self):
        """Test input validation for credible interval calculation."""
        # Test non-positive standard error
        with pytest.raises(ValueError, match="standard_error must be positive"):
            calculate_credible_interval(
                estimate=5.0, standard_error=0.0, degrees_freedom=30
            )

        # Test non-positive degrees of freedom
        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            calculate_credible_interval(
                estimate=5.0, standard_error=2.0, degrees_freedom=-5
            )

        # Test invalid confidence level
        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            calculate_credible_interval(
                estimate=5.0,
                standard_error=2.0,
                degrees_freedom=30,
                confidence_level=0.0,
            )

        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            calculate_credible_interval(
                estimate=5.0,
                standard_error=2.0,
                degrees_freedom=30,
                confidence_level=1.0,
            )

        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            calculate_credible_interval(
                estimate=5.0,
                standard_error=2.0,
                degrees_freedom=30,
                confidence_level=1.5,
            )

        # Test non-numeric inputs for TypeError coverage
        with pytest.raises(TypeError, match="estimate must be numeric"):
            calculate_credible_interval(
                estimate="5.0", standard_error=2.0, degrees_freedom=30
            )

        with pytest.raises(TypeError, match="standard_error must be numeric"):
            calculate_credible_interval(
                estimate=5.0, standard_error="2.0", degrees_freedom=30
            )

        with pytest.raises(TypeError, match="degrees_freedom must be integer"):
            calculate_credible_interval(
                estimate=5.0, standard_error=2.0, degrees_freedom="30"
            )

        with pytest.raises(TypeError, match="confidence_level must be numeric"):
            calculate_credible_interval(
                estimate=5.0,
                standard_error=2.0,
                degrees_freedom=30,
                confidence_level="0.95",
            )


class TestCalculateCriticalValue:
    """Test critical value calculation function."""

    def test_two_tailed_critical_value(self):
        """Test two-tailed critical value calculation."""
        df = 20
        confidence = 0.95

        cv = calculate_critical_value(
            degrees_freedom=df, confidence_level=confidence, two_tailed=True
        )

        # Cross-validate with scipy
        alpha = 1 - confidence
        expected = stats.t.ppf(1 - alpha / 2, df=df)
        assert abs(cv - expected) < 1e-10

    def test_one_tailed_critical_value(self):
        """Test one-tailed critical value calculation."""
        df = 15
        confidence = 0.90

        cv = calculate_critical_value(
            degrees_freedom=df, confidence_level=confidence, two_tailed=False
        )

        # Cross-validate with scipy
        alpha = 1 - confidence
        expected = stats.t.ppf(1 - alpha, df=df)
        assert abs(cv - expected) < 1e-10

    def test_critical_value_comparison(self):
        """Test that two-tailed critical values are larger than one-tailed."""
        df = 25
        confidence = 0.95

        cv_two_tailed = calculate_critical_value(
            degrees_freedom=df, confidence_level=confidence, two_tailed=True
        )
        cv_one_tailed = calculate_critical_value(
            degrees_freedom=df, confidence_level=confidence, two_tailed=False
        )

        assert cv_two_tailed > cv_one_tailed

    def test_different_confidence_levels(self):
        """Test critical values with different confidence levels."""
        df = 30

        cv_80 = calculate_critical_value(degrees_freedom=df, confidence_level=0.80)
        cv_95 = calculate_critical_value(degrees_freedom=df, confidence_level=0.95)
        cv_99 = calculate_critical_value(degrees_freedom=df, confidence_level=0.99)

        # Higher confidence should give larger critical values
        assert cv_80 < cv_95 < cv_99

    def test_degrees_freedom_effect(self):
        """Test effect of degrees of freedom on critical values."""
        confidence = 0.95

        cv_small_df = calculate_critical_value(
            degrees_freedom=5, confidence_level=confidence
        )
        cv_large_df = calculate_critical_value(
            degrees_freedom=100, confidence_level=confidence
        )

        # Smaller df should give larger critical values
        assert cv_small_df > cv_large_df

        # Large df should approach normal distribution critical value
        z_critical = 1.96  # Approximate 95% normal critical value
        assert abs(cv_large_df - z_critical) < 0.1

    def test_return_type(self):
        """Test that critical value returns proper float type."""
        cv = calculate_critical_value(
            degrees_freedom=np.int32(25), confidence_level=np.float64(0.95)
        )
        assert isinstance(cv, float)

    def test_input_validation(self):
        """Test input validation for critical value calculation."""
        # Test non-positive degrees of freedom
        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            calculate_critical_value(degrees_freedom=0, confidence_level=0.95)

        # Test invalid confidence level
        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            calculate_critical_value(degrees_freedom=30, confidence_level=0.0)

        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            calculate_critical_value(degrees_freedom=30, confidence_level=1.0)

        # Test non-boolean two_tailed
        with pytest.raises(TypeError, match="two_tailed must be boolean"):
            calculate_critical_value(
                degrees_freedom=30, confidence_level=0.95, two_tailed="True"
            )

        # Test non-numeric inputs for TypeError coverage
        with pytest.raises(TypeError, match="degrees_freedom must be integer"):
            calculate_critical_value(degrees_freedom="30", confidence_level=0.95)

        with pytest.raises(TypeError, match="confidence_level must be numeric"):
            calculate_critical_value(degrees_freedom=30, confidence_level="0.95")


class TestIntegrationWithTBRWorkflow:
    """Test integration of inference functions with TBR workflow."""

    def test_complete_inference_workflow(self):
        """Test complete statistical inference workflow."""
        # Simulate TBR analysis results
        estimate = 25.8
        se = 6.4
        df = 42
        threshold = 0.0
        confidence = 0.95

        # Calculate t-statistic
        t_stat = calculate_t_statistic(
            estimate=estimate, standard_error=se, null_value=threshold
        )

        # Calculate p-value
        p_val = calculate_p_value(t_statistic=t_stat, degrees_freedom=df)

        # Calculate posterior probability
        prob = calculate_posterior_probability(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            threshold=threshold,
        )

        # Calculate credible interval
        ci = calculate_credible_interval(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            confidence_level=confidence,
        )

        # Verify consistency
        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        assert isinstance(prob, float)
        assert isinstance(ci, dict)

        # Check logical relationships
        assert 0.0 <= p_val <= 1.0
        assert 0.0 <= prob <= 1.0
        assert ci["lower"] < estimate < ci["upper"]

        # For positive effect with reasonable SE, should have:
        # - Positive t-statistic
        # - Low p-value (significant effect)
        # - High posterior probability
        assert t_stat > 0
        assert p_val < 0.05  # Significant at 5% level
        assert prob > 0.95  # High probability of positive effect

    def test_mathematical_consistency(self):
        """Test mathematical consistency between functions."""
        estimate = 18.5
        se = 4.2
        df = 38

        # Calculate t-statistic and p-value
        t_stat = calculate_t_statistic(estimate=estimate, standard_error=se)
        p_val = calculate_p_value(t_statistic=t_stat, degrees_freedom=df)

        # Calculate posterior probability for threshold = 0
        prob = calculate_posterior_probability(
            estimate=estimate, standard_error=se, degrees_freedom=df, threshold=0.0
        )

        # For two-tailed test: p-value and posterior probability should be related
        # p_val ≈ 2 * (1 - prob) for positive t-statistic
        expected_p_val = 2 * (1 - prob)
        assert abs(p_val - expected_p_val) < 0.01

    def test_edge_case_consistency(self):
        """Test consistency in edge cases."""
        # Case: estimate = 0, se > 0, threshold = 0
        estimate = 0.0
        se = 2.0
        df = 30
        threshold = 0.0

        t_stat = calculate_t_statistic(
            estimate=estimate, standard_error=se, null_value=threshold
        )
        p_val = calculate_p_value(t_statistic=t_stat, degrees_freedom=df)
        prob = calculate_posterior_probability(
            estimate=estimate,
            standard_error=se,
            degrees_freedom=df,
            threshold=threshold,
        )

        # Should all reflect no effect
        assert abs(t_stat) < 1e-10  # t-statistic should be zero
        assert abs(p_val - 1.0) < 1e-10  # p-value should be 1
        assert abs(prob - 0.5) < 1e-10  # posterior probability should be 0.5
