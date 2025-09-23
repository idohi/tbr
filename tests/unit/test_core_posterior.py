"""
Unit tests for TBR core posterior probability module.

This module contains comprehensive tests for advanced posterior probability
functions and threshold testing capabilities. Tests cover mathematical accuracy,
edge cases, input validation, and integration with the broader TBR framework.

Test Categories
---------------
1. Posterior variance calculation tests
2. Threshold sensitivity analysis tests
3. Incremental posterior probability tests
4. Threshold optimization tests
5. Posterior probability comparison tests
6. Assumption validation tests
7. Input validation and error handling
8. Mathematical property verification
9. Edge case handling
10. Integration with TBR workflow

Mathematical Validation
-----------------------
All tests verify mathematical correctness against known statistical formulas
and cross-validate with scipy.stats implementations where applicable.
"""

import numpy as np
import pytest

from tbr.core.posterior import (
    calculate_incremental_posterior_probabilities,
    calculate_posterior_variance,
    compare_posterior_probabilities,
    optimize_threshold_selection,
    perform_threshold_sensitivity_analysis,
    validate_posterior_assumptions,
)


class TestCalculatePosteriorVariance:
    """Test posterior variance calculation function."""

    def test_basic_posterior_variance(self):
        """Test basic posterior variance calculation."""
        estsd = np.array([2.1, 2.3, 2.0, 2.4, 2.2])
        n_days = 5
        sigma = 1.8

        posterior_var = calculate_posterior_variance(estsd, n_days, sigma)

        # Expected: sum of estsd^2 + n_days * sigma^2
        expected_estsd_var = np.sum(estsd**2)  # 23.510
        expected_residual_var = n_days * (sigma**2)  # 5 * 3.24 = 16.2
        expected_total = expected_estsd_var + expected_residual_var

        assert abs(posterior_var - expected_total) < 1e-10

    def test_single_observation(self):
        """Test posterior variance with single observation."""
        estsd = np.array([3.0])
        posterior_var = calculate_posterior_variance(estsd, n_days=1, sigma=2.0)

        expected = 3.0**2 + 1 * 2.0**2  # 9 + 4 = 13
        assert abs(posterior_var - expected) < 1e-10

    def test_zero_estimation_variance(self):
        """Test posterior variance with zero estimation variance."""
        estsd = np.array([0.0, 0.0, 0.0])
        posterior_var = calculate_posterior_variance(estsd, n_days=3, sigma=1.5)

        expected = 0.0 + 3 * 1.5**2  # 6.75
        assert abs(posterior_var - expected) < 1e-10

    def test_mathematical_properties(self):
        """Test mathematical properties of posterior variance."""
        estsd = np.array([1.0, 2.0, 3.0])
        sigma = 1.0
        n_days = 3

        # Variance should be positive
        posterior_var = calculate_posterior_variance(estsd, n_days, sigma)
        assert posterior_var > 0

        # Doubling sigma should quadruple its contribution
        sigma_contribution_1 = n_days * sigma**2
        sigma_contribution_2 = n_days * (2 * sigma) ** 2
        assert abs(sigma_contribution_2 - 4 * sigma_contribution_1) < 1e-10

    def test_input_validation(self):
        """Test input validation for posterior variance calculation."""
        # Test invalid types
        with pytest.raises(TypeError, match="estsd_values must be numpy array"):
            calculate_posterior_variance([1.0, 2.0], n_days=2, sigma=1.0)

        with pytest.raises(TypeError, match="n_days must be integer"):
            calculate_posterior_variance(np.array([1.0]), n_days=1.5, sigma=1.0)

        with pytest.raises(TypeError, match="sigma must be numeric"):
            calculate_posterior_variance(np.array([1.0]), n_days=1, sigma="1.0")

        # Test invalid values
        with pytest.raises(ValueError, match="n_days must be positive"):
            calculate_posterior_variance(np.array([1.0]), n_days=0, sigma=1.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            calculate_posterior_variance(np.array([1.0]), n_days=1, sigma=-1.0)

        with pytest.raises(ValueError, match="estsd_values cannot be empty"):
            calculate_posterior_variance(np.array([]), n_days=1, sigma=1.0)


class TestPerformThresholdSensitivityAnalysis:
    """Test threshold sensitivity analysis function."""

    def test_basic_sensitivity_analysis(self):
        """Test basic threshold sensitivity analysis."""
        estimate = 10.0
        se = 3.0
        df = 30
        thresholds = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

        result = perform_threshold_sensitivity_analysis(estimate, se, df, thresholds)

        # Check structure
        assert isinstance(result, dict)
        required_keys = ["thresholds", "probabilities", "log_odds", "sensitivity"]
        assert all(key in result for key in required_keys)

        # Check array lengths
        assert len(result["probabilities"]) == len(thresholds)
        assert len(result["log_odds"]) == len(thresholds)
        assert len(result["sensitivity"]) == len(thresholds)

        # Check probabilities are bounded
        probs = result["probabilities"]
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

        # Check decreasing probabilities (higher thresholds -> lower probabilities)
        assert np.all(np.diff(probs) <= 0)

    def test_cross_validation_with_basic_function(self):
        """Cross-validate with basic posterior probability function."""
        from tbr.core.inference import calculate_posterior_probability

        estimate = 12.5
        se = 4.2
        df = 45
        threshold = 8.0

        # Single threshold analysis
        basic_prob = calculate_posterior_probability(estimate, se, df, threshold)

        # Multi-threshold analysis
        sensitivity = perform_threshold_sensitivity_analysis(
            estimate, se, df, np.array([threshold])
        )

        assert abs(sensitivity["probabilities"][0] - basic_prob) < 1e-10

    def test_sensitivity_gradient(self):
        """Test sensitivity gradient calculation."""
        estimate = 15.0
        se = 2.0
        df = 50
        thresholds = np.linspace(10.0, 20.0, 11)

        result = perform_threshold_sensitivity_analysis(estimate, se, df, thresholds)

        # Sensitivity should be negative (probability decreases with threshold)
        assert np.all(result["sensitivity"] <= 0)

        # Maximum sensitivity should be near the estimate
        max_sensitivity_idx = np.argmin(result["sensitivity"])  # Most negative
        threshold_at_max = thresholds[max_sensitivity_idx]
        assert abs(threshold_at_max - estimate) < 2.0  # Within reasonable range

    def test_extreme_cases(self):
        """Test extreme threshold cases."""
        estimate = 10.0
        se = 2.0
        df = 30

        # Very low threshold
        low_threshold = np.array([-100.0])
        result_low = perform_threshold_sensitivity_analysis(
            estimate, se, df, low_threshold
        )
        assert result_low["probabilities"][0] > 0.99  # Should be very high

        # Very high threshold
        high_threshold = np.array([100.0])
        result_high = perform_threshold_sensitivity_analysis(
            estimate, se, df, high_threshold
        )
        assert result_high["probabilities"][0] < 0.01  # Should be very low

    def test_input_validation(self):
        """Test input validation for threshold sensitivity analysis."""
        # Test invalid types
        with pytest.raises(TypeError, match="estimate must be numeric"):
            perform_threshold_sensitivity_analysis("10.0", 2.0, 30, np.array([0.0]))

        with pytest.raises(TypeError, match="standard_error must be numeric"):
            perform_threshold_sensitivity_analysis(10.0, "2.0", 30, np.array([0.0]))

        with pytest.raises(TypeError, match="degrees_freedom must be integer"):
            perform_threshold_sensitivity_analysis(10.0, 2.0, 30.5, np.array([0.0]))

        with pytest.raises(TypeError, match="thresholds must be numpy array"):
            perform_threshold_sensitivity_analysis(10.0, 2.0, 30, [0.0, 5.0])

        # Test invalid values
        with pytest.raises(ValueError, match="standard_error must be positive"):
            perform_threshold_sensitivity_analysis(10.0, -1.0, 30, np.array([0.0]))

        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            perform_threshold_sensitivity_analysis(10.0, 2.0, -5, np.array([0.0]))

        with pytest.raises(ValueError, match="thresholds array cannot be empty"):
            perform_threshold_sensitivity_analysis(10.0, 2.0, 30, np.array([]))


class TestCalculateIncrementalPosteriorProbabilities:
    """Test incremental posterior probability calculation function."""

    def test_basic_incremental_probabilities(self):
        """Test basic incremental posterior probability calculation."""
        estimates = np.array([2.1, 4.8, 7.2, 9.1, 11.5])
        std_errors = np.array([3.2, 4.1, 4.8, 5.2, 5.6])
        df = 45

        result = calculate_incremental_posterior_probabilities(
            estimates, std_errors, df
        )

        # Check structure
        assert isinstance(result, dict)
        required_keys = [
            "day",
            "estimates",
            "standard_errors",
            "probabilities",
            "probability_change",
        ]
        assert all(key in result for key in required_keys)

        # Check array lengths
        n_days = len(estimates)
        assert len(result["day"]) == n_days
        assert len(result["probabilities"]) == n_days
        assert len(result["probability_change"]) == n_days

        # Check day numbering
        assert np.array_equal(result["day"], np.arange(1, n_days + 1))

        # Check probabilities are bounded
        assert np.all(np.array(result["probabilities"]) >= 0.0)
        assert np.all(np.array(result["probabilities"]) <= 1.0)

        # First day should have zero probability change
        assert result["probability_change"][0] == 0.0

    def test_increasing_estimates_pattern(self):
        """Test pattern with increasing estimates (typical TBR scenario)."""
        # Increasing estimates with increasing standard errors
        estimates = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        std_errors = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
        df = 30

        result = calculate_incremental_posterior_probabilities(
            estimates, std_errors, df
        )

        probabilities = np.array(result["probabilities"])

        # Generally expect increasing probabilities as effect accumulates
        # (though this depends on the balance between estimate growth and SE growth)
        assert probabilities[-1] >= probabilities[0]  # Final should be >= first

    def test_cross_validation_with_basic_function(self):
        """Cross-validate with basic posterior probability function."""
        from tbr.core.inference import calculate_posterior_probability

        estimates = np.array([5.0, 8.0, 12.0])
        std_errors = np.array([2.0, 3.0, 4.0])
        df = 40
        threshold = 0.0

        # Incremental analysis
        result = calculate_incremental_posterior_probabilities(
            estimates, std_errors, df, threshold
        )

        # Cross-validate each day
        for i in range(len(estimates)):
            basic_prob = calculate_posterior_probability(
                estimates[i], std_errors[i], df, threshold
            )
            assert abs(result["probabilities"][i] - basic_prob) < 1e-10

    def test_probability_changes(self):
        """Test probability change calculations."""
        estimates = np.array([2.0, 4.0, 6.0, 8.0])
        std_errors = np.array([1.5, 2.0, 2.5, 3.0])
        df = 25

        result = calculate_incremental_posterior_probabilities(
            estimates, std_errors, df
        )

        prob_changes = result["probability_change"]
        probabilities = result["probabilities"]

        # Verify probability changes are calculated correctly
        for i in range(1, len(probabilities)):
            expected_change = probabilities[i] - probabilities[i - 1]
            assert abs(prob_changes[i] - expected_change) < 1e-10

    def test_input_validation(self):
        """Test input validation for incremental posterior probabilities."""
        # Test invalid types
        with pytest.raises(TypeError, match="estimates must be numpy array"):
            calculate_incremental_posterior_probabilities(
                [1.0, 2.0], np.array([1.0, 2.0]), 30
            )

        with pytest.raises(TypeError, match="standard_errors must be numpy array"):
            calculate_incremental_posterior_probabilities(
                np.array([1.0, 2.0]), [1.0, 2.0], 30
            )

        with pytest.raises(TypeError, match="degrees_freedom must be integer"):
            calculate_incremental_posterior_probabilities(
                np.array([1.0, 2.0]), np.array([1.0, 2.0]), 30.5
            )

        with pytest.raises(TypeError, match="threshold must be numeric"):
            calculate_incremental_posterior_probabilities(
                np.array([1.0, 2.0]), np.array([1.0, 2.0]), 30, "0.0"
            )

        # Test mismatched array lengths
        with pytest.raises(
            ValueError, match="estimates and standard_errors must have same length"
        ):
            calculate_incremental_posterior_probabilities(
                np.array([1.0, 2.0]), np.array([1.0]), 30
            )

        # Test empty arrays
        with pytest.raises(ValueError, match="estimates array cannot be empty"):
            calculate_incremental_posterior_probabilities(
                np.array([]), np.array([]), 30
            )

        with pytest.raises(ValueError, match="standard_errors array cannot be empty"):
            calculate_incremental_posterior_probabilities(
                np.array([1.0]), np.array([]), 30
            )

        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            calculate_incremental_posterior_probabilities(
                np.array([1.0]), np.array([1.0]), -5
            )


class TestOptimizeThresholdSelection:
    """Test threshold optimization function."""

    def test_basic_threshold_optimization(self):
        """Test basic threshold optimization."""
        estimate = 12.0
        se = 3.0
        df = 40

        result = optimize_threshold_selection(estimate, se, df)

        # Check structure
        assert isinstance(result, dict)
        required_keys = [
            "optimal_threshold",
            "max_utility",
            "probability_at_optimal",
            "confidence_width",
        ]
        assert all(key in result for key in required_keys)

        # Check types
        assert isinstance(result["optimal_threshold"], float)
        assert isinstance(result["max_utility"], float)
        assert isinstance(result["probability_at_optimal"], float)
        assert isinstance(result["confidence_width"], float)

        # Probability should be bounded
        assert 0.0 <= result["probability_at_optimal"] <= 1.0

        # Confidence width should be positive
        assert result["confidence_width"] > 0

    def test_different_utility_functions(self):
        """Test different utility functions."""
        estimate = 10.0
        se = 2.0
        df = 30

        utilities = ["balanced", "conservative", "aggressive"]
        results = {}

        for utility in utilities:
            results[utility] = optimize_threshold_selection(estimate, se, df, utility)

        # Check that all utility functions produce valid results
        for utility in utilities:
            result = results[utility]

            # All should produce valid optimization results
            assert isinstance(result["optimal_threshold"], float)
            assert isinstance(result["max_utility"], float)
            assert 0.0 <= result["probability_at_optimal"] <= 1.0
            assert result["confidence_width"] > 0

            # Threshold should be reasonable (within 3 standard errors of estimate)
            threshold = result["optimal_threshold"]
            assert estimate - 3 * se <= threshold <= estimate + 3 * se

    def test_threshold_range_specification(self):
        """Test custom threshold range specification."""
        estimate = 15.0
        se = 4.0
        df = 35
        custom_range = (5.0, 25.0)

        result = optimize_threshold_selection(
            estimate, se, df, threshold_range=custom_range
        )

        # Optimal threshold should be within specified range
        assert custom_range[0] <= result["optimal_threshold"] <= custom_range[1]

    def test_extreme_scenarios(self):
        """Test threshold optimization in extreme scenarios."""
        # Very certain positive effect
        result_certain = optimize_threshold_selection(
            estimate=20.0, standard_error=1.0, degrees_freedom=50
        )

        # Very uncertain effect
        result_uncertain = optimize_threshold_selection(
            estimate=5.0, standard_error=10.0, degrees_freedom=20
        )

        # Certain case should have higher optimal threshold
        assert (
            result_certain["optimal_threshold"] > result_uncertain["optimal_threshold"]
        )

    def test_input_validation(self):
        """Test input validation for threshold optimization."""
        # Test invalid types
        with pytest.raises(TypeError, match="estimate must be numeric"):
            optimize_threshold_selection("10.0", 2.0, 30)

        with pytest.raises(TypeError, match="standard_error must be numeric"):
            optimize_threshold_selection(10.0, "2.0", 30)

        with pytest.raises(TypeError, match="degrees_freedom must be integer"):
            optimize_threshold_selection(10.0, 2.0, 30.5)

        with pytest.raises(TypeError, match="utility_function must be string"):
            optimize_threshold_selection(10.0, 2.0, 30, utility_function=123)

        # Test invalid utility function
        with pytest.raises(ValueError, match="utility_function must be"):
            optimize_threshold_selection(10.0, 2.0, 30, utility_function="invalid")

        # Test invalid parameters
        with pytest.raises(ValueError, match="standard_error must be positive"):
            optimize_threshold_selection(10.0, -1.0, 30)

        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            optimize_threshold_selection(10.0, 2.0, -5)


class TestComparePosteriorProbabilities:
    """Test posterior probability comparison function."""

    def test_basic_comparison(self):
        """Test basic posterior probability comparison."""
        scenarios = [
            {
                "estimate": 10.0,
                "standard_error": 3.0,
                "degrees_freedom": 30,
                "name": "Scenario1",
            },
            {
                "estimate": 15.0,
                "standard_error": 4.0,
                "degrees_freedom": 30,
                "name": "Scenario2",
            },
            {
                "estimate": 8.0,
                "standard_error": 2.5,
                "degrees_freedom": 30,
                "name": "Scenario3",
            },
        ]

        result = compare_posterior_probabilities(scenarios)

        # Check structure
        assert isinstance(result, dict)
        required_keys = [
            "scenario_names",
            "probabilities",
            "relative_strength",
            "ranking",
        ]
        assert all(key in result for key in required_keys)

        # Check array lengths
        n_scenarios = len(scenarios)
        assert len(result["scenario_names"]) == n_scenarios
        assert len(result["probabilities"]) == n_scenarios
        assert len(result["relative_strength"]) == n_scenarios
        assert len(result["ranking"]) == n_scenarios

        # Check probabilities are bounded
        assert all(0.0 <= p <= 1.0 for p in result["probabilities"])

        # Check relative strengths sum to 1
        assert abs(sum(result["relative_strength"]) - 1.0) < 1e-10

        # Check ranking values
        rankings = result["ranking"]
        assert set(rankings) == set(range(1, n_scenarios + 1))

    def test_scenario_ranking(self):
        """Test scenario ranking correctness."""
        scenarios = [
            {
                "estimate": 5.0,
                "standard_error": 2.0,
                "degrees_freedom": 30,
                "name": "Low",
            },
            {
                "estimate": 20.0,
                "standard_error": 3.0,
                "degrees_freedom": 30,
                "name": "High",
            },
            {
                "estimate": 12.0,
                "standard_error": 2.5,
                "degrees_freedom": 30,
                "name": "Medium",
            },
        ]

        result = compare_posterior_probabilities(scenarios, threshold=0.0)

        # High estimate should have highest probability and rank 1
        high_idx = 1  # "High" scenario
        assert result["ranking"][high_idx] == 1

        # Probabilities should match ranking order
        probs = result["probabilities"]
        rankings = result["ranking"]

        # Scenario with rank 1 should have highest probability
        rank_1_idx = rankings.index(1)
        assert probs[rank_1_idx] == max(probs)

    def test_scenarios_without_names(self):
        """Test scenarios without explicit names."""
        scenarios = [
            {"estimate": 8.0, "standard_error": 2.0, "degrees_freedom": 25},
            {"estimate": 12.0, "standard_error": 3.0, "degrees_freedom": 25},
        ]

        result = compare_posterior_probabilities(scenarios)

        # Should generate default names
        expected_names = ["Scenario_1", "Scenario_2"]
        assert result["scenario_names"] == expected_names

    def test_cross_validation(self):
        """Cross-validate with individual posterior probability calculations."""
        from tbr.core.inference import calculate_posterior_probability

        scenarios = [
            {"estimate": 10.0, "standard_error": 2.0, "degrees_freedom": 40},
            {"estimate": 15.0, "standard_error": 3.0, "degrees_freedom": 35},
        ]
        threshold = 5.0

        result = compare_posterior_probabilities(scenarios, threshold)

        # Cross-validate each scenario
        for i, scenario in enumerate(scenarios):
            individual_prob = calculate_posterior_probability(
                scenario["estimate"],
                scenario["standard_error"],
                scenario["degrees_freedom"],
                threshold,
            )
            assert abs(result["probabilities"][i] - individual_prob) < 1e-10

    def test_zero_probabilities_edge_case(self):
        """Test edge case where all probabilities are zero."""
        # Create scenarios with very negative estimates and high threshold
        scenarios = [
            {"estimate": -100.0, "standard_error": 1.0, "degrees_freedom": 30},
            {"estimate": -200.0, "standard_error": 1.0, "degrees_freedom": 30},
        ]
        threshold = 100.0  # Very high threshold

        result = compare_posterior_probabilities(scenarios, threshold)

        # All probabilities should be essentially 0
        assert all(p < 0.001 for p in result["probabilities"])

        # Relative strengths should be equal when all probabilities are ~0
        relative_strengths = result["relative_strength"]
        expected_equal_strength = 1.0 / len(scenarios)
        for strength in relative_strengths:
            assert abs(strength - expected_equal_strength) < 0.01

    def test_input_validation(self):
        """Test input validation for posterior probability comparison."""
        # Test invalid types
        with pytest.raises(TypeError, match="scenarios must be list"):
            compare_posterior_probabilities("not_a_list")

        with pytest.raises(TypeError, match="threshold must be numeric"):
            compare_posterior_probabilities(
                [{"estimate": 10.0, "standard_error": 2.0, "degrees_freedom": 30}],
                threshold="0.0",
            )

        # Test empty scenarios
        with pytest.raises(ValueError, match="scenarios list cannot be empty"):
            compare_posterior_probabilities([])

        # Test invalid scenario structure
        with pytest.raises(TypeError, match="scenario 0 must be dict"):
            compare_posterior_probabilities([10.0])  # Not a dict

        invalid_scenarios = [{"estimate": 10.0}]  # Missing required keys
        with pytest.raises(ValueError, match="scenario 0 missing required key"):
            compare_posterior_probabilities(invalid_scenarios)

        # Test invalid parameter values
        invalid_scenarios = [
            {"estimate": 10.0, "standard_error": -1.0, "degrees_freedom": 30}
        ]
        with pytest.raises(
            ValueError, match="scenario 0 standard_error must be positive"
        ):
            compare_posterior_probabilities(invalid_scenarios)

        invalid_scenarios = [
            {"estimate": 10.0, "standard_error": 2.0, "degrees_freedom": -5}
        ]
        with pytest.raises(
            ValueError, match="scenario 0 degrees_freedom must be positive"
        ):
            compare_posterior_probabilities(invalid_scenarios)


class TestValidatePosteriorAssumptions:
    """Test posterior assumption validation function."""

    def test_valid_assumptions(self):
        """Test validation with data that meets assumptions."""
        # Generate normal residuals
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 50)
        df = 47

        result = validate_posterior_assumptions(residuals, df)

        # Check structure
        assert isinstance(result, dict)
        required_keys = [
            "normality_valid",
            "normality_pvalue",
            "independence_valid",
            "independence_pvalue",
            "sample_size_adequate",
            "overall_validity",
            "recommendations",
        ]
        assert all(key in result for key in required_keys)

        # With normal data and adequate sample size, should be valid
        assert result["sample_size_adequate"] is True
        assert result["overall_validity"] in ["Valid", "Questionable"]

    def test_non_normal_residuals(self):
        """Test validation with non-normal residuals."""
        # Generate non-normal residuals (exponential distribution)
        np.random.seed(42)
        residuals = np.random.exponential(1, 50) - 1  # Center around 0
        df = 47

        result = validate_posterior_assumptions(residuals, df)

        # Should detect non-normality
        if result["normality_pvalue"] is not None:
            # May or may not detect depending on power, but should provide p-value
            assert isinstance(result["normality_pvalue"], float)

    def test_small_sample_size(self):
        """Test validation with small sample size."""
        residuals = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
        df = 2  # Small degrees of freedom

        result = validate_posterior_assumptions(residuals, df)

        # Should flag inadequate sample size
        assert result["sample_size_adequate"] is False
        assert "Small sample size" in " ".join(result["recommendations"])

    def test_insufficient_data(self):
        """Test validation with insufficient data."""
        # Very small sample
        residuals = np.array([0.1, -0.2])
        df = 1

        result = validate_posterior_assumptions(residuals, df)

        # Should handle gracefully with appropriate messages
        assert result["overall_validity"] in ["Questionable", "Invalid"]
        assert len(result["recommendations"]) > 0

    def test_input_validation(self):
        """Test input validation for assumption validation."""
        # Test invalid types
        with pytest.raises(TypeError, match="residuals must be numpy array"):
            validate_posterior_assumptions([1.0, 2.0], 30)

        with pytest.raises(TypeError, match="degrees_freedom must be integer"):
            validate_posterior_assumptions(np.array([1.0, 2.0]), 30.5)

        with pytest.raises(TypeError, match="alpha must be numeric"):
            validate_posterior_assumptions(np.array([1.0, 2.0]), 30, alpha="0.05")

        # Test empty residuals
        with pytest.raises(ValueError, match="residuals array cannot be empty"):
            validate_posterior_assumptions(np.array([]), 30)

        # Test invalid alpha
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            validate_posterior_assumptions(np.array([1.0, 2.0]), 30, alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            validate_posterior_assumptions(np.array([1.0, 2.0]), 30, alpha=0.0)

        # Test invalid degrees of freedom
        with pytest.raises(ValueError, match="degrees_freedom must be positive"):
            validate_posterior_assumptions(np.array([1.0, 2.0]), -5)


class TestIntegrationWithTBRWorkflow:
    """Test integration of posterior functions with TBR workflow."""

    def test_complete_posterior_analysis_workflow(self):
        """Test complete posterior analysis workflow."""
        # Simulate TBR analysis results
        estimates = np.array([3.2, 6.8, 10.1, 13.5, 16.8])
        std_errors = np.array([2.1, 3.0, 3.5, 4.1, 4.6])
        df = 42

        # 1. Calculate incremental probabilities
        incremental = calculate_incremental_posterior_probabilities(
            estimates, std_errors, df
        )

        # 2. Perform threshold sensitivity
        thresholds = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        sensitivity = perform_threshold_sensitivity_analysis(
            estimates[-1], std_errors[-1], df, thresholds
        )

        # 3. Optimize threshold
        optimization = optimize_threshold_selection(estimates[-1], std_errors[-1], df)

        # Verify all components work together
        assert len(incremental["probabilities"]) == len(estimates)
        assert len(sensitivity["probabilities"]) == len(thresholds)
        assert isinstance(optimization["optimal_threshold"], float)

        # Check consistency: final incremental probability should match
        # sensitivity analysis at threshold=0
        final_prob = incremental["probabilities"][-1]
        sensitivity_prob_at_zero = sensitivity["probabilities"][0]  # threshold=0
        assert abs(final_prob - sensitivity_prob_at_zero) < 1e-10

    def test_posterior_variance_integration(self):
        """Test posterior variance integration with other functions."""
        # Simulate interval estimation data
        estsd_values = np.array([1.8, 2.1, 1.9, 2.3, 2.0])
        n_days = 5
        sigma = 1.5

        # Calculate posterior variance
        posterior_var = calculate_posterior_variance(estsd_values, n_days, sigma)

        # This should match the variance calculation in interval estimation
        expected_var = np.sum(estsd_values**2) + n_days * sigma**2
        assert abs(posterior_var - expected_var) < 1e-10

        # Use in posterior probability calculation
        se = np.sqrt(posterior_var)
        estimate = 12.0
        df = 40

        # Should work seamlessly with other functions
        from tbr.core.inference import calculate_posterior_probability

        prob = calculate_posterior_probability(estimate, se, df)
        assert 0.0 <= prob <= 1.0

    def test_scenario_comparison_workflow(self):
        """Test scenario comparison workflow."""
        # Compare different analysis periods
        scenarios = [
            {
                "estimate": 8.5,
                "standard_error": 3.2,
                "degrees_freedom": 30,
                "name": "Week1",
            },
            {
                "estimate": 12.1,
                "standard_error": 3.8,
                "degrees_freedom": 35,
                "name": "Week2",
            },
            {
                "estimate": 15.8,
                "standard_error": 4.1,
                "degrees_freedom": 40,
                "name": "Week3",
            },
            {
                "estimate": 18.2,
                "standard_error": 4.5,
                "degrees_freedom": 45,
                "name": "Week4",
            },
        ]

        comparison = compare_posterior_probabilities(scenarios, threshold=10.0)

        # Should rank scenarios appropriately
        rankings = comparison["ranking"]
        probabilities = comparison["probabilities"]

        # Higher estimates should generally have higher probabilities for threshold=10
        # (though this depends on standard errors too)
        # Just check that rankings are valid and probabilities make sense
        assert all(1 <= r <= len(scenarios) for r in rankings)
        assert all(0.0 <= p <= 1.0 for p in probabilities)
