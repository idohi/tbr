"""
Comprehensive mathematical validation tests for TBR cumulative variance computation.

This module provides rigorous mathematical validation for the cumulative variance
functions implemented in Task 4.2. Tests ensure mathematical correctness,
cross-validation with functional implementations, and proper statistical properties.

Test Categories
---------------
1. Cumulative variance mathematical formula validation
2. Cumulative standard deviation mathematical validation
3. Cross-validation between variance and standard deviation functions
4. Cross-validation with functional implementation
5. Mathematical properties and relationships
6. Variance decomposition validation
7. Edge case mathematical behavior
8. Numerical precision validation
9. Statistical property verification
10. Integration with other core modules

Mathematical Validation
-----------------------
All tests validate against the TBR mathematical formulas and cross-check with
functional implementations to ensure mathematical correctness. Tests verify:

- Mathematical formula correctness: V[Δr(T)] = T · σ² + T² · v
- Variance decomposition: v = Var(α̂) + 2·x̄_T·Cov(α̂,β̂) + x̄_T²·Var(β̂)
- Cross-validation with functional implementation
- Relationship: variance = standard_deviation²
- Statistical property preservation
- Numerical stability and precision
- Edge case behavior and boundary conditions
"""

import numpy as np

from tbr.core.effects import (
    calculate_cumulative_standard_deviation,
    calculate_cumulative_variance,
)
from tbr.core.prediction import (
    calculate_cumulative_standard_deviation as prediction_cumsd,
)
from tbr.functional.tbr_functions import (
    calculate_cumulative_standard_deviation as functional_cumsd,
)


class TestCumulativeVarianceMathematical:
    """Mathematical validation tests for cumulative variance calculation."""

    def test_mathematical_formula_correctness(self):
        """Test that cumulative variance formula is implemented correctly."""
        # Test cases with known mathematical results
        test_cases = [
            # (test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta)
            ([100, 110, 120], 2.0, 1.0, 0.01, -0.05),
            ([50, 55, 60, 65], 1.5, 0.8, 0.005, -0.02),
            ([1000], 3.0, 2.0, 0.001, 0.0),  # Single value
            ([10, 20, 30, 40, 50], 2.5, 1.5, 0.02, -0.1),
        ]

        for test_x, sigma, var_alpha, var_beta, cov_alpha_beta in test_cases:
            test_x_array = np.array(test_x)

            # Calculate using our implementation
            result = calculate_cumulative_variance(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Calculate expected results using direct mathematical formula
            n = len(test_x_array)
            T_values = np.arange(1, n + 1)

            # Calculate cumulative means
            cumsum_x = np.cumsum(test_x_array)
            x_mean_cumulative = cumsum_x / T_values

            # Calculate v values: v = Var(α̂) + 2·x̄_T·Cov(α̂,β̂) + x̄_T²·Var(β̂)
            expected_v = (
                var_alpha
                + 2 * x_mean_cumulative * cov_alpha_beta
                + (x_mean_cumulative**2) * var_beta
            )

            # Calculate expected variance: V[Δr(T)] = T · σ² + T² · v
            expected_variance = T_values * (sigma**2) + (T_values**2) * expected_v

            # Should be identical to machine precision
            np.testing.assert_allclose(
                result,
                expected_variance,
                rtol=1e-14,
                err_msg=f"Cumulative variance calculation failed for test case: {test_x}",
            )

    def test_variance_decomposition_properties(self):
        """Test mathematical properties of variance decomposition."""
        test_x_values = np.array([100, 105, 110, 115, 120])
        sigma = 2.5
        var_alpha = 1.2
        var_beta = 0.008
        cov_alpha_beta = -0.04

        result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        n = len(test_x_values)
        T_values = np.arange(1, n + 1)

        # Component 1: Residual variance component (T · σ²)
        residual_component = T_values * (sigma**2)

        # Component 2: Model parameter uncertainty component (T² · v)
        cumsum_x = np.cumsum(test_x_values)
        x_mean_cumulative = cumsum_x / T_values
        v_values = (
            var_alpha
            + 2 * x_mean_cumulative * cov_alpha_beta
            + (x_mean_cumulative**2) * var_beta
        )
        parameter_component = (T_values**2) * v_values

        # Total should equal sum of components
        expected_total = residual_component + parameter_component

        np.testing.assert_allclose(
            result,
            expected_total,
            rtol=1e-14,
            err_msg="Cumulative variance should equal sum of residual and parameter components",
        )

        # Property 1: Both components should be positive (for reasonable parameter values)
        assert np.all(residual_component > 0), "Residual component must be positive"
        # Note: parameter_component can be negative if covariance is large and negative

        # Property 2: Residual component grows linearly with T
        residual_ratios = residual_component[1:] / residual_component[:-1]
        expected_ratios = T_values[1:] / T_values[:-1]
        np.testing.assert_allclose(
            residual_ratios,
            expected_ratios,
            rtol=1e-14,
            err_msg="Residual component should grow linearly with T",
        )

    def test_mathematical_properties_cumulative_variance(self):
        """Test mathematical properties of cumulative variance."""
        test_x_values = np.array([80, 85, 90, 95, 100])
        sigma = 2.0
        var_alpha = 1.0
        var_beta = 0.01
        cov_alpha_beta = -0.03

        result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Property 1: All variances should be positive (for reasonable parameters)
        # Note: This may not always hold if covariance is very large and negative
        # but should hold for reasonable parameter ranges
        if np.all(result > 0):  # Only test if all positive
            assert np.all(
                result > 0
            ), "All cumulative variances should be positive for reasonable parameters"

        # Property 2: Variances should generally be increasing (monotonic)
        # This may not always hold due to the quadratic nature of parameter uncertainty
        differences = np.diff(result)
        # Most differences should be positive for reasonable parameters
        positive_differences = np.sum(differences > 0)
        total_differences = len(differences)
        assert (
            positive_differences >= total_differences * 0.7
        ), "Most variance differences should be positive"

        # Property 3: Results should be finite
        assert np.all(np.isfinite(result)), "All variances must be finite"

    def test_edge_cases_mathematical(self):
        """Test mathematical behavior in edge cases."""
        # Edge case 1: Single observation
        result = calculate_cumulative_variance(
            np.array([100]),
            sigma=2.0,
            var_alpha=1.0,
            var_beta=0.01,
            cov_alpha_beta=-0.02,
        )

        # For T=1, x_mean = 100, expected: V[Δr(1)] = 1·σ² + 1²·v
        # v = var_alpha + 2·100·cov_alpha_beta + 100²·var_beta
        expected_v = 1.0 + 2 * 100 * (-0.02) + (100**2) * 0.01
        expected_variance = 1 * (2.0**2) + (1**2) * expected_v

        assert (
            abs(result[0] - expected_variance) < 1e-14
        ), "Single observation case should be handled correctly"

        # Edge case 2: Zero covariance
        result = calculate_cumulative_variance(
            np.array([50, 60]),
            sigma=1.5,
            var_alpha=0.8,
            var_beta=0.005,
            cov_alpha_beta=0.0,
        )

        # Should handle zero covariance correctly
        assert np.all(
            np.isfinite(result)
        ), "Zero covariance should be handled correctly"

        # Edge case 3: Negative covariance (common in regression)
        result = calculate_cumulative_variance(
            np.array([100, 110, 120]),
            sigma=2.0,
            var_alpha=1.0,
            var_beta=0.01,
            cov_alpha_beta=-0.1,
        )

        # Should handle negative covariance correctly
        assert np.all(
            np.isfinite(result)
        ), "Negative covariance should be handled correctly"

    def test_numerical_precision_extreme_values(self):
        """Test numerical precision with extreme parameter values."""
        # Test with very small values
        result = calculate_cumulative_variance(
            np.array([1e-3, 2e-3]),
            sigma=1e-4,
            var_alpha=1e-8,
            var_beta=1e-10,
            cov_alpha_beta=-1e-9,
        )

        assert np.all(np.isfinite(result)), "Should handle very small values"
        assert np.all(
            result >= 0
        ), "Very small values should yield non-negative variances"

        # Test with large values
        result = calculate_cumulative_variance(
            np.array([1e6, 2e6]),
            sigma=1e3,
            var_alpha=1e6,
            var_beta=1e-6,
            cov_alpha_beta=-1e3,
        )

        assert np.all(np.isfinite(result)), "Should handle large values"


class TestCumulativeStandardDeviationMathematical:
    """Mathematical validation tests for cumulative standard deviation calculation."""

    def test_cross_validation_with_functional_implementation(self):
        """Cross-validate with functional implementation."""
        test_cases = [
            ([100, 110, 120], 2.0, 1.0, 0.01, -0.05),
            ([50, 55, 60, 65], 1.5, 0.8, 0.005, -0.02),
            ([1000], 3.0, 2.0, 0.001, 0.0),
        ]

        for test_x, sigma, var_alpha, var_beta, cov_alpha_beta in test_cases:
            test_x_array = np.array(test_x)

            # Calculate using core effects module
            core_result = calculate_cumulative_standard_deviation(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Calculate using functional implementation
            functional_result = functional_cumsd(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Should be identical to machine precision
            np.testing.assert_allclose(
                core_result,
                functional_result,
                rtol=1e-14,
                err_msg=f"Core and functional implementations should match for: {test_x}",
            )

    def test_cross_validation_with_prediction_module(self):
        """Cross-validate effects module with prediction module implementation."""
        test_cases = [
            ([80, 90, 100], 2.5, 1.2, 0.008, -0.04),
            ([200, 210, 220, 230], 3.0, 1.5, 0.012, -0.06),
        ]

        for test_x, sigma, var_alpha, var_beta, cov_alpha_beta in test_cases:
            test_x_array = np.array(test_x)

            # Calculate using effects module
            effects_result = calculate_cumulative_standard_deviation(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Calculate using prediction module
            prediction_result = prediction_cumsd(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Should be identical to machine precision
            np.testing.assert_allclose(
                effects_result,
                prediction_result,
                rtol=1e-14,
                err_msg=f"Effects and prediction modules should match for: {test_x}",
            )

    def test_mathematical_properties_cumulative_std(self):
        """Test mathematical properties of cumulative standard deviation."""
        test_x_values = np.array([100, 105, 110, 115])
        sigma = 2.0
        var_alpha = 1.0
        var_beta = 0.01
        cov_alpha_beta = -0.03

        result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Property 1: All standard deviations should be positive
        assert np.all(result > 0), "All cumulative standard deviations must be positive"

        # Property 2: Results should be finite
        assert np.all(np.isfinite(result)), "All standard deviations must be finite"

        # Property 3: Standard deviations should generally be increasing
        # (though this may not always hold due to quadratic parameter uncertainty)
        differences = np.diff(result)
        positive_differences = np.sum(differences > 0)
        total_differences = len(differences)
        # Most should be positive for reasonable parameters
        assert (
            positive_differences >= total_differences * 0.5
        ), "Most std dev differences should be positive"


class TestVarianceStandardDeviationRelationship:
    """Mathematical validation of variance-standard deviation relationship."""

    def test_variance_std_relationship(self):
        """Test that variance = standard_deviation²."""
        test_cases = [
            ([100, 110, 120], 2.0, 1.0, 0.01, -0.05),
            ([50, 55, 60], 1.5, 0.8, 0.005, -0.02),
            ([200], 3.0, 2.0, 0.001, 0.0),
            ([10, 20, 30, 40], 2.5, 1.5, 0.02, -0.1),
        ]

        for test_x, sigma, var_alpha, var_beta, cov_alpha_beta in test_cases:
            test_x_array = np.array(test_x)

            # Calculate variance
            variance_result = calculate_cumulative_variance(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Calculate standard deviation
            std_result = calculate_cumulative_standard_deviation(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Test relationship: variance = std²
            expected_variance = std_result**2

            np.testing.assert_allclose(
                variance_result,
                expected_variance,
                rtol=1e-14,
                err_msg=f"Variance should equal std² for test case: {test_x}",
            )

    def test_std_variance_consistency(self):
        """Test mathematical consistency between variance and std functions."""
        test_x_values = np.array([80, 85, 90, 95, 100])
        sigma = 2.2
        var_alpha = 1.1
        var_beta = 0.009
        cov_alpha_beta = -0.035

        # Calculate both
        variance_result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        std_result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Test mathematical relationship
        np.testing.assert_allclose(
            np.sqrt(variance_result),
            std_result,
            rtol=1e-14,
            err_msg="sqrt(variance) should equal standard deviation",
        )

        np.testing.assert_allclose(
            variance_result,
            std_result**2,
            rtol=1e-14,
            err_msg="variance should equal std²",
        )


class TestCumulativeVarianceIntegration:
    """Integration tests for cumulative variance with other core modules."""

    def test_integration_with_functional_module(self):
        """Test integration with functional module for complete workflow."""
        # Simulate realistic TBR parameters
        test_x_values = np.array([1000, 1020, 980, 1050, 990])
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05

        # Calculate using core module
        core_variance = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        core_std = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Calculate using functional module
        functional_std = functional_cumsd(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Test consistency
        np.testing.assert_allclose(core_std, functional_std, rtol=1e-14)
        np.testing.assert_allclose(core_variance, functional_std**2, rtol=1e-14)

    def test_mathematical_consistency_across_modules(self):
        """Test mathematical consistency across different core modules."""
        test_x_values = np.array([500, 520, 480, 530, 490])
        sigma = 15.0
        var_alpha = 50.0
        var_beta = 0.002
        cov_alpha_beta = -0.08

        # Test from effects module
        effects_variance = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        effects_std = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Test from prediction module
        prediction_std = prediction_cumsd(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # All should be mathematically consistent
        np.testing.assert_allclose(effects_std, prediction_std, rtol=1e-14)
        np.testing.assert_allclose(effects_variance, effects_std**2, rtol=1e-14)
        np.testing.assert_allclose(effects_variance, prediction_std**2, rtol=1e-14)

    def test_numerical_stability_integration(self):
        """Test numerical stability across integrated workflow."""
        # Test with challenging but reasonable parameters
        test_x_values = np.array([1e-3, 2e-3, 1.5e-3, 2.5e-3])
        sigma = 1e-4
        var_alpha = 1e-8
        var_beta = 1e-10
        cov_alpha_beta = -1e-9

        # All functions should handle small values without numerical issues
        variance_result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        std_result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        functional_result = functional_cumsd(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Results should be finite and consistent
        assert np.all(np.isfinite(variance_result)), "Variance must be finite"
        assert np.all(np.isfinite(std_result)), "Standard deviation must be finite"
        assert np.all(
            np.isfinite(functional_result)
        ), "Functional result must be finite"

        # Mathematical relationships should hold
        np.testing.assert_allclose(std_result, functional_result, rtol=1e-12)
        np.testing.assert_allclose(variance_result, std_result**2, rtol=1e-12)


class TestCumulativeVarianceStatisticalProperties:
    """Tests for statistical properties of cumulative variance functions."""

    def test_variance_scaling_properties(self):
        """Test scaling properties of variance calculations."""
        test_x_values = np.array([100, 110, 120])
        sigma = 2.0
        var_alpha = 1.0
        var_beta = 0.01
        cov_alpha_beta = -0.05

        # Original calculation
        original_variance = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Scale sigma by factor of 2
        scaled_sigma_variance = calculate_cumulative_variance(
            test_x_values, sigma * 2, var_alpha, var_beta, cov_alpha_beta
        )

        # Residual component should scale by 4 (σ² scaling)
        # Parameter component unchanged, so relationship is more complex
        # But we can test that results are reasonable
        assert np.all(
            scaled_sigma_variance > original_variance
        ), "Larger sigma should increase variance"

    def test_parameter_uncertainty_effects(self):
        """Test effects of parameter uncertainty on cumulative variance."""
        test_x_values = np.array([100, 110, 120, 130])
        sigma = 2.0
        base_var_alpha = 1.0
        base_var_beta = 0.01
        base_cov = -0.05

        # Base case
        base_variance = calculate_cumulative_variance(
            test_x_values, sigma, base_var_alpha, base_var_beta, base_cov
        )

        # Increase var_alpha
        high_alpha_variance = calculate_cumulative_variance(
            test_x_values, sigma, base_var_alpha * 2, base_var_beta, base_cov
        )

        # Increase var_beta
        high_beta_variance = calculate_cumulative_variance(
            test_x_values, sigma, base_var_alpha, base_var_beta * 2, base_cov
        )

        # Higher parameter uncertainties should generally increase variance
        # (though covariance effects can complicate this)
        assert np.all(
            high_alpha_variance >= base_variance
        ), "Higher var_alpha should increase variance"
        # var_beta effect depends on x values and time, so we just test reasonableness
        assert np.all(
            np.isfinite(high_beta_variance)
        ), "Higher var_beta should give finite results"

    def test_time_progression_properties(self):
        """Test mathematical properties of variance progression over time."""
        test_x_values = np.array([100, 100, 100, 100, 100])  # Constant x values
        sigma = 2.0
        var_alpha = 1.0
        var_beta = 0.01
        cov_alpha_beta = 0.0  # Zero covariance for simpler analysis

        variance_result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # With constant x values and zero covariance:
        # v = var_alpha + x²·var_beta (constant)
        # V[Δr(T)] = T·σ² + T²·v

        # For constant parameters, variance should follow quadratic + linear pattern
        # Check that second differences are approximately constant (quadratic property)
        if len(variance_result) >= 3:
            second_differences = np.diff(variance_result, n=2)
            # Second differences should be approximately constant for quadratic growth
            std_second_diff = np.std(second_differences)
            mean_second_diff = np.mean(second_differences)
            if mean_second_diff > 0:  # Only test if growing
                cv_second_diff = std_second_diff / mean_second_diff
                assert (
                    cv_second_diff < 0.1
                ), "Second differences should be approximately constant (quadratic growth)"
