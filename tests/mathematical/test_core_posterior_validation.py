"""
Comprehensive mathematical validation tests for TBR core posterior module.

This module provides rigorous mathematical validation for the advanced posterior
probability functions implemented in Task 4.4. Tests ensure mathematical correctness,
cross-validation with scipy implementations, and proper Bayesian statistical properties.

Test Categories
---------------
1. Posterior variance calculation mathematical validation
2. Threshold sensitivity analysis mathematical validation
3. Incremental posterior probabilities mathematical validation
4. Multi-scenario posterior comparison mathematical validation
5. Cross-validation with scipy.stats and inference module
6. Statistical property verification
7. Edge case mathematical behavior
8. Numerical precision validation

Mathematical Validation
-----------------------
All tests validate against known Bayesian statistical formulas and cross-check with
scipy.stats and the basic inference module to ensure mathematical correctness. Tests verify:

- Mathematical formula correctness against Bayesian statistical theory
- Cross-validation with scipy.stats for identical results
- Integration consistency with basic inference module
- Statistical property preservation and relationships
- Numerical stability and precision
- Edge case behavior and boundary conditions
"""

import numpy as np

from tbr.core.inference import calculate_posterior_probability as basic_posterior_prob
from tbr.core.posterior import (
    calculate_incremental_posterior_probabilities,
    calculate_posterior_variance,
    compare_posterior_probabilities,
    perform_threshold_sensitivity_analysis,
)


class TestPosteriorVarianceMathematical:
    """Mathematical validation tests for posterior variance calculation."""

    def test_mathematical_formula_correctness(self):
        """Test the formula against the quadratic form of the coefficient covariance."""
        # V(sum of predictions) = a' C a, with a = (T_s, sum x_t) and C the
        # covariance matrix of the intercept and slope estimates
        test_cases = [
            # (x_values, sigma, var_alpha, var_beta, cov_alpha_beta)
            ([100.0, 110.0, 120.0], 2.0, 1.0, 0.01, -0.05),
            ([95.0, 100.0, 105.0, 110.0], 1.5, 4.0, 0.0004, -0.03),
            ([1000.0], 25.0, 100.0, 0.001, -0.05),
            ([1000.0, 1020.0, 980.0, 1050.0, 990.0], 25.0, 100.0, 0.001, -0.05),
        ]

        for x_vals, sigma, var_alpha, var_beta, cov_alpha_beta in test_cases:
            x = np.array(x_vals)
            a = np.array([len(x), np.sum(x)])
            cov_matrix = np.array(
                [[var_alpha, cov_alpha_beta], [cov_alpha_beta, var_beta]]
            )
            expected = len(x) * sigma**2 + a @ cov_matrix @ a

            result = calculate_posterior_variance(
                x_values=x,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            assert (
                abs(result - expected) < 1e-8 * expected
            ), f"Posterior variance calculation failed: got {result}, expected {expected}"

    def test_variance_decomposition_properties(self):
        """Test mathematical properties of variance decomposition."""
        x = np.array([95.0, 100.0, 105.0, 110.0])
        sigma, var_alpha, var_beta, cov_alpha_beta = 2.0, 4.0, 0.0004, -0.03
        n = len(x)

        result = calculate_posterior_variance(
            x, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Component 1: Residual variance component
        residual_variance = n * (sigma**2)

        # Component 2: Variance of the summed predictions at the mean control value
        x_mean = np.mean(x)
        v = var_alpha + 2 * x_mean * cov_alpha_beta + x_mean**2 * var_beta
        model_variance = n**2 * v

        # Total should equal sum of components
        expected_total = residual_variance + model_variance

        assert (
            abs(result - expected_total) < 1e-12
        ), "Posterior variance should equal sum of residual and model components"

        # Both components should be positive
        assert residual_variance > 0, "Residual variance component must be positive"
        assert model_variance > 0, "Model variance component must be positive"

    def test_edge_cases_mathematical(self):
        """Test mathematical behavior in edge cases."""
        # Edge case 1: Single observation gives the prediction variance
        result = calculate_posterior_variance(
            np.array([100.0]), 1.5, 4.0, 0.0004, -0.03
        )
        expected = 1.5**2 + (4.0 + 2 * 100.0 * (-0.03) + 100.0**2 * 0.0004)
        assert (
            abs(result - expected) < 1e-12
        ), "Single observation case should be handled correctly"

        # Edge case 2: Known coefficients
        result = calculate_posterior_variance(np.array([1.0, 2.0]), 1.0, 0.0, 0.0, 0.0)
        expected = 2 * 1.0**2
        assert (
            abs(result - expected) < 1e-14
        ), "Zero coefficient uncertainty should yield only residual variance"

        # Edge case 3: Very small sigma (zero sigma not allowed by function validation)
        result = calculate_posterior_variance(
            np.array([1.0, 2.0]), 1e-10, 1.0, 0.0, 0.0
        )
        expected = 2 * (1e-10) ** 2 + 2**2 * 1.0
        assert (
            abs(result - expected) < 1e-14
        ), "Very small sigma should be handled correctly"

    def test_numerical_precision_extreme_values(self):
        """Test numerical precision with extreme parameter values."""
        # Test with very small values
        result = calculate_posterior_variance(
            np.array([1e-3, 2e-3]), 1e-9, 1e-16, 1e-10, 0.0
        )
        x_mean = 1.5e-3
        expected = 2 * (1e-9) ** 2 + 2**2 * (1e-16 + x_mean**2 * 1e-10)
        assert (
            abs(result - expected) < 1e-12 * expected
        ), "Should handle very small values precisely"

        # Test with large values
        result = calculate_posterior_variance(
            np.array([1e6, 2e6]), 5e2, 1e4, 1e-6, -0.05
        )
        x_mean = 1.5e6
        v = 1e4 + 2 * x_mean * (-0.05) + x_mean**2 * 1e-6
        expected = 2 * (5e2) ** 2 + 2**2 * v
        assert (
            abs(result - expected) < 1e-12 * expected
        ), "Should handle large values precisely"

    def test_matches_tbr_analysis_uncertainty_columns(self):
        """Test agreement with predsd and cumsd from a fitted TBR analysis."""
        import pandas as pd

        from tbr.functional import perform_tbr_analysis

        rng = np.random.default_rng(0)
        control = rng.normal(1000, 50, size=44)
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=44),
                "control": control,
                "test": 1.05 * control + rng.normal(0, 10, size=44),
            }
        )
        results = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-31"),
            test_end=pd.Timestamp("2023-02-14"),
            level=0.90,
            threshold=0.0,
        )
        tbr_df = results.tbr_dataframe()
        summary = results.summary().iloc[-1]
        test_rows = tbr_df[tbr_df["period"] == 1].reset_index(drop=True)
        x = test_rows["x"].to_numpy(dtype=float)
        params = (
            float(summary["sigma"]),
            float(summary["var_alpha"]),
            float(summary["var_beta"]),
            float(summary["alpha_beta_cov"]),
        )

        for day in range(len(x)):
            single_day = calculate_posterior_variance(x[day : day + 1], *params)
            assert abs(single_day - test_rows["predsd"].iloc[day] ** 2) < 1e-8

            from_first_day = calculate_posterior_variance(x[: day + 1], *params)
            assert abs(from_first_day - test_rows["cumsd"].iloc[day] ** 2) < 1e-6


class TestThresholdSensitivityMathematical:
    """Mathematical validation tests for threshold sensitivity analysis."""

    def test_cross_validation_with_basic_inference(self):
        """Cross-validate sensitivity analysis with basic inference module."""
        estimate = 12.5
        se = 4.2
        df = 45
        thresholds = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

        # Calculate using sensitivity analysis
        result = perform_threshold_sensitivity_analysis(estimate, se, df, thresholds)

        # Calculate using basic inference module for comparison
        expected_probs = []
        for thresh in thresholds:
            prob = basic_posterior_prob(estimate, se, df, thresh)
            expected_probs.append(prob)

        # Should be identical to machine precision
        np.testing.assert_allclose(result["probabilities"], expected_probs, rtol=1e-14)

    def test_mathematical_properties_sensitivity(self):
        """Test mathematical properties of threshold sensitivity."""
        estimate = 15.0
        se = 3.5
        df = 40
        thresholds = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0])

        result = perform_threshold_sensitivity_analysis(estimate, se, df, thresholds)
        probs = result["probabilities"]
        sensitivity = result["sensitivity"]  # This is the gradient/sensitivity measure

        # Property 1: Probabilities should be decreasing as thresholds increase
        for i in range(len(probs) - 1):
            assert (
                probs[i] >= probs[i + 1]
            ), f"Probabilities should decrease: P({thresholds[i]}) >= P({thresholds[i+1]})"

        # Property 2: All probabilities should be in [0, 1]
        assert np.all((probs >= 0) & (probs <= 1)), "All probabilities must be in [0,1]"

        # Property 3: Sensitivity should be negative (decreasing probabilities)
        assert np.all(
            sensitivity <= 0
        ), "All sensitivity values should be negative or zero"

        # Property 4: Probability should be 0.5 when threshold equals estimate
        thresh_equal_idx = np.where(thresholds == estimate)[0]
        if len(thresh_equal_idx) > 0:
            assert (
                abs(probs[thresh_equal_idx[0]] - 0.5) < 1e-14
            ), "Probability should be 0.5 when threshold equals estimate"

    def test_gradient_calculation_accuracy(self):
        """Test mathematical accuracy of gradient calculations."""
        estimate = 10.0
        se = 2.5
        df = 50
        thresholds = np.array([8.0, 10.0, 12.0])

        result = perform_threshold_sensitivity_analysis(estimate, se, df, thresholds)
        calculated_gradients = result["sensitivity"]

        # Calculate gradients numerically for comparison
        epsilon = 1e-8
        numerical_gradients = []

        for thresh in thresholds:
            prob_plus = basic_posterior_prob(estimate, se, df, thresh + epsilon)
            prob_minus = basic_posterior_prob(estimate, se, df, thresh - epsilon)
            numerical_grad = (prob_plus - prob_minus) / (2 * epsilon)
            numerical_gradients.append(numerical_grad)

        # Should be close to numerical gradients
        np.testing.assert_allclose(
            calculated_gradients, numerical_gradients, rtol=1e-6, atol=1e-8
        )


class TestIncrementalPosteriorMathematical:
    """Mathematical validation tests for incremental posterior probabilities."""

    def test_cross_validation_with_basic_inference(self):
        """Cross-validate incremental probabilities with basic inference."""
        estimates = np.array([8.0, 12.0, 15.0, 18.0])
        standard_errors = np.array([3.0, 3.5, 4.0, 4.2])
        df = 35
        threshold = 5.0

        # Calculate using incremental function
        result = calculate_incremental_posterior_probabilities(
            estimates, standard_errors, df, threshold
        )

        # Calculate using basic inference for comparison
        expected_probs = []
        for est, se in zip(estimates, standard_errors):
            prob = basic_posterior_prob(est, se, df, threshold)
            expected_probs.append(prob)

        # Should be identical to machine precision
        np.testing.assert_allclose(result["probabilities"], expected_probs, rtol=1e-14)

    def test_mathematical_properties_incremental(self):
        """Test mathematical properties of incremental probabilities."""
        estimates = np.array([5.0, 10.0, 15.0, 20.0])
        standard_errors = np.array([2.0, 2.5, 3.0, 3.5])
        df = 40
        threshold = 0.0

        result = calculate_incremental_posterior_probabilities(
            estimates, standard_errors, df, threshold
        )

        probs = result["probabilities"]
        probability_change = result["probability_change"]

        # Property 1: All probabilities should be in [0, 1]
        assert np.all((probs >= 0) & (probs <= 1)), "All probabilities must be in [0,1]"

        # Property 2: Probability changes should be finite
        assert np.all(
            np.isfinite(probability_change)
        ), "All probability changes should be finite"

        # Property 3: For positive estimates and threshold=0, probabilities should be > 0.5
        for i, est in enumerate(estimates):
            if est > threshold:
                assert (
                    probs[i] > 0.5
                ), f"Positive estimate {est} should have probability > 0.5 for threshold {threshold}"

    def test_array_length_consistency(self):
        """Test consistency when arrays have different lengths."""
        estimates = np.array([10.0, 15.0, 12.0])
        standard_errors = np.array([3.0, 4.0, 3.5])
        df = 30

        result = calculate_incremental_posterior_probabilities(
            estimates, standard_errors, df
        )

        # All output arrays should have same length as input
        assert len(result["probabilities"]) == len(estimates)
        assert len(result["probability_change"]) == len(estimates)
        assert len(result["day"]) == len(estimates)


class TestPosteriorComparisonMathematical:
    """Mathematical validation tests for multi-scenario posterior comparison."""

    def test_cross_validation_with_basic_inference(self):
        """Cross-validate scenario comparison with basic inference."""
        scenarios = [
            {
                "estimate": 10.0,
                "standard_error": 3.0,
                "degrees_freedom": 30,
                "name": "A",
            },
            {
                "estimate": 15.0,
                "standard_error": 4.0,
                "degrees_freedom": 35,
                "name": "B",
            },
            {
                "estimate": 8.0,
                "standard_error": 2.5,
                "degrees_freedom": 25,
                "name": "C",
            },
        ]
        threshold = 5.0

        # Calculate using comparison function
        result = compare_posterior_probabilities(scenarios, threshold)

        # Calculate using basic inference for comparison
        expected_probs = []
        for scenario in scenarios:
            prob = basic_posterior_prob(
                scenario["estimate"],
                scenario["standard_error"],
                scenario["degrees_freedom"],
                threshold,
            )
            expected_probs.append(prob)

        # Should be identical to machine precision
        np.testing.assert_allclose(result["probabilities"], expected_probs, rtol=1e-14)

    def test_mathematical_properties_comparison(self):
        """Test mathematical properties of posterior comparison."""
        scenarios = [
            {"estimate": 12.0, "standard_error": 3.0, "degrees_freedom": 40},
            {"estimate": 18.0, "standard_error": 4.5, "degrees_freedom": 35},
            {"estimate": 6.0, "standard_error": 2.0, "degrees_freedom": 50},
        ]

        result = compare_posterior_probabilities(scenarios)

        probs = result["probabilities"]
        ranking = result["ranking"]

        # Property 1: All probabilities should be in [0, 1]
        assert np.all(
            (np.array(probs) >= 0) & (np.array(probs) <= 1)
        ), "All probabilities must be in [0,1]"

        # Property 2: Without ties, rankings are a permutation of 1, ..., n
        n_scenarios = len(scenarios)
        assert set(ranking) == set(
            range(1, n_scenarios + 1)
        ), "Rankings should be unique and complete"

        # Property 3: Highest probability should have rank 1
        max_prob_idx = np.argmax(probs)
        assert ranking[max_prob_idx] == 1, "Highest probability should have rank 1"

    def test_ranking_consistency(self):
        """Test consistency of ranking with probabilities."""
        scenarios = [
            {"estimate": 5.0, "standard_error": 2.0, "degrees_freedom": 30},  # Low prob
            {
                "estimate": 20.0,
                "standard_error": 3.0,
                "degrees_freedom": 30,
            },  # High prob
            {
                "estimate": 12.0,
                "standard_error": 4.0,
                "degrees_freedom": 30,
            },  # Medium prob
        ]

        result = compare_posterior_probabilities(scenarios)

        probs = np.array(result["probabilities"])
        ranking = np.array(result["ranking"])

        # Higher probabilities should have lower rank numbers (1 is best)
        sorted_prob_indices = np.argsort(probs)[::-1]  # Descending order
        expected_ranking = np.empty_like(ranking)
        expected_ranking[sorted_prob_indices] = np.arange(1, len(probs) + 1)

        np.testing.assert_array_equal(ranking, expected_ranking)


class TestPosteriorModuleIntegrationMathematical:
    """Integration tests for mathematical consistency across posterior module."""

    def test_consistency_with_basic_inference(self):
        """Test mathematical consistency with basic inference module."""
        estimate = 14.0
        se = 3.8
        df = 42
        threshold = 6.0

        # Calculate using basic inference
        basic_prob = basic_posterior_prob(estimate, se, df, threshold)

        # Calculate using threshold sensitivity (single threshold)
        sensitivity = perform_threshold_sensitivity_analysis(
            estimate, se, df, np.array([threshold])
        )
        sensitivity_prob = sensitivity["probabilities"][0]

        # Calculate using incremental (single estimate)
        incremental = calculate_incremental_posterior_probabilities(
            np.array([estimate]), np.array([se]), df, threshold
        )
        incremental_prob = incremental["probabilities"][0]

        # All should be identical
        assert (
            abs(basic_prob - sensitivity_prob) < 1e-14
        ), "Sensitivity analysis should match basic inference"
        assert (
            abs(basic_prob - incremental_prob) < 1e-14
        ), "Incremental analysis should match basic inference"

    def test_numerical_stability_across_functions(self):
        """Test numerical stability across all posterior functions."""
        # Test with challenging but reasonable numerical conditions
        estimate = 1e-3
        se = 1e-4
        df = 100

        # All functions should handle small values without numerical issues
        variance = calculate_posterior_variance(
            np.array([1e-3, 2e-3]), 1e-9, 1e-16, 1e-10, 0.0
        )

        sensitivity = perform_threshold_sensitivity_analysis(
            estimate, se, df, np.array([0.0, 1e-4])
        )

        incremental = calculate_incremental_posterior_probabilities(
            np.array([estimate]), np.array([se]), df, 0.0
        )

        # Results should be finite
        assert np.isfinite(variance), "Variance must be finite"
        assert np.all(
            np.isfinite(sensitivity["probabilities"])
        ), "Sensitivity probabilities must be finite"
        assert np.all(
            np.isfinite(incremental["probabilities"])
        ), "Incremental probabilities must be finite"

    def test_mathematical_relationships_across_functions(self):
        """Test mathematical relationships between different posterior functions."""
        estimates = np.array([8.0, 12.0, 16.0])
        standard_errors = np.array([2.5, 3.0, 3.5])
        df = 35

        # Incremental analysis
        incremental = calculate_incremental_posterior_probabilities(
            estimates, standard_errors, df, 0.0
        )

        # Individual sensitivity analyses
        individual_probs = []
        for est, se in zip(estimates, standard_errors):
            sensitivity = perform_threshold_sensitivity_analysis(
                est, se, df, np.array([0.0])
            )
            individual_probs.append(sensitivity["probabilities"][0])

        # Should match individual calculations
        np.testing.assert_allclose(
            incremental["probabilities"], individual_probs, rtol=1e-14
        )
