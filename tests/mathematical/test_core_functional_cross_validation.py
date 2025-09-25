"""
Comprehensive cross-validation tests between core modules and functional implementation.

This module provides rigorous cross-validation between the Phase 2 core modules
(effects, inference, posterior, cumulative variance) and the proven functional
implementation. Tests ensure perfect mathematical consistency and identical results.

Test Categories
---------------
1. Core effects vs functional implementation cross-validation
2. Core inference vs functional implementation cross-validation
3. Core posterior vs functional implementation cross-validation
4. Cumulative variance vs functional implementation cross-validation
5. End-to-end workflow cross-validation
6. Statistical consistency validation
7. Numerical precision cross-validation
8. Edge case cross-validation
9. Integration workflow validation
10. Mathematical relationship preservation

Cross-Validation Methodology
-----------------------------
All tests compare core module results with functional implementation results
using machine precision tolerances (rtol=1e-14) to ensure identical mathematical
behavior. Tests verify:

- Identical numerical results between implementations
- Preservation of mathematical relationships
- Statistical property consistency
- Edge case behavior matching
- Numerical precision maintenance
- Workflow integration consistency
"""

import numpy as np
import pandas as pd
import pytest

from tbr.core.effects import calculate_cumulative_standard_deviation as core_cumsd
from tbr.core.effects import calculate_cumulative_variance as core_cumvar
from tbr.core.effects import compute_interval_estimate_and_ci as core_interval
from tbr.core.effects import create_tbr_summary as core_summary
from tbr.core.inference import calculate_credible_interval as core_credible
from tbr.core.inference import calculate_p_value as core_p_value
from tbr.core.inference import calculate_posterior_probability as core_posterior_prob
from tbr.core.inference import calculate_t_statistic as core_t_stat
from tbr.core.prediction import (
    calculate_cumulative_standard_deviation as core_pred_cumsd,
)
from tbr.core.prediction import generate_counterfactual_predictions as core_predictions
from tbr.functional.tbr_functions import (
    calculate_cumulative_standard_deviation as func_cumsd,
)
from tbr.functional.tbr_functions import (
    compute_interval_estimate_and_ci as func_interval,
)
from tbr.functional.tbr_functions import create_tbr_summary as func_summary
from tbr.functional.tbr_functions import (
    generate_counterfactual_predictions as func_predictions,
)


class TestCoreEffectsFunctionalCrossValidation:
    """Cross-validation tests for core effects module vs functional implementation."""

    def test_cumulative_standard_deviation_cross_validation(self):
        """Cross-validate cumulative standard deviation calculations."""
        test_cases = [
            # (test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta)
            ([100, 110, 120], 2.0, 1.0, 0.01, -0.05),
            ([50, 55, 60, 65], 1.5, 0.8, 0.005, -0.02),
            ([1000], 3.0, 2.0, 0.001, 0.0),
            ([10, 20, 30, 40, 50], 2.5, 1.5, 0.02, -0.1),
            ([500, 520, 480, 530, 490, 510], 25.0, 100.0, 0.001, -0.05),
        ]

        for test_x, sigma, var_alpha, var_beta, cov_alpha_beta in test_cases:
            test_x_array = np.array(test_x)

            # Calculate using core effects module
            core_result = core_cumsd(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Calculate using functional implementation
            func_result = func_cumsd(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Should be identical to machine precision
            np.testing.assert_allclose(
                core_result,
                func_result,
                rtol=1e-14,
                err_msg=f"Core and functional cumsd should match exactly for: {test_x}",
            )

    def test_cumulative_variance_cross_validation(self):
        """Cross-validate cumulative variance with functional standard deviation."""
        test_cases = [
            ([100, 110, 120], 2.0, 1.0, 0.01, -0.05),
            ([80, 85, 90, 95], 1.8, 0.9, 0.008, -0.03),
            ([200], 4.0, 3.0, 0.002, 0.0),
        ]

        for test_x, sigma, var_alpha, var_beta, cov_alpha_beta in test_cases:
            test_x_array = np.array(test_x)

            # Calculate variance using core effects module
            core_variance = core_cumvar(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Calculate standard deviation using functional implementation
            func_std = func_cumsd(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Core variance should equal functional std²
            expected_variance = func_std**2
            np.testing.assert_allclose(
                core_variance,
                expected_variance,
                rtol=1e-14,
                err_msg=f"Core variance should equal functional std² for: {test_x}",
            )

    def test_interval_estimate_cross_validation(self):
        """Cross-validate interval estimation and confidence intervals."""
        # Create realistic TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [1, 1, 1, 1, 1],
                "y": [100, 105, 110, 108, 112],
                "pred": [98, 103, 107, 106, 110],
                "cumdif": [5.0, 12.0, 20.0, 28.0, 35.0],
                "cumsd": [8.0, 11.0, 14.0, 16.0, 18.0],
                "estsd": [3.2, 4.1, 4.8, 5.2, 5.6],
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "estimate": [35.0],
                "precision": [9.2],
                "lower": [25.8],
                "upper": [44.2],
                "sigma": [15.0],
                "t_dist_df": [30],
            }
        )

        test_cases = [
            (2, 4, 0.90),  # start_day, end_day, ci_level
            (1, 3, 0.95),
            (3, 5, 0.80),
        ]

        for start_day, end_day, ci_level in test_cases:
            # Calculate using core effects module
            core_result = core_interval(
                tbr_df=tbr_df,
                tbr_summary=tbr_summary,
                start_day=start_day,
                end_day=end_day,
                ci_level=ci_level,
            )

            # Calculate using functional implementation
            func_result = func_interval(
                tbr_df=tbr_df,
                tbr_summary=tbr_summary,
                start_day=start_day,
                end_day=end_day,
                ci_level=ci_level,
            )

            # All results should be identical
            for key in core_result.keys():
                assert key in func_result, f"Missing key {key} in functional result"
                np.testing.assert_allclose(
                    core_result[key],
                    func_result[key],
                    rtol=1e-14,
                    err_msg=f"Core and functional {key} should match for interval ({start_day}, {end_day})",
                )

    def test_tbr_summary_cross_validation(self):
        """Cross-validate TBR summary creation."""
        # Create realistic TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [1] * 10,
                "y": [100, 105, 110, 108, 112, 115, 118, 120, 122, 125],
                "pred": [98, 103, 107, 106, 110, 113, 116, 118, 120, 123],
                "cumdif": [2, 4, 7, 9, 11, 13, 15, 17, 19, 21],
                "cumsd": [5, 7, 9, 11, 12, 14, 15, 16, 17, 18],
                "estsd": [2.1, 2.8, 3.2, 3.6, 3.9, 4.1, 4.3, 4.5, 4.6, 4.8],
            }
        )

        # Parameters for TBR summary
        alpha = 10.5
        beta = 0.98
        sigma = 15.0
        var_alpha = 85.0
        var_beta = 0.0008
        cov_alpha_beta = -0.042
        degrees_freedom = 25
        level = 0.80
        threshold = 0.0

        # Calculate using core effects module
        core_result = core_summary(
            tbr_dataframe=tbr_df,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
            degrees_freedom=degrees_freedom,
            level=level,
            threshold=threshold,
        )

        # Calculate using functional implementation
        func_result = func_summary(
            tbr_dataframe=tbr_df,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
            degrees_freedom=degrees_freedom,
            level=level,
            threshold=threshold,
        )

        # Results should be identical DataFrames
        pd.testing.assert_frame_equal(
            core_result,
            func_result,
            check_exact=False,
            rtol=1e-14,
            obj="Core and functional TBR summaries should be identical",
        )


class TestCorePredictionFunctionalCrossValidation:
    """Cross-validation tests for core prediction module vs functional implementation."""

    def test_counterfactual_predictions_cross_validation(self):
        """Cross-validate counterfactual predictions generation."""
        # Create realistic test period data
        test_period_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "control": [1000, 1020, 980, 1050, 990],
                "treatment": [1100, 1130, 1080, 1160, 1090],
            }
        )

        test_cases = [
            # (alpha, beta, sigma, x_mean, n_pretest)
            (10.5, 0.95, 25.0, 1000.0, 30),
            (5.2, 1.02, 18.5, 850.0, 45),
            (15.8, 0.88, 32.0, 1200.0, 25),
        ]

        for alpha, beta, sigma, x_mean, n_pretest in test_cases:
            # Calculate using core prediction module
            core_result = core_predictions(
                alpha=alpha,
                beta=beta,
                sigma=sigma,
                x_mean=x_mean,
                n_pretest=n_pretest,
                test_period_data=test_period_data,
                control_col="control",
                time_col="date",
            )

            # Calculate using functional implementation
            func_result = func_predictions(
                alpha=alpha,
                beta=beta,
                sigma=sigma,
                x_mean=x_mean,
                n_pretest=n_pretest,
                test_period_data=test_period_data,
                control_col="control",
                time_col="date",
            )

            # DataFrames should be identical
            pd.testing.assert_frame_equal(
                core_result,
                func_result,
                check_exact=False,
                rtol=1e-14,
                obj=f"Core and functional predictions should be identical for params: {(alpha, beta, sigma)}",
            )

    def test_cumulative_std_prediction_cross_validation(self):
        """Cross-validate cumulative standard deviation in prediction module."""
        test_cases = [
            ([100, 110, 120], 2.0, 1.0, 0.01, -0.05),
            ([500, 520, 480], 15.0, 50.0, 0.002, -0.08),
        ]

        for test_x, sigma, var_alpha, var_beta, cov_alpha_beta in test_cases:
            test_x_array = np.array(test_x)

            # Calculate using core prediction module
            core_pred_result = core_pred_cumsd(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Calculate using functional implementation
            func_result = func_cumsd(
                test_x_values=test_x_array,
                sigma=sigma,
                var_alpha=var_alpha,
                var_beta=var_beta,
                cov_alpha_beta=cov_alpha_beta,
            )

            # Should be identical to machine precision
            np.testing.assert_allclose(
                core_pred_result,
                func_result,
                rtol=1e-14,
                err_msg=f"Core prediction and functional cumsd should match for: {test_x}",
            )


class TestCoreInferenceFunctionalCrossValidation:
    """Cross-validation tests for core inference module vs scipy implementations."""

    def test_statistical_inference_scipy_cross_validation(self):
        """Cross-validate statistical inference functions with scipy."""
        from scipy import stats

        test_cases = [
            # (estimate, standard_error, degrees_freedom, null_value)
            (10.5, 3.2, 25, 0.0),
            (8.7, 2.1, 45, 5.0),
            (-5.3, 1.8, 35, -2.0),
            (15.2, 4.5, 60, 10.0),
        ]

        for estimate, se, df, null_val in test_cases:
            # Test t-statistic calculation
            core_t = core_t_stat(estimate, se, null_val)
            expected_t = (estimate - null_val) / se
            assert (
                abs(core_t - expected_t) < 1e-14
            ), f"T-statistic should match formula for {(estimate, se, null_val)}"

            # Test p-value calculation
            core_p = core_p_value(core_t, df)
            scipy_p = 2 * (1 - stats.t.cdf(abs(core_t), df))
            np.testing.assert_allclose(
                core_p,
                scipy_p,
                rtol=1e-14,
                err_msg=f"P-value should match scipy for {(core_t, df)}",
            )

            # Test posterior probability calculation
            threshold = 0.0
            core_post_prob = core_posterior_prob(estimate, se, df, threshold)
            t_stat = (threshold - estimate) / se
            scipy_post_prob = 1 - stats.t.cdf(t_stat, df)
            np.testing.assert_allclose(
                core_post_prob,
                scipy_post_prob,
                rtol=1e-14,
                err_msg=f"Posterior probability should match scipy for {(estimate, se, df, threshold)}",
            )

            # Test credible interval calculation
            for conf_level in [0.90, 0.95, 0.99]:
                core_ci = core_credible(estimate, se, df, conf_level)
                alpha = 1 - conf_level
                t_critical = stats.t.ppf(1 - alpha / 2, df)
                margin = t_critical * se
                expected_ci = {
                    "lower": estimate - margin,
                    "upper": estimate + margin,
                    "margin_of_error": margin,
                    "critical_value": t_critical,
                }

                for key in expected_ci:
                    np.testing.assert_allclose(
                        core_ci[key],
                        expected_ci[key],
                        rtol=1e-14,
                        err_msg=f"Credible interval {key} should match scipy for {(estimate, se, df, conf_level)}",
                    )


class TestIntegratedWorkflowCrossValidation:
    """Cross-validation tests for integrated TBR workflows."""

    def test_end_to_end_tbr_workflow_cross_validation(self):
        """Cross-validate complete TBR analysis workflow."""
        # Create realistic learning and test period data
        np.random.seed(42)  # For reproducible results

        test_period_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=7),
                "control": [980, 1020, 990, 1030, 1010, 1040, 995],
                "treatment": [1080, 1130, 1095, 1140, 1115, 1150, 1105],
            }
        )

        # This test validates that when we use the same parameters derived from
        # the functional workflow, the core modules produce identical results

        # Use known regression parameters (would normally come from regression fitting)
        alpha = 12.5
        beta = 0.98
        sigma = 28.5
        x_mean = 1005.0
        n_pretest = 30
        var_alpha = 85.0
        var_beta = 0.0008
        cov_alpha_beta = -0.042

        # Generate predictions using both implementations
        core_predictions_result = core_predictions(
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            x_mean=x_mean,
            n_pretest=n_pretest,
            test_period_data=test_period_data,
            control_col="control",
            time_col="date",
        )

        func_predictions_result = func_predictions(
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            x_mean=x_mean,
            n_pretest=n_pretest,
            test_period_data=test_period_data,
            control_col="control",
            time_col="date",
        )

        # Predictions should be identical
        pd.testing.assert_frame_equal(
            core_predictions_result,
            func_predictions_result,
            check_exact=False,
            rtol=1e-14,
            obj="Core and functional predictions should be identical in end-to-end workflow",
        )

        # Calculate cumulative standard deviations using both implementations
        test_x_values = test_period_data["control"].values

        core_cumsd_result = core_cumsd(
            test_x_values=test_x_values,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
        )

        func_cumsd_result = func_cumsd(
            test_x_values=test_x_values,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
        )

        # Cumulative standard deviations should be identical
        np.testing.assert_allclose(
            core_cumsd_result,
            func_cumsd_result,
            rtol=1e-14,
            err_msg="Core and functional cumsd should be identical in end-to-end workflow",
        )

    def test_statistical_consistency_cross_validation(self):
        """Cross-validate statistical consistency across implementations."""
        # Test that statistical relationships are preserved across implementations

        estimate = 12.8
        se = 3.5
        df = 40
        threshold = 0.0

        # Calculate using core modules
        core_t = core_t_stat(estimate, se, threshold)
        core_p = core_p_value(core_t, df)
        core_post_prob = core_posterior_prob(estimate, se, df, threshold)

        # Verify mathematical relationships are preserved
        # For threshold = 0: posterior_prob = 1 - F_t((threshold - estimate) / se)
        from scipy import stats

        t_stat_for_posterior = (threshold - estimate) / se
        expected_post_prob = 1 - stats.t.cdf(t_stat_for_posterior, df)

        np.testing.assert_allclose(
            core_post_prob,
            expected_post_prob,
            rtol=1e-14,
            err_msg="Posterior probability should match statistical relationship",
        )

        # For positive t-statistic: p_value ≈ 2 * (1 - posterior_prob)
        if core_t > 0:
            expected_p_val = 2 * (1 - core_post_prob)
            np.testing.assert_allclose(
                core_p,
                expected_p_val,
                rtol=1e-12,
                err_msg="P-value and posterior probability should satisfy mathematical relationship",
            )


class TestNumericalPrecisionCrossValidation:
    """Cross-validation tests for numerical precision maintenance."""

    def test_extreme_values_cross_validation(self):
        """Cross-validate behavior with challenging but realistic parameter values."""
        # Test with small but realistic values
        test_x_values = np.array([0.01, 0.02, 0.015])
        sigma = 0.001
        var_alpha = 1e-6
        var_beta = 1e-8
        cov_alpha_beta = -1e-7

        core_result = core_cumsd(
            test_x_values=test_x_values,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
        )

        func_result = func_cumsd(
            test_x_values=test_x_values,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
        )

        # Should maintain precision with small values
        np.testing.assert_allclose(
            core_result,
            func_result,
            rtol=1e-12,
            err_msg="Should maintain precision with small realistic values",
        )

        # Test with large but realistic values
        test_x_values_large = np.array([1000, 1100, 1050])
        sigma_large = 50.0
        var_alpha_large = 100.0
        var_beta_large = 0.001
        cov_alpha_beta_large = -0.05  # Realistic covariance

        core_result_large = core_cumsd(
            test_x_values=test_x_values_large,
            sigma=sigma_large,
            var_alpha=var_alpha_large,
            var_beta=var_beta_large,
            cov_alpha_beta=cov_alpha_beta_large,
        )

        func_result_large = func_cumsd(
            test_x_values=test_x_values_large,
            sigma=sigma_large,
            var_alpha=var_alpha_large,
            var_beta=var_beta_large,
            cov_alpha_beta=cov_alpha_beta_large,
        )

        # Should maintain precision with large values
        np.testing.assert_allclose(
            core_result_large,
            func_result_large,
            rtol=1e-12,
            err_msg="Should maintain precision with large realistic values",
        )

    def test_core_negative_variance_handling(self):
        """Test core implementation handles negative variance correctly."""
        # Parameters that cause negative variance
        test_x_values = np.array([1000, 2000, 1500])
        sigma = 50.0
        var_alpha = 100.0
        var_beta = 1e-6
        cov_alpha_beta = -100.0  # Large negative covariance causes negative variance

        with pytest.raises(ValueError, match="Negative variance detected"):
            core_cumsd(test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta)

    def test_functional_negative_variance_handling(self):
        """Test functional implementation handles negative variance correctly."""
        # Parameters that cause negative variance
        test_x_values = np.array([1000, 2000, 1500])
        sigma = 50.0
        var_alpha = 100.0
        var_beta = 1e-6
        cov_alpha_beta = -100.0  # Large negative covariance causes negative variance

        with pytest.raises(ValueError, match="Negative variance detected"):
            func_cumsd(test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta)

    def test_edge_case_cross_validation(self):
        """Cross-validate edge case behavior."""
        # Test single observation
        single_x = np.array([100])

        core_single = core_cumsd(
            test_x_values=single_x,
            sigma=2.0,
            var_alpha=1.0,
            var_beta=0.01,
            cov_alpha_beta=-0.05,
        )

        func_single = func_cumsd(
            test_x_values=single_x,
            sigma=2.0,
            var_alpha=1.0,
            var_beta=0.01,
            cov_alpha_beta=-0.05,
        )

        np.testing.assert_allclose(
            core_single,
            func_single,
            rtol=1e-14,
            err_msg="Single observation should be handled identically",
        )

        # Test zero covariance
        zero_cov_core = core_cumsd(
            test_x_values=np.array([100, 110]),
            sigma=2.0,
            var_alpha=1.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
        )

        zero_cov_func = func_cumsd(
            test_x_values=np.array([100, 110]),
            sigma=2.0,
            var_alpha=1.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
        )

        np.testing.assert_allclose(
            zero_cov_core,
            zero_cov_func,
            rtol=1e-14,
            err_msg="Zero covariance should be handled identically",
        )
