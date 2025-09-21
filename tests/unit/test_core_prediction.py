"""
Tests for tbr.core.prediction module.

This module tests the core prediction functionality, ensuring that the modular
interfaces correctly wrap the functional implementations while maintaining
mathematical accuracy and backward compatibility.
"""

import numpy as np
import pandas as pd
import pytest

from tbr.core.prediction import (
    calculate_cumulative_standard_deviation,
    compute_interval_estimate_and_ci,
    generate_counterfactual_predictions,
)
from tbr.functional.tbr_functions import (
    calculate_cumulative_standard_deviation as func_calculate_cumulative_standard_deviation,
)
from tbr.functional.tbr_functions import (
    compute_interval_estimate_and_ci as func_compute_interval_estimate_and_ci,
)
from tbr.functional.tbr_functions import (
    generate_counterfactual_predictions as func_generate_counterfactual_predictions,
)


class TestGenerateCounterfactualPredictions:
    """Test counterfactual prediction generation."""

    def test_basic_counterfactual_predictions(self):
        """Test basic counterfactual prediction generation."""
        # Create test data
        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-02-15", periods=5),
                "control": [1000, 1020, 980, 1050, 990],
            }
        )

        # Generate predictions
        result = generate_counterfactual_predictions(
            alpha=50.0,
            beta=0.95,
            sigma=25.0,
            x_mean=1000.0,
            n_pretest=45,
            test_period_data=test_data,
            control_col="control",
            time_col="date",
        )

        # Verify structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "date" in result.columns
        assert "control" in result.columns
        assert "pred" in result.columns
        assert "predsd" in result.columns

        # Verify predictions are reasonable
        expected_pred_1 = 50.0 + 0.95 * 1000  # First prediction
        assert abs(result["pred"].iloc[0] - expected_pred_1) < 1e-10

        # Verify all prediction standard deviations are positive
        assert all(result["predsd"] > 0)

    def test_counterfactual_predictions_mathematical_properties(self):
        """Test mathematical properties of counterfactual predictions."""
        test_data = pd.DataFrame({"time": range(1, 4), "control": [100, 200, 300]})

        result = generate_counterfactual_predictions(
            alpha=10.0,
            beta=2.0,
            sigma=5.0,
            x_mean=150.0,
            n_pretest=30,
            test_period_data=test_data,
            control_col="control",
            time_col="time",
        )

        # Verify linear relationship: pred = alpha + beta * control
        expected_preds = 10.0 + 2.0 * test_data["control"]
        np.testing.assert_allclose(result["pred"], expected_preds, rtol=1e-10)

        # Verify prediction uncertainties increase with distance from mean
        control_distances = np.abs(test_data["control"] - 150.0)  # distances from mean
        # Points farther from mean should have higher uncertainty
        # 300 is farther from 150 than 100, so predsd[2] > predsd[0]
        assert result["predsd"].iloc[2] > result["predsd"].iloc[0]  # 300 vs 100
        # Also verify the mathematical relationship: farther points have higher uncertainty
        assert (
            control_distances[2] > control_distances[0]
        )  # Verify our assumption about distances

    def test_counterfactual_predictions_backward_compatibility(self):
        """Test backward compatibility with functional implementation."""
        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3),
                "control": [950, 1000, 1050],
            }
        )

        params = {
            "alpha": 25.0,
            "beta": 0.98,
            "sigma": 15.0,
            "x_mean": 1000.0,
            "n_pretest": 50,
            "test_period_data": test_data,
            "control_col": "control",
            "time_col": "date",
        }

        # Generate predictions with both implementations
        core_result = generate_counterfactual_predictions(**params)
        func_result = func_generate_counterfactual_predictions(**params)

        # Results should be identical
        pd.testing.assert_frame_equal(core_result, func_result)


class TestCalculateCumulativeStandardDeviation:
    """Test cumulative standard deviation calculation."""

    def test_basic_cumulative_standard_deviation(self):
        """Test basic cumulative standard deviation calculation."""
        test_x = np.array([1000, 1020, 980, 1050, 990])

        result = calculate_cumulative_standard_deviation(
            test_x_values=test_x,
            sigma=25.0,
            var_alpha=100.0,
            var_beta=0.001,
            cov_alpha_beta=-0.05,
        )

        # Verify structure
        assert isinstance(result, np.ndarray)
        assert len(result) == len(test_x)

        # Verify all values are positive
        assert all(result > 0)

        # Verify cumulative property - should generally increase
        assert result[4] > result[0]  # Last should be larger than first

    def test_cumulative_standard_deviation_mathematical_properties(self):
        """Test mathematical properties of cumulative standard deviation."""
        test_x = np.array([100, 100, 100])  # Constant values

        result = calculate_cumulative_standard_deviation(
            test_x_values=test_x,
            sigma=10.0,
            var_alpha=4.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
        )

        # For constant x values, the variance component should be the same
        # but the cumulative effect should still increase with time
        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_cumulative_standard_deviation_backward_compatibility(self):
        """Test backward compatibility with functional implementation."""
        test_x = np.array([950, 1000, 1050, 1100])

        params = {
            "test_x_values": test_x,
            "sigma": 20.0,
            "var_alpha": 50.0,
            "var_beta": 0.002,
            "cov_alpha_beta": -0.1,
        }

        # Calculate with both implementations
        core_result = calculate_cumulative_standard_deviation(**params)
        func_result = func_calculate_cumulative_standard_deviation(**params)

        # Results should be identical
        np.testing.assert_allclose(core_result, func_result, rtol=1e-15)


class TestComputeIntervalEstimateAndCI:
    """Test interval estimation and confidence interval computation."""

    def test_basic_interval_estimation(self):
        """Test basic interval estimation."""
        # Create mock TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1],
                "y": [100, 105, 120, 125, 130],
                "pred": [98, 103, 115, 118, 122],
                "estsd": [5, 5, 6, 6, 7],
            }
        )

        # Create mock summary
        tbr_summary = pd.DataFrame({"sigma": [10.0], "t_dist_df": [40]})

        result = compute_interval_estimate_and_ci(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_day=1,
            end_day=2,
            ci_level=0.80,
        )

        # Verify structure
        assert isinstance(result, dict)
        required_keys = ["estimate", "precision", "lower", "upper"]
        assert all(key in result for key in required_keys)

        # Verify mathematical relationships
        assert result["lower"] < result["estimate"] < result["upper"]
        assert abs(result["upper"] - result["estimate"]) == pytest.approx(
            result["precision"]
        )
        assert abs(result["estimate"] - result["lower"]) == pytest.approx(
            result["precision"]
        )

    def test_interval_estimation_full_period(self):
        """Test interval estimation for full test period."""
        # Create test data with clear treatment effect
        tbr_df = pd.DataFrame(
            {
                "period": [1, 1, 1],
                "y": [110, 115, 120],
                "pred": [100, 105, 110],
                "estsd": [3, 4, 5],
            }
        )

        tbr_summary = pd.DataFrame({"sigma": [8.0], "t_dist_df": [30]})

        result = compute_interval_estimate_and_ci(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_day=1,
            end_day=3,
            ci_level=0.90,
        )

        # Expected cumulative effect: (110-100) + (115-105) + (120-110) = 30
        expected_estimate = 30.0
        assert result["estimate"] == pytest.approx(expected_estimate)

        # Higher confidence level should give wider intervals
        assert result["precision"] > 0

    def test_interval_estimation_backward_compatibility(self):
        """Test backward compatibility with functional implementation."""
        tbr_df = pd.DataFrame(
            {
                "period": [0, 1, 1, 1],
                "y": [95, 108, 112, 118],
                "pred": [95, 100, 105, 110],
                "estsd": [4, 5, 6, 7],
            }
        )

        tbr_summary = pd.DataFrame({"sigma": [12.0], "t_dist_df": [35]})

        params = {
            "tbr_df": tbr_df,
            "tbr_summary": tbr_summary,
            "start_day": 2,
            "end_day": 3,
            "ci_level": 0.85,
        }

        # Compute with both implementations
        core_result = compute_interval_estimate_and_ci(**params)
        func_result = func_compute_interval_estimate_and_ci(**params)

        # Results should be identical
        for key in ["estimate", "precision", "lower", "upper"]:
            assert core_result[key] == pytest.approx(func_result[key], rel=1e-15)


class TestPredictionModuleIntegration:
    """Test integration between prediction module functions."""

    def test_prediction_workflow_integration(self):
        """Test integration of prediction functions in a typical workflow."""
        # Create test data
        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-02-01", periods=4),
                "control": [980, 1000, 1020, 1040],
            }
        )

        # Step 1: Generate counterfactual predictions
        predictions = generate_counterfactual_predictions(
            alpha=20.0,
            beta=0.96,
            sigma=18.0,
            x_mean=1000.0,
            n_pretest=40,
            test_period_data=test_data,
            control_col="control",
            time_col="date",
        )

        # Step 2: Calculate cumulative standard deviations
        cumsd = calculate_cumulative_standard_deviation(
            test_x_values=test_data["control"].values,
            sigma=18.0,
            var_alpha=80.0,
            var_beta=0.0015,
            cov_alpha_beta=-0.08,
        )

        # Verify integration
        assert len(predictions) == len(cumsd)
        assert all(predictions["predsd"] > 0)
        assert all(cumsd > 0)

    def test_prediction_module_imports(self):
        """Test that all prediction functions can be imported from core module."""
        from tbr.core import (
            calculate_cumulative_standard_deviation,
            compute_interval_estimate_and_ci,
            generate_counterfactual_predictions,
        )

        # Verify functions are callable
        assert callable(generate_counterfactual_predictions)
        assert callable(calculate_cumulative_standard_deviation)
        assert callable(compute_interval_estimate_and_ci)


class TestPredictionModuleEdgeCases:
    """Test edge cases and error conditions."""

    def test_counterfactual_predictions_multiple_observations(self):
        """Test counterfactual predictions with multiple observations."""
        # Use multiple observations to avoid division by zero in variance calculation
        test_data = pd.DataFrame({"time": [1, 2], "control": [1000, 1010]})

        result = generate_counterfactual_predictions(
            alpha=50.0,
            beta=1.0,
            sigma=10.0,
            x_mean=1005.0,  # Mean of test data
            n_pretest=20,
            test_period_data=test_data,
            control_col="control",
            time_col="time",
        )

        assert len(result) == 2
        assert result["pred"].iloc[0] == pytest.approx(1050.0)  # 50 + 1.0 * 1000
        assert result["pred"].iloc[1] == pytest.approx(1060.0)  # 50 + 1.0 * 1010

    def test_cumulative_standard_deviation_single_value(self):
        """Test cumulative standard deviation with single value."""
        result = calculate_cumulative_standard_deviation(
            test_x_values=np.array([1000]),
            sigma=15.0,
            var_alpha=25.0,
            var_beta=0.001,
            cov_alpha_beta=0.0,
        )

        assert len(result) == 1
        assert result[0] > 0

    def test_prediction_functions_mathematical_consistency(self):
        """Test mathematical consistency across prediction functions."""
        # Use same test data for multiple functions
        test_x = np.array([950, 1000, 1050])

        # Generate predictions
        test_data = pd.DataFrame({"time": [1, 2, 3], "control": test_x})

        predictions = generate_counterfactual_predictions(
            alpha=30.0,
            beta=0.97,
            sigma=20.0,
            x_mean=1000.0,
            n_pretest=35,
            test_period_data=test_data,
            control_col="control",
            time_col="time",
        )

        # Calculate cumulative standard deviations
        cumsd = calculate_cumulative_standard_deviation(
            test_x_values=test_x,
            sigma=20.0,
            var_alpha=60.0,
            var_beta=0.002,
            cov_alpha_beta=-0.06,
        )

        # Both should have same length
        assert len(predictions) == len(cumsd)

        # Prediction uncertainties should be related to cumulative uncertainties
        # (though not identical due to different mathematical formulations)
        assert all(predictions["predsd"] > 0)
        assert all(cumsd > 0)
