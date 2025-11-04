"""
Tests for TBR result export utilities.

This module tests export functionality for all TBR result objects,
including JSON and CSV export with metadata preservation.
"""

import json

import numpy as np
import pandas as pd
import pytest

from tbr.core.results import TBRPredictionResult, TBRSubintervalResult, TBRSummaryResult
from tbr.utils.export import (
    export_to_csv,
    export_to_json,
    load_json,
    safe_json_serialize,
)


class TestSafeJsonSerialize:
    """Test safe_json_serialize function with various data types."""

    def test_none_value(self):
        """Test serialization of None."""
        assert safe_json_serialize(None) is None

    def test_numpy_scalars(self):
        """Test serialization of numpy scalar types."""
        assert safe_json_serialize(np.int64(42)) == 42
        assert safe_json_serialize(np.int32(10)) == 10
        assert safe_json_serialize(np.float64(3.14)) == 3.14
        # float32 has limited precision, use approximate comparison
        assert abs(safe_json_serialize(np.float32(2.71)) - 2.71) < 0.01
        assert safe_json_serialize(np.bool_(True)) is True
        assert safe_json_serialize(np.bool_(False)) is False

    def test_numpy_array(self):
        """Test serialization of numpy arrays."""
        arr = np.array([1, 2, 3])
        result = safe_json_serialize(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_pandas_dataframe(self):
        """Test serialization of pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = safe_json_serialize(df)

        assert isinstance(result, dict)
        assert "data" in result
        assert "columns" in result
        assert "index" in result
        assert result["columns"] == ["a", "b"]

    def test_pandas_series(self):
        """Test serialization of pandas Series."""
        series = pd.Series([1, 2, 3], name="test_series")
        result = safe_json_serialize(series)

        assert isinstance(result, dict)
        assert "values" in result
        assert "name" in result
        assert "index" in result
        assert result["name"] == "test_series"
        assert result["values"] == [1, 2, 3]

    def test_pandas_timestamp(self):
        """Test serialization of pandas Timestamp."""
        ts = pd.Timestamp("2023-01-01")
        result = safe_json_serialize(ts)
        assert isinstance(result, str)
        assert "2023-01-01" in result

    def test_datetime_index(self):
        """Test serialization of DatetimeIndex."""
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        result = safe_json_serialize(idx)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_integer_index(self):
        """Test serialization of integer Index."""
        idx = pd.Index([1, 2, 3, 4, 5])
        result = safe_json_serialize(idx)
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]

    def test_range_index(self):
        """Test serialization of RangeIndex."""
        idx = pd.RangeIndex(start=0, stop=5, step=1)
        result = safe_json_serialize(idx)
        assert isinstance(result, list)
        assert result == [0, 1, 2, 3, 4]

    def test_dict_recursive(self):
        """Test recursive serialization of nested dictionaries."""
        data = {
            "scalar": np.int64(42),
            "array": np.array([1, 2, 3]),
            "nested": {"value": np.float64(3.14)},
        }
        result = safe_json_serialize(data)

        assert result["scalar"] == 42
        assert result["array"] == [1, 2, 3]
        assert result["nested"]["value"] == 3.14

    def test_list_recursive(self):
        """Test recursive serialization of lists."""
        data = [np.int64(1), np.array([2, 3]), {"value": np.float64(4.0)}]
        result = safe_json_serialize(data)

        assert result[0] == 1
        assert result[1] == [2, 3]
        assert result[2]["value"] == 4.0

    def test_json_safe_types(self):
        """Test that JSON-safe types pass through unchanged."""
        assert safe_json_serialize(42) == 42
        assert safe_json_serialize(3.14) == 3.14
        assert safe_json_serialize("test") == "test"
        assert safe_json_serialize(True) is True
        assert safe_json_serialize([1, 2, 3]) == [1, 2, 3]
        assert safe_json_serialize({"a": 1}) == {"a": 1}


class TestExportToJson:
    """Test export_to_json function with various objects."""

    def test_export_dict_with_metadata(self, tmp_path):
        """Test exporting dictionary with metadata."""
        data = {"value": 42, "name": "test"}
        filepath = tmp_path / "test.json"

        export_to_json(data, str(filepath), include_metadata=True)

        assert filepath.exists()
        with open(filepath) as f:
            result = json.load(f)

        assert "type" in result
        assert "data" in result
        assert result["type"] == "dict"
        assert result["data"] == data

    def test_export_dict_without_metadata(self, tmp_path):
        """Test exporting dictionary without metadata."""
        data = {"value": 42, "name": "test"}
        filepath = tmp_path / "test.json"

        export_to_json(data, str(filepath), include_metadata=False)

        with open(filepath) as f:
            result = json.load(f)

        assert result == data
        assert "type" not in result

    def test_export_dataframe(self, tmp_path):
        """Test exporting pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        filepath = tmp_path / "test.json"

        export_to_json(df, str(filepath))

        assert filepath.exists()
        with open(filepath) as f:
            result = json.load(f)

        assert result["type"] == "DataFrame"
        assert "data" in result

    def test_export_dataframe_directly(self, tmp_path):
        """Test exporting DataFrame directly (not via to_dict)."""
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        filepath = tmp_path / "direct.json"

        # This tests the branch where obj is a DataFrame
        export_to_json(df, str(filepath), include_metadata=True)

        with open(filepath) as f:
            result = json.load(f)

        # With metadata, should have type and data fields
        assert result["type"] == "DataFrame"
        assert "data" in result

    def test_export_with_numpy_arrays(self, tmp_path):
        """Test exporting dictionary with numpy arrays."""
        data = {"array": np.array([1, 2, 3]), "value": np.int64(42)}
        filepath = tmp_path / "test.json"

        export_to_json(data, str(filepath))

        with open(filepath) as f:
            result = json.load(f)

        # Should be JSON-serializable
        assert result["data"]["array"] == [1, 2, 3]
        assert result["data"]["value"] == 42

    def test_export_compact_format(self, tmp_path):
        """Test exporting with compact formatting."""
        data = {"a": 1, "b": 2}
        filepath = tmp_path / "test.json"

        export_to_json(data, str(filepath), indent=None)

        # Compact format should have no newlines
        content = filepath.read_text()
        assert "\n" not in content

    def test_export_with_path_object(self, tmp_path):
        """Test exporting with Path object as filepath."""
        data = {"test": 123}
        filepath = tmp_path / "test.json"

        export_to_json(data, filepath)  # Pass Path object directly

        assert filepath.exists()

    def test_export_invalid_type_error(self, tmp_path):
        """Test error handling for unsupported types."""
        filepath = tmp_path / "test.json"

        with pytest.raises(TypeError, match="Cannot export object"):
            export_to_json(123, str(filepath))  # Scalar, not dict/DataFrame

    def test_export_result_object_with_to_dict(self, tmp_path):
        """Test exporting result object with to_dict method."""
        # Create a TBRSummaryResult
        result = TBRSummaryResult(
            estimate=100.0,
            lower=80.0,
            upper=120.0,
            se=10.0,
            prob=0.95,
            precision=20.0,
            level=0.90,
            threshold=0.0,
            alpha=5.0,
            beta=1.2,
            sigma=8.0,
            var_alpha=0.5,
            var_beta=0.1,
            cov_alpha_beta=0.05,
            degrees_freedom=30,
        )
        filepath = tmp_path / "result.json"

        export_to_json(result, str(filepath))

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded["type"] == "TBRSummaryResult"
        assert loaded["data"]["estimate"] == 100.0


class TestExportToCsv:
    """Test export_to_csv function with various objects."""

    def test_export_dataframe_with_index(self, tmp_path):
        """Test exporting DataFrame with index."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        filepath = tmp_path / "test.csv"

        export_to_csv(df, str(filepath), include_index=True)

        assert filepath.exists()
        loaded = pd.read_csv(filepath, index_col=0)
        pd.testing.assert_frame_equal(loaded, df)

    def test_export_dataframe_without_index(self, tmp_path):
        """Test exporting DataFrame without index."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        filepath = tmp_path / "test.csv"

        export_to_csv(df, str(filepath), include_index=False)

        loaded = pd.read_csv(filepath)
        pd.testing.assert_frame_equal(loaded, df)

    def test_export_with_custom_separator(self, tmp_path):
        """Test exporting with custom separator."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        filepath = tmp_path / "test.tsv"

        export_to_csv(df, str(filepath), include_index=False, sep="\t")

        content = filepath.read_text()
        assert "\t" in content

    def test_export_result_with_to_dataframe(self, tmp_path):
        """Test exporting result object with to_dataframe method."""
        result = TBRSummaryResult(
            estimate=100.0,
            lower=80.0,
            upper=120.0,
            se=10.0,
            prob=0.95,
            precision=20.0,
            level=0.90,
            threshold=0.0,
            alpha=5.0,
            beta=1.2,
            sigma=8.0,
            var_alpha=0.5,
            var_beta=0.1,
            cov_alpha_beta=0.05,
            degrees_freedom=30,
        )
        filepath = tmp_path / "result.csv"

        export_to_csv(result, str(filepath), include_index=False)

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert "estimate" in loaded.columns
        assert loaded["estimate"].iloc[0] == 100.0

    def test_export_prediction_result(self, tmp_path):
        """Test exporting TBRPredictionResult."""
        predictions_df = pd.DataFrame(
            {"pred": [1.0, 2.0, 3.0], "predsd": [0.1, 0.2, 0.3]}
        )
        result = TBRPredictionResult(
            predictions=predictions_df,
            n_predictions=3,
            model_params={"alpha": 1.0, "beta": 0.5},
            control_values=np.array([10, 20, 30]),
        )
        filepath = tmp_path / "predictions.csv"

        export_to_csv(result, str(filepath), include_index=False)

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert "pred" in loaded.columns
        assert "predsd" in loaded.columns

    def test_export_with_path_object(self, tmp_path):
        """Test exporting with Path object."""
        df = pd.DataFrame({"a": [1, 2]})
        filepath = tmp_path / "test.csv"

        export_to_csv(df, filepath, include_index=False)

        assert filepath.exists()

    def test_export_invalid_type_error(self, tmp_path):
        """Test error handling for unsupported types."""
        filepath = tmp_path / "test.csv"

        with pytest.raises(TypeError, match="Cannot export object"):
            export_to_csv({"not": "dataframe"}, str(filepath))


class TestLoadJson:
    """Test load_json function."""

    def test_load_with_metadata(self, tmp_path):
        """Test loading JSON with metadata."""
        data = {"type": "TestType", "data": {"value": 42}}
        filepath = tmp_path / "test.json"

        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = load_json(str(filepath))
        assert loaded["type"] == "TestType"
        assert loaded["data"]["value"] == 42

    def test_load_without_metadata(self, tmp_path):
        """Test loading JSON without metadata."""
        data = {"value": 42, "name": "test"}
        filepath = tmp_path / "test.json"

        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = load_json(str(filepath))
        assert loaded == data

    def test_load_with_path_object(self, tmp_path):
        """Test loading with Path object."""
        data = {"test": 123}
        filepath = tmp_path / "test.json"

        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = load_json(filepath)  # Pass Path object
        assert loaded == data


class TestResultObjectExportMethods:
    """Test export methods on result objects."""

    def test_prediction_result_to_json(self, tmp_path):
        """Test TBRPredictionResult.to_json()."""
        predictions_df = pd.DataFrame({"pred": [1.0, 2.0], "predsd": [0.1, 0.2]})
        result = TBRPredictionResult(
            predictions=predictions_df,
            n_predictions=2,
            model_params={"alpha": 1.0},
            control_values=np.array([10, 20]),
        )
        filepath = tmp_path / "pred.json"

        result.to_json(str(filepath))

        assert filepath.exists()
        loaded = load_json(filepath)
        assert loaded["type"] == "TBRPredictionResult"

    def test_prediction_result_to_csv(self, tmp_path):
        """Test TBRPredictionResult.to_csv()."""
        predictions_df = pd.DataFrame({"pred": [1.0, 2.0], "predsd": [0.1, 0.2]})
        result = TBRPredictionResult(
            predictions=predictions_df,
            n_predictions=2,
            model_params={"alpha": 1.0},
            control_values=np.array([10, 20]),
        )
        filepath = tmp_path / "pred.csv"

        result.to_csv(str(filepath), index=False)

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert "pred" in loaded.columns

    def test_summary_result_to_json(self, tmp_path):
        """Test TBRSummaryResult.to_json()."""
        result = TBRSummaryResult(
            estimate=100.0,
            lower=80.0,
            upper=120.0,
            se=10.0,
            prob=0.95,
            precision=20.0,
            level=0.90,
            threshold=0.0,
            alpha=5.0,
            beta=1.2,
            sigma=8.0,
            var_alpha=0.5,
            var_beta=0.1,
            cov_alpha_beta=0.05,
            degrees_freedom=30,
        )
        filepath = tmp_path / "summary.json"

        result.to_json(str(filepath))

        assert filepath.exists()
        loaded = load_json(filepath)
        assert loaded["type"] == "TBRSummaryResult"
        assert loaded["data"]["estimate"] == 100.0

    def test_summary_result_to_csv(self, tmp_path):
        """Test TBRSummaryResult.to_csv()."""
        result = TBRSummaryResult(
            estimate=100.0,
            lower=80.0,
            upper=120.0,
            se=10.0,
            prob=0.95,
            precision=20.0,
            level=0.90,
            threshold=0.0,
            alpha=5.0,
            beta=1.2,
            sigma=8.0,
            var_alpha=0.5,
            var_beta=0.1,
            cov_alpha_beta=0.05,
            degrees_freedom=30,
        )
        filepath = tmp_path / "summary.csv"

        result.to_csv(str(filepath), index=False)

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert loaded["estimate"].iloc[0] == 100.0

    def test_subinterval_result_to_json(self, tmp_path):
        """Test TBRSubintervalResult.to_json()."""
        result = TBRSubintervalResult(
            estimate=50.0,
            lower=40.0,
            upper=60.0,
            se=5.0,
            ci_level=0.90,
            start_day=1,
            end_day=7,
            n_days=7,
        )
        filepath = tmp_path / "subinterval.json"

        result.to_json(str(filepath))

        assert filepath.exists()
        loaded = load_json(filepath)
        assert loaded["type"] == "TBRSubintervalResult"
        assert loaded["data"]["estimate"] == 50.0

    def test_subinterval_result_to_csv(self, tmp_path):
        """Test TBRSubintervalResult.to_csv()."""
        result = TBRSubintervalResult(
            estimate=50.0,
            lower=40.0,
            upper=60.0,
            se=5.0,
            ci_level=0.90,
            start_day=1,
            end_day=7,
            n_days=7,
        )
        filepath = tmp_path / "subinterval.csv"

        result.to_csv(str(filepath), index=False)

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert loaded["estimate"].iloc[0] == 50.0
        assert loaded["start_day"].iloc[0] == 1


class TestEndToEndExport:
    """Test end-to-end export workflows."""

    def test_export_and_load_roundtrip(self, tmp_path):
        """Test exporting and loading back JSON."""
        result = TBRSummaryResult(
            estimate=100.0,
            lower=80.0,
            upper=120.0,
            se=10.0,
            prob=0.95,
            precision=20.0,
            level=0.90,
            threshold=0.0,
            alpha=5.0,
            beta=1.2,
            sigma=8.0,
            var_alpha=0.5,
            var_beta=0.1,
            cov_alpha_beta=0.05,
            degrees_freedom=30,
        )
        filepath = tmp_path / "roundtrip.json"

        # Export
        result.to_json(str(filepath))

        # Load and verify
        loaded = load_json(filepath)
        assert loaded["data"]["estimate"] == result.estimate
        assert loaded["data"]["prob"] == result.prob

    def test_multiple_formats_same_data(self, tmp_path):
        """Test exporting same data to multiple formats."""
        result = TBRSummaryResult(
            estimate=100.0,
            lower=80.0,
            upper=120.0,
            se=10.0,
            prob=0.95,
            precision=20.0,
            level=0.90,
            threshold=0.0,
            alpha=5.0,
            beta=1.2,
            sigma=8.0,
            var_alpha=0.5,
            var_beta=0.1,
            cov_alpha_beta=0.05,
            degrees_freedom=30,
        )

        # Export to JSON
        json_file = tmp_path / "data.json"
        result.to_json(str(json_file))

        # Export to CSV
        csv_file = tmp_path / "data.csv"
        result.to_csv(str(csv_file), index=False)

        # Both should exist and contain same core data
        assert json_file.exists()
        assert csv_file.exists()

        json_data = load_json(json_file)
        csv_data = pd.read_csv(csv_file)

        assert json_data["data"]["estimate"] == csv_data["estimate"].iloc[0]
