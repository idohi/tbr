"""
Tests for Core Module Initialization.

This module tests the core package initialization and exports,
ensuring that the core regression module is properly accessible
through the core package interface.
"""


class TestCoreModuleImports:
    """Test core module import functionality."""

    def test_core_regression_imports(self):
        """Test that core regression functions can be imported."""
        from tbr.core import (
            calculate_sum_squared_deviations,
            calculate_variances,
            convert_to_integer,
            extract_sum_squared_deviations_from_model,
            fit_regression_model,
        )

        # Verify all functions are callable
        assert callable(fit_regression_model)
        assert callable(calculate_variances)
        assert callable(calculate_sum_squared_deviations)
        assert callable(extract_sum_squared_deviations_from_model)
        assert callable(convert_to_integer)

    def test_core_module_all_exports(self):
        """Test that __all__ exports are properly defined."""
        import tbr.core as core_module

        # Should have __all__ defined
        assert hasattr(core_module, "__all__")

        # All exported functions should be accessible
        for func_name in core_module.__all__:
            assert hasattr(core_module, func_name), f"Missing export: {func_name}"
            assert callable(
                getattr(core_module, func_name)
            ), f"Not callable: {func_name}"

    def test_core_module_docstring(self):
        """Test that core module has proper documentation."""
        import tbr.core as core_module

        # Should have module docstring
        assert core_module.__doc__ is not None
        assert len(core_module.__doc__.strip()) > 0
        assert "TBR Core Modules" in core_module.__doc__
