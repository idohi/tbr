"""Unit tests for TBR package constants."""


from tbr.utils.constants import CONTROL_VAL, TEST_VAL


class TestConstants:
    """Test cases for package constants."""

    def test_control_val_type_and_value(self) -> None:
        """Test CONTROL_VAL is correct string value."""
        assert isinstance(CONTROL_VAL, str)
        assert CONTROL_VAL == "control"

    def test_test_val_type_and_value(self) -> None:
        """Test TEST_VAL is correct string value."""
        assert isinstance(TEST_VAL, str)
        assert TEST_VAL == "test"

    def test_constants_are_different(self) -> None:
        """Test that constants have different values."""
        assert CONTROL_VAL != TEST_VAL

    def test_constants_immutability(self) -> None:
        """Test that constants maintain their values."""
        # This test ensures constants aren't accidentally modified
        original_control = CONTROL_VAL
        original_test = TEST_VAL

        # Constants should remain the same
        assert CONTROL_VAL == original_control
        assert TEST_VAL == original_test
