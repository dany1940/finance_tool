# tests/test_main.py
from financial_models.main import add


def test_add():
    """Test the add function."""
    result = add(2, 3)
    assert result == 5, "Expected 2 + 3 to equal 5"
