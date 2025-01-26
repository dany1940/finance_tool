# tests/test_main.py

from finance_tool.main import add

def test_add():
    """Test the add function."""
    result = add(2, 3)
    assert result == 5, "Expected 2 + 3 to equal 5"
