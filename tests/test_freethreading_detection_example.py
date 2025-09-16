"""Test the freethreading detection example from README.md."""

import io
from contextlib import redirect_stdout

from woeboost import WoeLearner


def test_freethreading_detection_example():
    """Test the exact example from README.md: print(f"Free-threading detected: {learner.is_freethreaded}")."""
    # Create a WoeLearner instance
    learner = WoeLearner()

    # Test that is_freethreaded is accessible and works
    assert hasattr(learner, "is_freethreaded")
    assert isinstance(learner.is_freethreaded, bool)

    # Test the exact print statement from README
    # We'll capture the output to verify it works
    f = io.StringIO()
    with redirect_stdout(f):
        print(f"Free-threading detected: {learner.is_freethreaded}")

    output = f.getvalue().strip()
    expected = f"Free-threading detected: {learner.is_freethreaded}"
    assert output == expected


def test_freethreading_detection_function():
    """Test that the freethreading detection function works correctly."""
    from woeboost.learner import _detect_freethreading

    # Test that the function returns a boolean
    result = _detect_freethreading()
    assert isinstance(result, bool)

    # Test that it's consistent across calls
    result2 = _detect_freethreading()
    assert result == result2


def test_woelearner_freethreading_property():
    """Test that WoeLearner correctly sets the is_freethreaded property."""
    learner1 = WoeLearner()
    learner2 = WoeLearner()

    # Both instances should have the same freethreading detection
    assert learner1.is_freethreaded == learner2.is_freethreaded

    # The property should be a boolean
    assert isinstance(learner1.is_freethreaded, bool)
    assert isinstance(learner2.is_freethreaded, bool)
