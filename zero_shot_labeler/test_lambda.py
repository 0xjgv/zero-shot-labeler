import json
import time
from typing import Any

import pytest

from zero_shot_labeler.lambda_handler import handler


def invoke_lambda(payload: dict[str, Any]) -> dict[str, Any]:
    """Simulate a Lambda invocation locally."""
    return handler(payload, None)


def test_basic_classification():
    """Test a basic classification scenario."""
    event = {
        "text": "The customer service was excellent and resolved my issue quickly!",
        "labels": ["positive", "negative", "neutral"],
    }

    response = invoke_lambda(event)

    assert sum(response["scores"].values()) == pytest.approx(1.0, rel=1e-5)
    assert len(response["scores"]) == len(event["labels"])
    assert isinstance(response["duration"], float)
    assert isinstance(response["scores"], dict)
    assert 10 > response["duration"] > 0
    assert "duration" in response
    assert "scores" in response
    assert "labels" in response
    assert "text" in response


def test_invalid_inputs():
    """Test various invalid input scenarios."""
    test_cases = [
        ({"labels": ["positive"]}, "Both 'text' and 'labels' are required"),
        ({"text": 123, "labels": ["positive"]}, "'text' must be a string"),
        ({"text": "Hello"}, "Both 'text' and 'labels' are required"),
        ({}, "Both 'text' and 'labels' are required"),
        (
            {"text": "Hello", "labels": "positive"},
            "'labels' must be a list of strings",
        ),
        (
            {"text": "Hello", "labels": [1, 2, 3]},
            "'labels' must be a list of strings",
        ),
    ]

    for payload, expected_error in test_cases:
        with pytest.raises(ValueError, match=expected_error):
            invoke_lambda(payload)


def test_performance():
    """Test classification performance."""
    event = {
        "text": "Your account has been locked due to multiple failed login attempts.",
        "labels": ["automatic_message", "urgent_message", "churn_message"],
    }

    start_time = time.time()
    response = invoke_lambda(event)
    end_time = time.time()
    measured_duration = end_time - start_time

    print(f"\nClassification took {measured_duration:.2f} seconds")
    print(f"Results: {json.dumps(response, indent=2)}")

    assert "duration" in response
    assert (
        abs(response["duration"] - measured_duration) < 0.1
    )  # Within 100ms of measured time
    assert (
        response["duration"] < 2.0
    )  # Assuming 2 seconds is acceptable for local testing


if __name__ == "__main__":
    """Manual testing section."""
    # Example 1: Basic classification
    event = {
        "text": "The customer service was excellent and resolved my issue quickly!",
        "labels": ["positive", "negative", "neutral"],
    }
    print("\nExample 1: Basic sentiment classification")
    print(json.dumps(invoke_lambda(event), indent=2))

    # Example 2: Message type classification
    event = {
        "text": "Your account has been locked due to multiple failed login attempts.",
        "labels": ["automatic_message", "urgent_message", "churn_message"],
    }
    print("\nExample 2: Message type classification")
    print(json.dumps(invoke_lambda(event), indent=2))
