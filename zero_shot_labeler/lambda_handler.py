from time import time
from typing import Any

from zero_shot_labeler.labeler import Labeler

labeler = Labeler()


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda handler for zero-shot classification.

    Args:
        event: Must contain 'text' and 'labels' keys
        context: AWS Lambda context (unused)

    Returns:
        Dictionary with classification scores for each label

    Raises:
        ValueError: If required fields are missing or invalid
    """
    labels = event.get("labels")
    text = event.get("text")

    if not text or not labels:
        raise ValueError("Both 'text' and 'labels' are required")
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError("'labels' must be a list of strings")
    if not isinstance(text, str):
        raise ValueError("'text' must be a string")

    start_time = time()
    scores = labeler(text, labels)
    duration = time() - start_time

    # Return classification results
    return {"scores": scores, "text": text, "labels": labels, "duration": duration}
