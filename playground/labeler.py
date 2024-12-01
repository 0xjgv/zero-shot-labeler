from pathlib import Path
from threading import Lock
from time import time
from typing import cast

from transformers import pipeline

default_model = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
default_device = "cpu"

MODEL_PATH = Path(__file__).parent / "opt/ml/model"


class Labeler:
    __slots__ = ("pipeline",)
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Labeler, cls).__new__(cls)
        return cls._instance

    @classmethod
    def preload_model(cls, model: str = default_model, device: str = default_device):
        """Preload the model during container initialization"""
        if MODEL_PATH.exists():
            print(f"Model already exists at {MODEL_PATH}")
            return

        print(f"Preloading model from {model} to {MODEL_PATH}")
        starting_time = time()
        pipeline(
            "zero-shot-classification", model=model, device=device
        ).save_pretrained(MODEL_PATH)
        print(f"Model preloaded in {time() - starting_time:.2f} seconds")

    def __init__(self, model: str = default_model, device: str = default_device):
        if not hasattr(self, "pipeline"):
            model_path = MODEL_PATH.as_posix() if MODEL_PATH.exists() else model
            starting_time = time()
            print(f"Loading model from {model_path}")
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=model_path,
                device=device,
            )
            print(f"Model loaded in {time() - starting_time:.2f} seconds")

    def __call__(self, text: str, labels: list[str]) -> dict[str, float]:
        starting_time = time()
        print(f"Classifying text: {text}")
        output = cast(dict[str, list], self.pipeline(text, labels))
        print(f"Classification in {time() - starting_time:.2f} seconds")
        labels = output.get("labels", [])
        scores = output.get("scores", [])
        return {label: score for label, score in zip(labels, scores)}


# Preload the model during container initialization.
# This saves time on the first request and allows
# for faster subsequent requests.
preload = Labeler.preload_model
