from pathlib import Path
from sys import argv
from threading import Lock
from time import time
from typing import NamedTuple, cast

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, pipeline

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# The model is stored in the Docker image at this path
# /var/task/zero_shot_labeler/opt/ml/model
MODEL_PATH = Path(__file__).parent / "opt/ml/model"


LabelerOutput = dict[str, float]


class LabelScore(NamedTuple):
    score: float
    label: str


class ZeroShotLabeler:
    __slots__ = ("pipeline",)
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ZeroShotLabeler, cls).__new__(cls)
        return cls._instance

    def log(self, *args, **kwargs):
        print(f"[> {self.__class__.__name__}]:", *args, **kwargs)

    @classmethod
    def preload_model(cls, *, model: str = DEFAULT_MODEL, force_download: bool = False):
        """Preload the model during container initialization"""
        if MODEL_PATH.exists() and not force_download:
            print(f"[> {cls.__name__}]:", f"Model already exists at {MODEL_PATH}")
            return

        print(f"[> {cls.__name__}]:", f"Preloading model from {model} to {MODEL_PATH}")
        starting_time = time()
        snapshot_download(
            model,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
            local_dir=MODEL_PATH,  # Save the model to the MODEL_PATH
        )
        print(
            f"[> {cls.__name__}]:",
            f"Model preloaded in {time() - starting_time:.2f} seconds",
        )

    def __init__(self, model: str = DEFAULT_MODEL):
        starting_time = time()
        if MODEL_PATH.exists() and (model_path := MODEL_PATH.as_posix()):
            self.log(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.pipeline = pipeline(
                "zero-shot-classification",
                tokenizer=tokenizer,
                model=model_path,
            )
        else:
            self.log(f"Loading model from {model}")
            tokenizer = AutoTokenizer.from_pretrained(model)
            self.pipeline = pipeline(
                "zero-shot-classification",
                tokenizer=tokenizer,
                model=model,
            )
            self.pipeline.save_pretrained(MODEL_PATH)
        self.log(f"Model loaded in {time() - starting_time:.2f} seconds")

    def __call__(self, text: str, labels: list[str]) -> LabelerOutput:
        starting_time = time()
        self.log(f"Classifying text: {text}")
        output = cast(dict[str, list], self.pipeline(text, labels))
        self.log(f"Classification in {time() - starting_time:.2f} seconds")
        labels = output.get("labels", [])
        scores = output.get("scores", [])
        return {label: score for label, score in zip(labels, scores)}


# Preload the model during container initialization.
# This saves time on the first request and allows
# for faster subsequent requests.
preload = ZeroShotLabeler.preload_model

if __name__ == "__main__":
    force_download = "--force-download" in argv
    preload(force_download=force_download)
