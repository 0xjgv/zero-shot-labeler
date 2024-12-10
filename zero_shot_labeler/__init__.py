from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import time
from typing import cast

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, pipeline
from transformers.utils import move_cache

# The model is stored in the Docker image at this path
# /var/task/zero_shot_labeler/opt/ml/models
MODELS_DIR_PATH = Path(__file__).parent.parent / "opt/ml/models"
# Default model ID
# https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 (best)
MODEL_ID = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# Multilingual model
# https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0-c
# MODEL_ID = "MoritzLaurer/bge-m3-zeroshot-v2.0-c"

# Embedding model (few shot)
# https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1
# EMBEDDING_MODEL_ID = "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"
# https://huggingface.co/jinaai/jina-embeddings-v3 (best)
EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v3"


@dataclass
class LabelerOutput:
    scores: dict[str, float]
    input_text: str
    model_id: str


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
    def preload_model(cls):
        """Preload the model during container initialization"""
        if MODELS_DIR_PATH.exists():
            print(f"[> {cls.__name__}]:", f"Model already exists at {MODELS_DIR_PATH}")
            return

        print(
            f"[> {cls.__name__}]:",
            f"Preloading model from {MODEL_ID} to {MODELS_DIR_PATH}",
        )
        starting_time = time()
        snapshot_download(
            MODEL_ID,
            allow_patterns=["*.json", "*.safetensors"],  # Only save the model weights
            local_dir=MODELS_DIR_PATH,  # Save the model to the MODEL_PATH
        )

        # Update the cache directory to the new location
        move_cache()
        print(
            f"[> {cls.__name__}]:",
            f"Model preloaded in {time() - starting_time:.2f} seconds",
        )

    def __init__(self, model: str = MODEL_ID, gpu: bool = False):
        starting_time = time()
        if MODELS_DIR_PATH.exists() and (model_path := MODELS_DIR_PATH.as_posix()):
            self.log(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.pipeline = pipeline(
                "zero-shot-classification",
                device="cuda" if gpu else "cpu",
                tokenizer=tokenizer,
                model=model_path,
            )
        else:
            self.log(f"Loading model from {model}")
            tokenizer = AutoTokenizer.from_pretrained(model)
            self.pipeline = pipeline(
                "zero-shot-classification",
                device="cuda" if gpu else "cpu",
                tokenizer=tokenizer,
                model=model,
            )
            self.pipeline.save_pretrained(MODELS_DIR_PATH)
        self.log(f"Model loaded in {time() - starting_time:.2f} seconds")

    def __call__(self, text: str, labels: list[str]) -> LabelerOutput:
        starting_time = time()
        self.log(f"Classifying text: {text}")
        output = cast(dict[str, list], self.pipeline(text, labels))
        self.log(f"Classification in {time() - starting_time:.2f} seconds")
        labels = output.get("labels", [])
        scores = output.get("scores", [])
        score_by_label = {label: score for label, score in zip(labels, scores)}
        return LabelerOutput(scores=score_by_label, model_id=MODEL_ID, input_text=text)


# Preload the model during container initialization.
# This saves time on the first request and allows
# for faster subsequent requests.
preload = ZeroShotLabeler.preload_model

if __name__ == "__main__":
    preload()
