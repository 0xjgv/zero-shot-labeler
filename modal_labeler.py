from os import environ

import modal
from modal import Secret, gpu

from zero_shot_labeler import LabelerOutput
from zero_shot_labeler import ZeroShotLabeler as _ZeroShotLabeler

zero_shot_labeler_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers[torch]==4.46.3", "deepl"
)

app = modal.App("zero-shot-labeler", image=zero_shot_labeler_image)

# NVIDIA T4 GPU ($0.000164 / sec)
gpu_config = gpu.T4(count=1)
# # Uncomment to disable GPU
# Physical core ($0.000038 / core / sec)
# gpu_config = None


# @app.cls(gpu=gpu_config, secrets=[Secret.from_name("deepl-api")], keep_warm=1)
@app.cls(gpu=gpu_config, secrets=[Secret.from_name("deepl-api")])
class ZeroShotLabeler:
    @modal.build()
    def preload_model(self):
        _ZeroShotLabeler.preload_model()

    @modal.enter()
    def initialize_model(self):
        from deepl import Translator

        self.labeler = _ZeroShotLabeler(gpu=gpu_config is not None)
        self.translator = Translator(environ["DEEPL_API_KEY"])

    @modal.method()
    def generate_labels(self, text: str, labels: list[str]) -> LabelerOutput:
        return self.labeler(text, labels)

    # TODO: Uncomment to enable web endpoint
    @modal.web_endpoint(method="POST", docs=True)
    def label(self, data: dict) -> LabelerOutput:
        labels = data["labels"]
        text = data["text"]
        if not (labels and text):
            raise ValueError("labels and text must be provided")
        if not isinstance(labels, list):
            raise ValueError("labels must be a list")
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        result = self.translator.translate_text(text, target_lang="EN-US")
        return self.labeler(result.text, labels)


@app.local_entrypoint()
def local_test():
    labels = ["positive", "negative"]
    print(f"Labels: {labels}")

    text = "I really love Modal a lot!"
    print(f"Labeling text: {text}")

    scores = ZeroShotLabeler.generate_labels.remote(text, labels)
    print(f"Labeling result: {scores}")
