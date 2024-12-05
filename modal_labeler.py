import modal
from modal import gpu

from zero_shot_labeler import ZeroShotLabeler as _ZeroShotLabeler

zero_shot_labeler_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers[torch]==4.46.3"
)

app = modal.App("zero-shot-labeler", image=zero_shot_labeler_image)

# NVIDIA T4 GPU ($0.000164 / sec)
gpu_config = gpu.T4(count=1)
# # Uncomment to disable GPU
# Physical core ($0.000038 / core / sec)
# gpu_config = None


@app.cls(gpu=gpu_config)
class ZeroShotLabeler:
    @modal.build()
    def preload_model(self):
        _ZeroShotLabeler.preload_model()

    @modal.enter()
    def initialize_model(self):
        self.labeler = _ZeroShotLabeler(gpu=gpu_config is not None)

    # TODO: Uncomment to enable web endpoint
    # @modal.web_endpoint(method="POST", docs=True)
    # def label(self, data: dict) -> dict[str, float]:
    #     return self.labeler(data["text"], data["labels"])

    @modal.method()
    def generate_labels(self, text: str, labels: list[str]) -> dict[str, float]:
        return self.labeler(text, labels)


@app.local_entrypoint()
def local_test():
    labels = ["positive", "negative"]
    print(f"Labels: {labels}")

    text = "I really love Modal a lot!"
    print(f"Labeling text: {text}")

    scores = ZeroShotLabeler.generate_labels.remote(text, labels)
    print(f"Labeling result: {scores}")
