import modal

from zero_shot_labeler import ZeroShotLabeler

zero_shot_labeler_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers[torch]==4.46.3")
    .run_function(ZeroShotLabeler.preload_model)
)

app = modal.App("zero-shot-labeler", image=zero_shot_labeler_image)


@app.cls()
class Labeler:
    @modal.enter()
    def setup_labeler(self):
        self.zero_shot_labeler = ZeroShotLabeler()

    @modal.method()
    def label(self, text: str, labels: list[str]) -> dict[str, float]:
        return self.zero_shot_labeler(text, labels)


@app.function()
def labeler(text: str, labels: list[str]) -> dict[str, float]:
    labeler = Labeler()
    scores = labeler.label.remote(text, labels)
    print("This code is running on a remote worker!")
    return scores


@app.local_entrypoint()
def main():
    labels = ["positive", "negative"]
    text = "I love Modal"
    print(f"Labeling text: {text}")
    print(f"Labels: {labels}")

    scores = labeler.remote(text, labels)
    print(f"Labeling result: {scores}")
