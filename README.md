# Zero-Shot Labeler README

## Overview

The Zero-Shot Labeler project provides a Python-based solution for text classification using a pretrained Hugging Face zero-shot classification model. The solution efficiently preloads, serves, and classifies text inputs with specified labels. This implementation includes containerization with Docker for streamlined deployment.

## Features
- Containerization: Includes a Docker image for deploying the service in a consistent environment.
- Zero-Shot Classification: Classify text into arbitrary categories without additional training.
- Preloading: Speeds up inference by preloading the model during initialization.
- Singleton Pattern: Ensures only one model instance is loaded at any time.

## Prerequisites
- Python: 3.11
- HuggingFace Transformers: Library used for loading the zero-shot classification pipeline.
- Docker: For building and running the containerized application.
- Poetry: For dependency management and virtual environments.

## Installation

1. Clone the repository
```bash
git clone https://github.com/0xjgv/zero-shot-labeler.git
cd zero-shot-labeler
make install
```
2. Install the dependencies
```bash
make install
```

## Usage

### Preload the Model (optional)
To preload the model and save it locally in the container path (opt/ml/model):
```bash
make preload-model
```

### Build the Docker Image
```bash
make image
```

### Serve the Model
```bash
make run-image
```

### Perform Zero-Shot Classification
1.	Import the Labeler class in your Python script:
```python
from labeler import Labeler
```

2.	Initialize the model and classify text:
```python
labeler = Labeler()
result = labeler("The customer service was excellent!", ["positive", "negative", "neutral"])
print(result)  # {'positive': 0.9, 'neutral': 0.08, 'negative': 0.02}
```

or using the Lambda function (requires a running container `make run-image`):
```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{ \
  "text": "The customer service was excellent and resolved my issue quickly!", \
  "labels": ["positive", "negative", "neutral"] \
}'
```

## How It Works
1.	Model Preloading:
-	Checks if the model exists locally in the opt/ml/model directory.
-	Downloads and caches the model if not already available.
2.	Singleton Implementation:
-	Ensures only one instance of the Labeler class is created, even in multithreaded environments.
3.	Classification Pipeline:
-	Uses Hugging Face’s pipeline to classify input text into specified labels.
4.	Docker Integration:
-	The model is stored inside the container for faster inference.
-	Easily deployable in cloud or local environments.


## Development Notes
- Thread-safe singleton implementation ensures efficient resource usage.
- Customize the model by modifying the DEFAULT_MODEL variable.
- The Labeler class uses __slots__ for memory optimization.

## Testing

Run tests:
```bash
make test
```
Quick endpoint testing (requires a running container `make run-image`):
```bash
make test-endpoint
```
