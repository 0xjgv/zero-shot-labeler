# Load environment variables from .env into the current shell
include .env

install:
	poetry config virtualenvs.in-project true
	poetry env use python3.11
	poetry install -vv

format:
	poetry run ruff check --fix .
	poetry run ruff format .

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	poetry run ruff clean

lint: format
	poetry run ruff check .
	make clean

test:
	poetry run pytest -vv

run-lambda:
	poetry run python playground/test_lambda.py

preload-model:
	poetry run python zero_shot_labeler/__init__.py

image:
	docker build -t zero-shot-labeler --force-rm --progress=plain --no-cache .

image-inspect:
	docker image inspect zero-shot-labeler:latest

image-inspect-size:
	docker run --rm --entrypoint sh zero-shot-labeler:latest -c "du -ah / | sort -rh | head -n 20"

run-image:
	docker run -p 9000:8080 zero-shot-labeler:latest

test-endpoint:
	curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{ \
		"text": "The customer service was excellent and resolved my issue quickly!", \
		"labels": ["positive", "negative", "neutral", "customer_feedback"] \
	}'

run-modal:
	modal serve modal_labeler.py

test-modal-endpoint:
	curl -XPOST -H "Content-Type: application/json" "https://zero-shot-labeler.modal.run" -d '{ \
		"text": "The customer service was excellent and resolved my issue quickly!", \
		"labels": ["positive", "negative", "neutral", "customer_feedback"] \
	}'

modal-deploy:
	modal deploy modal_labeler.py

# Configure Poetry with PyPI token from .env
config-pypi:
	@poetry config pypi-token.pypi $(POETRY_PYPI_TOKEN)

# Publish to PyPI
publish: config-pypi
	poetry publish --build