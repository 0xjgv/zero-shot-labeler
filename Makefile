install:
	poetry config virtualenvs.in-project true
	poetry env use python3.11
	poetry install -vv

format:
	poetry run ruff check --fix .
	poetry run ruff format .

clean:
	rm -rf .pytest_cache */__pycache__ */*/__pycache__
	poetry run ruff clean

lint: format clean
	poetry run ruff check .

test:
	poetry run pytest -vv

test-lambda:
	poetry run pytest playground/test_lambda.py -v

run-lambda:
	poetry run python playground/test_lambda.py

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
		"labels": ["positive", "negative", "neutral"] \
	}'