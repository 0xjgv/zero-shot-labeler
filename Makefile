install:
	poetry config virtualenvs.in-project true
	poetry env use python3.12
	poetry install -vv
	poetry --version

format:
	poetry run ruff check --fix .
	poetry run ruff format .

clean:
	rm -rf .pytest_cache */__pycache__ */*/__pycache__
	poetry run ruff clean

lint: format
	poetry run ruff check .

test:
	poetry run pytest -vv
