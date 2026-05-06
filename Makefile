.PHONY: install lint test run data train train-mlp train-baselines mlflow clean


install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest tests/ -v --tb=short

run:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

data:
	python -m src.data.pipeline

train-mlp:
	python -m src.training.train

train-baselines:
	python -m src.training.train_baselines

train: data train-baselines train-mlp

mlflow:
	mlflow ui --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
