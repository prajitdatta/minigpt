# MiniGPT-Forge Makefile
# Author: Prajit Datta (https://github.com/prajitdatta)

.PHONY: install install-dev test lint format clean docker

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short --cov=minigpt --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -x

lint:
	ruff check minigpt/ tests/ scripts/
	mypy minigpt/ --ignore-missing-imports

format:
	black minigpt/ tests/ scripts/
	ruff check --fix minigpt/ tests/ scripts/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker:
	docker build -t minigpt-forge .

train-shakespeare:
	python scripts/download_data.py shakespeare
	python scripts/train.py --config examples/configs/train_shakespeare.yaml

benchmark:
	python scripts/benchmark.py

serve:
	python -m minigpt.api.server --checkpoint checkpoints/best_model.pt --port 8000
