.PHONY: install run test typecheck lint format train sync

create-venv:
	uv venv

install:
	uv pip install .

sync:
	uv sync

run:
	PYTHONPATH=src uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

test:
	uv run pytest -q

typecheck:
	uv run mypy src

lint:
	uv run ruff check --fix src tests

format:
	uv run ruff format src tests

train:
	uv run python -m energy_forecast.train_xgb