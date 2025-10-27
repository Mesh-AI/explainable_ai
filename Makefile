.PHONY: install run test typecheck lint format train

install:
	pip install -r requirements.txt

run:
	uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

test:
	pytest -q

typecheck:
	mypy src

lint:
	flake8 src tests

format:
	black src tests app.py

train:
	python -m energy_forecast.train_xgb