.PHONY: install run test typecheck lint format train

install:
	pip install -r requirements.txt

run:
	PYTHONPATH=src uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

test:
	PYTHONPATH=src pytest -q

typecheck:
	mypy src

lint:
	flake8 src tests

format:
	black src tests app.py

train:
	python -m energy_forecast.train_xgb
	PYTHONPATH=src python -m energy_forecast.train_xgb

.PHONY: build install clean typecheck

CXX_FLAGS = -O3 -Wall -shared -std=c++11 -fPIC
PYTHON_INCLUDE = $(shell python -m pybind11 --includes)

