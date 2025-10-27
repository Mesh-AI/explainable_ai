# explainable_ai
Implement Explainable AI in Timeseries Forecast


## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
make test
make typecheck
make lint
make run    # serves uvicorn app:app on :8000
