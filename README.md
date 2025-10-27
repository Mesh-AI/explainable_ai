# explainable_ai
Implement Explainable AI in Timeseries Forecast


## Task 1 — Local production-like setup

### Setup
workon explainable_ai
pip install -r requirements.txt

### Quality gates
make test
make typecheck
make lint
make format

### Train once (saves artifacts/xgb_zone1.json + feature_cols_zone1.json)
make train

### Serve locally
make run
# then:
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"row":{"series":[1,2,4]}}'          # fallback path (before training)
# After training, send the full feature row that matches feature_cols_zone1.json.