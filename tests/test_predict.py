# tests/test_predict.py
from energy_forecast import predict


def test_predict_fallback_series() -> None:
    # Works even before training artifacts exist
    out = predict({"series": [1.0, 2.0, 4.0]})
    assert "prediction" in out
    assert isinstance(out["prediction"], float)
