# tests/test_predict.py
from energy_forecast import predict  # type: ignore[import-untyped]


def test_predict_fallback_series() -> None:
    """Test that predict works with fallback when no trained model exists."""
    # Works before training artifacts exist
    out = predict({"series": [1.0, 2.0, 4.0]})
    assert "prediction" in out
    assert isinstance(out["prediction"], (int, float))
    assert not isinstance(out["prediction"], bool)  # Ensure it's not a bool subclass
