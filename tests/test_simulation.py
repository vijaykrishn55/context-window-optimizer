from context_optimizer.simulation import run_position_simulation
from context_optimizer.tokenizer import ApproximateCodec


def test_mock_simulation_exports_all_required_depths():
    report = run_position_simulation("filler " * 80, "SECRET-KEY-999", "find key", mode="mock", max_context_tokens=2000, tokenizer=ApproximateCodec())
    assert [row["position_percent"] for row in report["positions"]] == [0, 25, 50, 75, 100]
    assert report["total_runs"] == 5
    assert report["successful_runs"] == 5
    assert "not empirical" in report["note"]


def test_empty_needle_rejected():
    try:
        run_position_simulation("some context", " ", "find", tokenizer=ApproximateCodec())
    except ValueError as exc:
        assert "needle" in str(exc)
    else:
        raise AssertionError("expected ValueError")
