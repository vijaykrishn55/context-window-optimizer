from context_optimizer.analyzer import ContextWindowAnalyzer
from context_optimizer.tokenizer import ApproximateCodec


def test_allocation_report_within_budget():
    analyzer = ContextWindowAnalyzer(tokenizer=ApproximateCodec())
    report = analyzer.audit_allocation("SYS", ["alpha", "middle data", "omega"], "query", 100)
    assert report["total_tokens"] == len("SYS") + len("alpha") + len("middle data") + len("omega") + len("query")
    assert report["status"] == "within_budget"
    assert report["remaining_tokens"] >= 0
    assert report["tokenizer"] == "character-fallback-estimate"


def test_budget_overflow_detected():
    analyzer = ContextWindowAnalyzer(tokenizer=ApproximateCodec())
    report = analyzer.audit_allocation("SYS", ["x" * 200], "query", 30)
    assert report["status"] == "exceeds_budget"
    assert report["total_tokens"] > report["max_tokens"]


def test_invalid_budget_rejected():
    analyzer = ContextWindowAnalyzer(tokenizer=ApproximateCodec())
    try:
        analyzer.audit_allocation("", [], "", 0)
    except ValueError as exc:
        assert "greater than zero" in str(exc)
    else:
        raise AssertionError("expected ValueError")
