import pytest
from context_optimizer.optimizer import ContextOptimizer
from context_optimizer.tokenizer import ApproximateCodec


def make_optimizer(budget=1000):
    return ContextOptimizer(max_tokens=budget, tokenizer=ApproximateCodec())


def test_sandwich_repeats_critical_and_ends_with_query():
    opt = make_optimizer()
    prompt = opt.construct_sandwich_prompt("CRITICAL", "Data payload", "Question?", "SECRET")
    assert prompt.startswith("CRITICAL\n")
    assert prompt.count("SECRET") == 2
    assert prompt.endswith("Question?")


def test_reference_sandwich_repeats_system_when_no_fact_supplied():
    prompt = make_optimizer().construct_sandwich_prompt("CRITICAL", "Data", "Question")
    assert prompt.count("CRITICAL") >= 2


def test_pruning_preserves_edges_when_possible():
    opt = make_optimizer()
    chunks = ["START_PRIORITY", "x" * 40, "y" * 40, "END_PRIORITY"]
    pruned = opt.prune_context(chunks, target_max_tokens=35)
    assert "START_PRIORITY" in pruned
    assert "END_PRIORITY" in pruned
    assert len(pruned) < len(chunks)


def test_empty_prompt_parts_are_allowed():
    opt = make_optimizer()
    assert isinstance(opt.construct_sandwich_prompt("", "", ""), str)
    assert opt.count_tokens("") == 0


def test_optimize_payload_fits_budget_and_tracks_removed_chunks():
    opt = make_optimizer(280)
    prompt, kept, dropped = opt.optimize_payload("SYS", "RULE-1", ["low " * 10, "important rule details " * 4, "misc " * 10], "What is RULE-1?", 280)
    assert opt.count_tokens(prompt) <= 280
    assert len(kept) + len(dropped) == 3
    assert dropped
    assert prompt.endswith("What is RULE-1?")


def test_mandatory_prompt_cannot_fit_raises():
    opt = make_optimizer(5)
    with pytest.raises(ValueError, match="alone exceed"):
        opt.optimize_payload("system " * 20, "fact", [], "query", 5)
