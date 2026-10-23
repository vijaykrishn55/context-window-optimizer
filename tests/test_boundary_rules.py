from context_optimizer.optimizer import ContextOptimizer
from context_optimizer.tokenizer import ApproximateCodec


def test_long_prompt_places_priority_at_start_and_query_at_end():
    codec = ApproximateCodec()
    optimizer = ContextOptimizer(max_tokens=20000, tokenizer=codec)
    system = "SYSTEM-RULE-KEEP"
    fact = "CRITICAL-FACT-802B"
    query = "FINAL-QUERY-802B?"
    prompt = optimizer.construct_sandwich_prompt(system, "ordinary supporting context " * 500, query, fact)
    tokens = codec.encode(prompt)
    assert len(tokens) > 10000
    assert system in codec.decode(tokens[:500])
    assert fact in codec.decode(tokens[:500])
    assert query in codec.decode(tokens[-200:])
    assert prompt.endswith(query)
