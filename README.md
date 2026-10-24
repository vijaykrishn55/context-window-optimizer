
# Context Window Optimizer

A modular Python utility for auditing token usage, enforcing context budgets, optimizing prompt structure, sliding-window chunking, and evaluating lost-in-the-middle retrieval.

## Features

- **Token Analysis:** Counts tokens by prompt component and detects budget overflow.
- **Sandwich Prompting:** Positions critical information at the beginning and near the final query.
- **Context Pruning:** Removes lower-priority context while protecting essential instructions.
- **Sliding-Window Chunking:** Splits documents with configurable overlap of at least 15%.
- **Retrieval Evaluation:** Tests needle retrieval at 0%, 25%, 50%, 75%, and 100%.
- **Provider Support:** Groq, OpenAI, Anthropic, Ollama, and vLLM integrations.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure your provider credentials in `.env`. Never commit API keys.

## CLI Usage

```powershell
# Analyze token allocation
python -m context_optimizer.cli analyze --input examples/sample_request.json --budget 12000 --output reports/audit.json

# Optimize prompt structure
python -m context_optimizer.cli optimize --input examples/sample_request.json --budget 12000 --output reports/optimized_prompt.txt --report reports/optimization.json

# Split a document into overlapping chunks
python -m context_optimizer.cli chunk --input examples/long_document.txt --chunk-tokens 2000 --overlap 0.2 --output-dir output/chunks

# Run offline simulation
python -m context_optimizer.cli simulate --input examples/long_document.txt --needle "SECRET-KEY-999" --query "Return the identifier" --mode mock --output reports/simulation.json
```

## Local Model Evaluation

A separate script supports local Hugging Face model evaluation:

```bash
python scripts/run_local_simulation.py \
  --input examples/os_note.txt \
  --needle THRASHING-TEST-001 \
  --query "Return the exact identifier" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-context-tokens 12000 \
  --output reports/simulation-full-notes.json
```

This script was tested on Kaggle using Qwen2.5-1.5B-Instruct against 11,713-token Operating Systems notes. It retrieved the inserted identifier at all five tested positions (5/5). This is a successful retrieval result, not evidence that lost-in-the-middle failures were eliminated.

## Verification

```powershell
python -m pytest -q
```

Automated tests run offline. The local evaluation script requires `torch` and `transformers`. Token counts and context limits depend on the selected tokenizer and model.

## Limitations

Sandwich prompting and context pruning are heuristics, not guarantees of accuracy. Mock simulation validates test plumbing only; real-model evaluation measures retrieval for the tested model, document, and prompt.
