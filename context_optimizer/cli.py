"""Command-line entry points for analysis, optimization, chunking, and simulation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from .service import analyze_request, optimize_request, read_request, simulate_document, split_document


def _json_out(path: str, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="context-window", description="Audit and optimize long LLM prompts.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("analyze", "Audit token allocation from a request JSON."),
                            ("optimize", "Build a sandwich prompt and prune context to budget.")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--input", required=True, help="Request JSON path.")
        cmd.add_argument("--budget", type=int, default=int(os.getenv("PROMPT_BUDGET_TOKENS", "12000")), help="Maximum input tokens.")
        cmd.add_argument("--output", required=True, help="JSON report path (analyze) or prompt path (optimize).")
        if name == "optimize":
            cmd.add_argument("--report", default="reports/optimization.json", help="Optimization report JSON path.")
    chunk = sub.add_parser("chunk", help="Split a text file with token overlap.")
    chunk.add_argument("--input", required=True)
    chunk.add_argument("--chunk-tokens", type=int, default=2000)
    chunk.add_argument("--overlap", type=float, default=0.2)
    chunk.add_argument("--output-dir", default="output/chunks")
    sim = sub.add_parser("simulate", help="Run needle retrieval at 0/25/50/75/100 percent depth.")
    sim.add_argument("--input", required=True, help="Base document text file.")
    sim.add_argument("--needle", required=True)
    sim.add_argument("--query", required=True)
    sim.add_argument("--mode", choices=("mock", "api"), default="mock")
    sim.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "groq"))
    sim.add_argument("--model", default=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"))
    sim.add_argument("--max-context-tokens", type=int, default=int(os.getenv("CONTEXT_WINDOW_TOKENS", "12000")))
    sim.add_argument("--output", default="reports/simulation.json")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    try:
        if args.command in {"analyze", "optimize"}:
            data = read_request(args.input)
            if args.command == "analyze":
                report = analyze_request(data, args.budget)
                _json_out(args.output, report)
                print(f"Audit: {report['status']} | {report['total_tokens']}/{report['max_tokens']} tokens | report: {args.output}")
            else:
                prompt, report = optimize_request(data, args.budget)
                target = Path(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(prompt, encoding="utf-8")
                _json_out(args.report, report)
                if report["dropped_chunks"]:
                    print(f"WARNING: Dropped {report['dropped_chunks']} low-priority context chunk(s) to fit the token budget.")
                print(f"Optimized: {report['output_tokens']}/{report['budget_tokens']} tokens | prompt: {args.output} | report: {args.report}")
        elif args.command == "chunk":
            text = Path(args.input).read_text(encoding="utf-8")
            chunks = split_document(text, args.chunk_tokens, args.overlap)
            target = Path(args.output_dir)
            target.mkdir(parents=True, exist_ok=True)
            for index, chunk in enumerate(chunks, start=1):
                (target / f"chunk_{index:04d}.txt").write_text(chunk, encoding="utf-8")
            _json_out(str(target / "manifest.json"), {"chunks": len(chunks), "overlap_ratio": args.overlap, "files": [f"chunk_{i:04d}.txt" for i in range(1, len(chunks) + 1)]})
            print(f"Created {len(chunks)} chunks in {target}")
        else:
            base = Path(args.input).read_text(encoding="utf-8")
            report = simulate_document(base, args.needle, args.query, mode=args.mode, provider=args.provider, model=args.model, max_context_tokens=args.max_context_tokens)
            _json_out(args.output, report)
            print(f"Simulation: {report['successful_runs']}/{report['total_runs']} retrievals | {args.mode} mode | report: {args.output}")
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
