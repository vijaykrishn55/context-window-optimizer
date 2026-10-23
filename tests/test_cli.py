from context_optimizer.cli import build_parser


def test_analyze_command_arguments():
    args = build_parser().parse_args(["analyze", "--input", "in.json", "--output", "out.json"])
    assert args.command == "analyze"
    assert args.budget == 12000


def test_simulator_accepts_mock_mode():
    args = build_parser().parse_args(["simulate", "--input", "doc.txt", "--needle", "N-1", "--query", "find", "--mode", "mock"])
    assert args.mode == "mock"
    assert args.model == "openai/gpt-oss-20b"
