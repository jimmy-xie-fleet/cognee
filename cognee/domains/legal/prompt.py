from pathlib import Path


def load_legal_extraction_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "legal_extraction_system.txt").read_text(
        encoding="utf-8"
    )
