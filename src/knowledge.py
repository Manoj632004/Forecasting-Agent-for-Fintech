"""Required forecasting guidance, independent of optional dataset retrieval."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "knowledge"


def load_guidance(agent: str) -> str:
    if agent not in {"discovery", "preprocess", "forecast", "visualization", "orchestrator"}:
        raise ValueError(f"Unknown agent: {agent}")
    sections = []
    for name in ("shared", agent):
        path = ROOT / f"{name}.md"
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Required guidance is empty: {path}")
        sections.append(text)
    return "\n\n".join(sections)
