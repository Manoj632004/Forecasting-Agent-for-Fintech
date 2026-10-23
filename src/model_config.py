"""Shared NVIDIA chat-model configuration."""
import os
from pathlib import Path

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def _load_env() -> dict:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
        os.environ.setdefault(key, value)
    return values


_ENV = _load_env()
MODEL = (_ENV.get("MODEL") or os.getenv("MODEL", "")).strip()
NVIDIA_API = (_ENV.get("NVIDIA_API") or os.getenv("NVIDIA_API", "")).strip()

if not MODEL:
    raise RuntimeError("MODEL is missing from .env")
if not NVIDIA_API:
    raise RuntimeError("NVIDIA_API is missing from .env")

MODEL_ID = MODEL if "/" in MODEL else f"nvidia/{MODEL}"
REQUEST_TIMEOUT_SECONDS = 300
os.environ["NVIDIA_API_KEY"] = NVIDIA_API


def build_chat_model(temperature: float = 0) -> ChatNVIDIA:
    return ChatNVIDIA(
        model=MODEL_ID,
        temperature=temperature,
        max_tokens=8192,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
