"""Project-local paths and shared RAG settings."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents"
DATABASE_PATH = PROJECT_ROOT / "data" / "campaign_rag.db"
CACHE_PATH = PROJECT_ROOT / ".cache"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
FOUNDRY_MODEL = "phi-4-mini"
TOP_K = 3
MIN_SIMILARITY = 0.25  # Heuristic cutoff, not a confidence percentage.
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_TEXT_CHARS = 2_000_000


def project_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("Dosya yolu proje klasörünün dışında olamaz.")
    return resolved


def configure_runtime():
    """Keep model downloads and application temp files inside the project."""
    for name in ("huggingface", "torch", "tmp", "foundry"):
        project_path(CACHE_PATH / name).mkdir(parents=True, exist_ok=True)
    for key, value in {
        "HF_HOME": CACHE_PATH / "huggingface",
        "HF_HUB_CACHE": CACHE_PATH / "huggingface" / "hub",
        "SENTENCE_TRANSFORMERS_HOME": CACHE_PATH / "huggingface" / "sentence_transformers",
        "TORCH_HOME": CACHE_PATH / "torch",
        "TEMP": CACHE_PATH / "tmp",
        "TMP": CACHE_PATH / "tmp",
    }.items():
        os.environ[key] = str(value)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


configure_runtime()
