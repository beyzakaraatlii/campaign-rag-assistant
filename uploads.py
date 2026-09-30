"""Safe, content-deduplicated uploads with no overwrite of existing files."""
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from config import DOCUMENTS_PATH, project_path
from database import INDEX_LOCK
from ingest import extract_pages, pdf_files, read_pdf_bytes


@dataclass(frozen=True)
class UploadResult:
    path: Path
    duplicate: bool


def safe_pdf_name(filename):
    if (not filename or any(c in filename for c in '/\\:')
            or filename in (".", "..") or not filename.lower().endswith(".pdf")):
        raise ValueError("Geçerli bir .pdf dosya adı kullanın; dosya adı yol içeremez.")
    stem = unicodedata.normalize("NFKC", filename[:-4])
    stem = re.sub(r"[^\w .-]", "_", stem, flags=re.UNICODE).strip(" .")[:100]
    if not stem:
        stem = "document"
    if stem.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        stem = "document_" + stem
    return stem + ".pdf"


def save_uploaded_pdf(filename, data, folder_path=DOCUMENTS_PATH):
    name = safe_pdf_name(filename)
    extract_pages(data)
    digest = hashlib.sha256(data).hexdigest()
    folder = project_path(folder_path)
    with INDEX_LOCK:
        folder.mkdir(parents=True, exist_ok=True)
        for existing in pdf_files(folder):
            if hashlib.sha256(read_pdf_bytes(existing)).hexdigest() == digest:
                return UploadResult(existing, True)
        target = folder / name
        suffix = 0
        while True:
            project_path(target)
            try:
                handle = target.open("xb")
                break
            except FileExistsError:
                suffix += 1
                target = folder / f"{Path(name).stem}_{digest[:12]}_{suffix}.pdf"
        try:
            with handle:
                handle.write(data)
                handle.flush()
        except Exception:
            target.unlink(missing_ok=True)  # Only the new file created by this call.
            raise
        return UploadResult(target, False)
