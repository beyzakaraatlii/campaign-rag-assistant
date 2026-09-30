"""Validated PDF ingestion with page-level source metadata."""
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from config import (
    DOCUMENTS_PATH,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    MAX_TEXT_CHARS,
    project_path,
)


class PDFValidationError(ValueError):
    pass


def extract_pages(data):
    if not data:
        raise PDFValidationError("PDF dosyası boş.")

    if len(data) > MAX_PDF_BYTES:
        raise PDFValidationError("PDF en fazla 25 MB olabilir.")

    if not data.lstrip().startswith(b"%PDF-"):
        raise PDFValidationError("Dosya geçerli bir PDF değil.")

    try:
        reader = PdfReader(BytesIO(data), strict=True)

        if reader.is_encrypted:
            raise PDFValidationError(
                "Şifreli PDF desteklenmiyor; şifresiz bir kopya yükleyin."
            )

        if len(reader.pages) > MAX_PDF_PAGES:
            raise PDFValidationError(
                "PDF en fazla 500 sayfa olabilir."
            )

        pages = []
        total = 0

        for number, page in enumerate(reader.pages, 1):
            text = (
                page.extract_text() or ""
            ).replace("\x00", "").strip()

            total += len(text)

            if total > MAX_TEXT_CHARS:
                raise PDFValidationError(
                    "PDF metni çok uzun; dosyayı bölerek yükleyin."
                )

            if text:
                pages.append({
                    "page": number,
                    "text": text
                })

        if not pages:
            raise PDFValidationError(
                "PDF'de okunabilir metin yok. "
                "Taranmış PDF için önce OCR uygulayın."
            )

        return pages

    except PDFValidationError:
        raise

    except Exception as exc:
        raise PDFValidationError(
            "PDF okunamadı; dosya bozuk veya desteklenmeyen biçimde."
        ) from exc


def pdf_files(folder_path=DOCUMENTS_PATH):
    folder = project_path(folder_path)

    if not folder.exists():
        return []

    return sorted(
        (
            p for p in folder.iterdir()
            if p.suffix.lower() == ".pdf"
        ),
        key=lambda p: p.name.casefold()
    )


def read_pdf_bytes(path):
    path = Path(path)

    if path.is_symlink():
        raise PDFValidationError(
            "Sembolik bağlantı olan PDF dosyaları desteklenmiyor."
        )

    path = project_path(path)

    with path.open("rb") as handle:
        data = handle.read(MAX_PDF_BYTES + 1)

    if len(data) > MAX_PDF_BYTES:
        raise PDFValidationError(
            "PDF en fazla 25 MB olabilir."
        )

    return data


def read_pdf(file_path):
    return "\n".join(
        p["text"]
        for p in extract_pages(
            read_pdf_bytes(file_path)
        )
    )


def load_documents(folder_path=DOCUMENTS_PATH):
    documents = []

    for path in pdf_files(folder_path):
        try:
            pages = extract_pages(
                read_pdf_bytes(path)
            )

        except (ValueError, OSError) as exc:
            raise PDFValidationError(
                f"{path.name}: {exc}"
            ) from exc

        documents.extend(
            {
                "source": path.name,
                **page
            }
            for page in pages
        )

    return documents


def create_chunks(
    documents,
    chunk_size=1000,
    overlap=200
):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError(
            "chunk_size pozitif, overlap ise "
            "0 ile chunk_size arasında olmalı."
        )

    chunks = []

    for document in documents:
        text = document["text"].strip()

        for start in range(
            0,
            len(text),
            chunk_size - overlap
        ):
            part = text[
                start:start + chunk_size
            ].strip()

            if part:
                chunks.append({
                    "source": document["source"],
                    "page": document.get("page"),
                    "text": part
                })

            if start + chunk_size >= len(text):
                break

    return chunks


if __name__ == "__main__":
    documents = load_documents()

    print(
        f"Metin içeren sayfa: {len(documents)}, "
        f"parça: {len(create_chunks(documents))}"
    )
