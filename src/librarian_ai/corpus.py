"""Ingestao do dataset: PDF -> paginas de texto limpo, com cache em disco."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

log = logging.getLogger(__name__)

# "Dom Casmurro - Machado de Assis.pdf" / "Machado de Assis - Dom Casmurro.pdf"
_FILENAME_SPLIT = re.compile(r"\s+[-_–]\s+")
_WHITESPACE = re.compile(r"[ \t ]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


@dataclass
class Book:
    """Um livro do dataset, ja paginado e filtrado."""

    book_id: str
    title: str
    author: str | None
    source_path: Path
    pages: list[str] = field(default_factory=list)
    total_pages_raw: int = 0

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def __repr__(self) -> str:  # pragma: no cover - conveniencia de debug
        return f"<Book {self.title!r} by {self.author!r} pages={self.n_pages}>"


def clean_page_text(text: str) -> str:
    """Normalizacao leve: junta hifenizacao de fim de linha e colapsa espacos.

    Deliberadamente conservadora -- o classificador zero-shot trabalha melhor
    com prosa intacta, entao nao removemos stopwords nem aplicamos stemming
    aqui (isso vive em ``preprocess.py``, so para o backend TF-IDF).
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def parse_filename(path: Path) -> tuple[str, str | None]:
    """Extrai (titulo, autor) do nome do arquivo, quando segue 'A - B'.

    Heuristica: se um dos lados tem <= 4 palavras e comeca com maiuscula em
    todas elas, tratamos esse lado como o autor.
    """
    stem = _WHITESPACE.sub(" ", path.stem.replace("_", " ")).strip()
    parts = [p.strip() for p in _FILENAME_SPLIT.split(stem) if p.strip()]
    if len(parts) < 2:
        return stem, None

    left, right = parts[0], parts[1]

    def looks_like_author(s: str) -> bool:
        words = s.split()
        return 1 < len(words) <= 4 and all(w[:1].isupper() for w in words)

    if looks_like_author(right) and not looks_like_author(left):
        return left, right
    if looks_like_author(left) and not looks_like_author(right):
        return right, left
    return left, right


def _pdf_metadata(reader: PdfReader) -> tuple[str | None, str | None]:
    try:
        meta = reader.metadata or {}
    except Exception:  # pragma: no cover - PDFs corrompidos
        return None, None
    title = (meta.get("/Title") or "").strip() or None
    author = (meta.get("/Author") or "").strip() or None
    # Muito PDF traz lixo do gerador no /Title ("Microsoft Word - doc1").
    if title and (len(title) < 3 or title.lower().startswith("microsoft word")):
        title = None
    return title, author


def _cache_key(path: Path) -> str:
    """Hash de conteudo (primeiros 4MB) + tamanho + mtime."""
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(4 * 1024 * 1024))
    return h.hexdigest()[:20]


def load_pdf(
    path: Path,
    *,
    cache_dir: Path | None = None,
    min_chars_per_page: int = 120,
    max_pages: int | None = None,
) -> Book:
    """Le um PDF e devolve um :class:`Book`. Usa cache JSON quando disponivel."""
    path = Path(path)
    cache_file: Path | None = None

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{path.stem}.{_cache_key(path)}.json"
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                log.debug("cache hit: %s", path.name)
                return Book(
                    book_id=payload["book_id"],
                    title=payload["title"],
                    author=payload.get("author"),
                    source_path=path,
                    pages=payload["pages"],
                    total_pages_raw=payload.get("total_pages_raw", len(payload["pages"])),
                )
            except (json.JSONDecodeError, KeyError):
                log.warning("cache corrompido, reextraindo: %s", cache_file.name)

    reader = PdfReader(str(path))
    meta_title, meta_author = _pdf_metadata(reader)
    file_title, file_author = parse_filename(path)

    raw_pages = reader.pages
    if max_pages is not None:
        raw_pages = raw_pages[:max_pages]

    pages: list[str] = []
    for i, page in enumerate(raw_pages):
        try:
            text = clean_page_text(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - paginas quebradas
            log.warning("falha ao extrair pagina %d de %s: %s", i, path.name, exc)
            continue
        if len(text) >= min_chars_per_page:
            pages.append(text)

    book = Book(
        book_id=path.stem,
        title=file_title or meta_title or path.stem,
        author=file_author or meta_author,
        source_path=path,
        pages=pages,
        total_pages_raw=len(reader.pages),
    )

    if not pages:
        log.warning(
            "%s: nenhuma pagina com >= %d caracteres. PDF digitalizado sem OCR?",
            path.name, min_chars_per_page,
        )

    if cache_file is not None:
        cache_file.write_text(
            json.dumps(
                {
                    "book_id": book.book_id,
                    "title": book.title,
                    "author": book.author,
                    "pages": book.pages,
                    "total_pages_raw": book.total_pages_raw,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return book


def load_corpus(
    input_dir: Path,
    *,
    cache_dir: Path | None = None,
    min_chars_per_page: int = 120,
    max_pages: int | None = None,
) -> list[Book]:
    """Carrega todos os PDFs de um diretorio (ordem alfabetica, deterministica)."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"diretorio do dataset nao existe: {input_dir}")

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"nenhum PDF encontrado em {input_dir}")

    books: list[Book] = []
    for pdf in pdfs:
        try:
            book = load_pdf(
                pdf, cache_dir=cache_dir,
                min_chars_per_page=min_chars_per_page, max_pages=max_pages,
            )
        except Exception as exc:
            log.error("ignorando %s: %s", pdf.name, exc)
            continue
        if book.n_pages == 0:
            log.error("ignorando %s: nenhuma pagina de texto utilizavel", pdf.name)
            continue
        books.append(book)

    if not books:
        raise RuntimeError(
            f"nenhum livro utilizavel em {input_dir} -- os PDFs podem ser "
            "digitalizacoes sem camada de texto (exigiriam OCR)"
        )
    return books
