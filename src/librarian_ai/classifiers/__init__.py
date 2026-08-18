"""Backends de classificacao de genero."""

from __future__ import annotations

from ..config import Config
from .base import ClassificationResult, GenreClassifier, chunk_pages

__all__ = ["ClassificationResult", "GenreClassifier", "chunk_pages", "build_classifier"]


def build_classifier(cfg: Config) -> GenreClassifier:
    """Fabrica o backend indicado em ``classifier.backend``.

    Os imports sao locais de proposito: quem roda so o backend TF-IDF nao
    precisa ter transformers/torch instalados, e vice-versa.
    """
    backend = cfg.require("classifier.backend")

    if backend == "zeroshot":
        from .zeroshot import ZeroShotClassifier
        return ZeroShotClassifier(
            model_name=cfg.get("classifier.zeroshot.model", "facebook/bart-large-mnli"),
            device=cfg.get("classifier.zeroshot.device", "auto"),
            fp16=bool(cfg.get("classifier.zeroshot.fp16", True)),
            batch_size=int(cfg.get("classifier.zeroshot.batch_size", 8)),
            max_chars_per_chunk=int(cfg.get("classifier.zeroshot.max_chars_per_chunk", 3500)),
            multi_label=bool(cfg.get("classifier.zeroshot.multi_label", True)),
        )

    if backend == "llm":
        from .llm import LLMClassifier
        return LLMClassifier(
            model=cfg.get("classifier.llm.model", "claude-sonnet-5"),
            max_pages_per_call=int(cfg.get("classifier.llm.max_pages_per_call", 12)),
            temperature=float(cfg.get("classifier.llm.temperature", 0.0)),
        )

    if backend == "tfidf":
        from .tfidf import TfidfClassifier
        return TfidfClassifier(model_path=cfg.get("classifier.tfidf.model_path"))

    raise ValueError(f"backend desconhecido: {backend!r}")
