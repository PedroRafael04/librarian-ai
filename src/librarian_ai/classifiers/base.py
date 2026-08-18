"""Contrato comum aos backends de classificacao."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..taxonomy import GENRE_IDS, N_GENRES


@dataclass
class ClassificationResult:
    """Saida de uma passada de classificacao sobre um conjunto de paginas."""

    distribution: np.ndarray          # vetor de probabilidade sobre GENRE_IDS
    n_chunks: int = 0
    backend: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {gid: float(p) for gid, p in zip(GENRE_IDS, self.distribution)}


class GenreClassifier(ABC):
    """Classificador de genero a partir de um conjunto de paginas.

    Implementacoes recebem as paginas ja amostradas pelo agente e devolvem uma
    distribuicao sobre a taxonomia. Sao intencionalmente sem estado entre
    chamadas: cada geracao e independente, e a memoria fica no agente.
    """

    name: str = "base"

    @abstractmethod
    def classify(self, pages: list[str]) -> ClassificationResult:
        """Classifica um conjunto de paginas."""

    def warmup(self) -> None:
        """Carrega pesos/valida credenciais antes do loop principal (opcional)."""

    def close(self) -> None:
        """Libera recursos (ex.: VRAM). Chamado ao fim do pipeline."""

    @staticmethod
    def uniform() -> np.ndarray:
        return np.full(N_GENRES, 1.0 / N_GENRES)


def chunk_pages(pages: list[str], max_chars: int) -> list[str]:
    """Agrupa paginas em blocos de ate ``max_chars`` caracteres.

    Paginas de livro sao curtas demais para saturar a janela do modelo; juntar
    varias por chamada reduz muito o numero de forward passes. Uma pagina que
    sozinha estoure o limite e truncada -- o corte fica no fim, preservando o
    inicio, que costuma carregar mais sinal narrativo.
    """
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for page in pages:
        text = page.strip()
        if not text:
            continue
        if len(text) > max_chars:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            chunks.append(text[:max_chars])
            continue
        if size + len(text) > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(text)
        size += len(text) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks
