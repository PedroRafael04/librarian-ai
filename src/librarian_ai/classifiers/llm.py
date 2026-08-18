"""Backend alternativo: classificacao por LLM (Claude) via API.

Serve de referencia de qualidade ("teto") para comparar com o zero-shot local
no relatorio. Exige ANTHROPIC_API_KEY no ambiente e tem custo por token.
"""

from __future__ import annotations

import json
import logging
import os
import re

import numpy as np

from ..taxonomy import GENRE_IDS, GENRES, N_GENRES
from .base import ClassificationResult, GenreClassifier, chunk_pages

log = logging.getLogger(__name__)

_SYSTEM = """Voce e um analista literario. Recebe TRECHOS AMOSTRADOS de um livro
(nao o livro inteiro) e estima a distribuicao de genero da OBRA COMO UM TODO.

Responda SOMENTE com um objeto JSON mapeando id de genero -> probabilidade,
usando exclusivamente estes ids, com os valores somando 1.0:
{ids}

Sem texto fora do JSON."""


class LLMClassifier(GenreClassifier):
    name = "llm"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_pages_per_call: int = 12,
        temperature: float = 0.0,
        max_chars_per_call: int = 40_000,
    ) -> None:
        self.model = model
        self.max_pages_per_call = max_pages_per_call
        self.temperature = temperature
        self.max_chars_per_call = max_chars_per_call
        self._client = None

    def warmup(self) -> None:
        if self._client is not None:
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY nao definida -- exigida pelo backend 'llm'. "
                "Use classifier.backend=zeroshot para rodar 100% local."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pacote 'anthropic' nao instalado: pip install anthropic") from exc
        self._client = Anthropic()

    def classify(self, pages: list[str]) -> ClassificationResult:
        self.warmup()
        chunks = chunk_pages(pages, self.max_chars_per_call)
        if not chunks:
            return ClassificationResult(self.uniform(), 0, self.name, {"empty": True})

        system = _SYSTEM.format(ids=", ".join(GENRE_IDS))
        acc = np.zeros(N_GENRES, dtype=np.float64)
        used = 0

        for chunk in chunks:
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": f"TRECHOS:\n\n{chunk}"}],
                )
                text = "".join(b.text for b in response.content if b.type == "text")
                vec = self._parse(text)
            except Exception as exc:
                log.warning("chamada ao LLM falhou, chunk ignorado: %s", exc)
                continue
            if vec is not None:
                acc += vec
                used += 1

        if used == 0:
            log.error("nenhuma chamada ao LLM teve sucesso; devolvendo distribuicao uniforme")
            return ClassificationResult(self.uniform(), 0, self.name, {"failed": True})

        dist = acc / acc.sum()
        return ClassificationResult(dist, used, self.name, {"model": self.model})

    @staticmethod
    def _parse(text: str) -> np.ndarray | None:
        """Extrai o JSON da resposta, tolerando cercas de codigo e texto solto."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        vec = np.zeros(N_GENRES, dtype=np.float64)
        valid = {g.id for g in GENRES}
        for key, value in payload.items():
            key = str(key).strip()
            if key in valid and isinstance(value, (int, float)) and value >= 0:
                vec[GENRE_IDS.index(key)] = float(value)
        total = vec.sum()
        return vec / total if total > 0 else None
