"""Obtencao da classificacao "real" do livro em plataformas consolidadas.

Provedores (consultados na ordem configurada, com cache em disco):

``google_books``
    ``https://www.googleapis.com/books/v1/volumes`` -- devolve ``categories``
    no padrao BISAC ("Fiction / Horror / Ghost"). Publico; a chave de API so
    aumenta a cota de requisicoes.
``open_library``
    ``https://openlibrary.org/search.json`` -- devolve ``subject``, uma lista
    longa e ruidosa de assuntos, boa para complementar o Google Books.

Ambos retornam vocabulario proprio; ``taxonomy.map_external_terms`` faz a
traducao para os 14 generos internos. Nenhuma das duas APIs exige autenticacao,
o que mantem o experimento reproduzivel por quem for avaliar o trabalho.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from rapidfuzz import fuzz

from .taxonomy import GENRE_IDS, N_GENRES, map_external_terms

log = logging.getLogger(__name__)

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
USER_AGENT = "LibrarianAI/1.0 (projeto academico)"

# Confianca minima no casamento titulo/autor para aceitar o registro externo.
MATCH_THRESHOLD = 72.0


@dataclass
class Candidate:
    """Um registro de catalogo que casou com o livro procurado."""

    terms: list[str]
    score: float
    title: str | None = None
    author: str | None = None


@dataclass
class GroundTruth:
    """Classificacao externa de um livro."""

    found: bool
    provider: str | None = None
    matched_title: str | None = None
    matched_author: str | None = None
    match_score: float = 0.0
    raw_terms: list[str] = field(default_factory=list)
    distribution: np.ndarray | None = None
    top_genre: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "provider": self.provider,
            "matched_title": self.matched_title,
            "matched_author": self.matched_author,
            "match_score": round(self.match_score, 2),
            "raw_terms": self.raw_terms,
            "top_genre": self.top_genre,
            "distribution": (
                {g: float(p) for g, p in zip(GENRE_IDS, self.distribution)}
                if self.distribution is not None
                else None
            ),
            "note": self.note,
        }


def score_match(
    title: str, author: str | None, cand_title: str, cand_authors: list[str]
) -> float:
    """Similaridade combinada titulo/autor, em [0, 100].

    O titulo pesa 70% porque os nomes de autor variam muito entre catalogos
    ("Machado de Assis" vs "Joaquim Maria Machado de Assis").
    """
    t = fuzz.token_set_ratio(title.lower(), (cand_title or "").lower())
    if not author or not cand_authors:
        return float(t)
    a = max(fuzz.token_set_ratio(author.lower(), (c or "").lower()) for c in cand_authors)
    return float(0.7 * t + 0.3 * a)


class GroundTruthClient:
    """Cliente HTTP com cache para as APIs bibliograficas."""

    def __init__(
        self,
        providers: list[str] | None = None,
        cache_dir: Path | None = None,
        timeout: int = 15,
        google_books_api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.providers = providers or ["google_books", "open_library"]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.api_key = google_books_api_key
        self.max_retries = max(1, max_retries)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache -------------------------------------------------------------
    def _cache_path(self, title: str, author: str | None) -> Path | None:
        if not self.cache_dir:
            return None
        raw = f"{title}|{author}|{','.join(self.providers)}".lower()
        key = hashlib.sha256(raw.encode()).hexdigest()[:20]
        return self.cache_dir / f"gt_{key}.json"

    # -- api publica -------------------------------------------------------
    def lookup(self, title: str, author: str | None = None) -> GroundTruth:
        """Busca o livro nos provedores configurados, parando no primeiro acerto."""
        cache_file = self._cache_path(title, author)
        if cache_file and cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                return from_cache_payload(payload)
            except (json.JSONDecodeError, KeyError):
                log.warning("cache de ground truth corrompido: %s", cache_file.name)

        result = GroundTruth(found=False, note="nenhum provedor retornou resultado")
        for provider in self.providers:
            try:
                if provider == "google_books":
                    result = self._google_books(title, author)
                elif provider == "open_library":
                    result = self._open_library(title, author)
                else:
                    log.warning("provedor desconhecido ignorado: %s", provider)
                    continue
            except requests.RequestException as exc:
                log.warning("%s indisponivel (%s); tentando o proximo", provider, exc)
                result = GroundTruth(found=False, note=f"erro de rede em {provider}: {exc}")
                continue
            except ValueError as exc:  # JSON invalido
                log.warning("%s devolveu resposta ilegivel: %s", provider, exc)
                result = GroundTruth(found=False, note=f"resposta invalida de {provider}")
                continue
            if result.found:
                break
            time.sleep(0.3)  # cortesia com APIs publicas

        if cache_file and result.found:
            cache_file.write_text(
                json.dumps(result.as_dict(), ensure_ascii=False), encoding="utf-8"
            )
        return result

    # -- provedores --------------------------------------------------------
    def _google_books(self, title: str, author: str | None) -> GroundTruth:
        query = f'intitle:"{title}"'
        if author:
            query += f' inauthor:"{author}"'
        params: dict[str, Any] = {"q": query, "maxResults": 10, "printType": "books"}
        if self.api_key:
            params["key"] = self.api_key

        items = self._get_json(GOOGLE_BOOKS_URL, params).get("items") or []
        if not items and author:
            # Segunda tentativa sem o autor: catalogos divergem muito na grafia.
            params["q"] = f'intitle:"{title}"'
            items = self._get_json(GOOGLE_BOOKS_URL, params).get("items") or []

        candidates: list[Candidate] = []
        for item in items:
            info = item.get("volumeInfo", {})
            if not info.get("categories"):
                continue  # sem categoria o registro nao serve de ground truth
            score = score_match(title, author, info.get("title", ""), info.get("authors", []))
            candidates.append(
                Candidate(
                    terms=list(info.get("categories", [])),
                    score=score,
                    title=info.get("title"),
                    author=", ".join(info.get("authors", []) or []) or None,
                )
            )

        return build_ground_truth(candidates, "google_books")

    def _open_library(self, title: str, author: str | None) -> GroundTruth:
        params: dict[str, Any] = {
            "title": title,
            "limit": 10,
            "fields": "title,author_name,subject",
        }
        if author:
            params["author"] = author

        docs = self._get_json(OPEN_LIBRARY_URL, params).get("docs") or []

        candidates: list[Candidate] = []
        for doc in docs:
            if not doc.get("subject"):
                continue
            score = score_match(title, author, doc.get("title", ""), doc.get("author_name", []))
            # O Open Library devolve centenas de subjects, ordenados por
            # frequencia entre as edicoes; truncar limita o ruido de cauda longa.
            candidates.append(
                Candidate(
                    terms=list(doc.get("subject", []))[:40],
                    score=score,
                    title=doc.get("title"),
                    author=", ".join(doc.get("author_name", []) or []) or None,
                )
            )

        return build_ground_truth(candidates, "open_library")

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET com backoff exponencial em 429/5xx.

        O Google Books limita agressivamente por IP quando nao ha chave de API
        (429 ja na primeira consulta, em rede compartilhada). O backoff cobre a
        limitacao transitoria; a persistente cai para o proximo provedor.
        """
        delay = 1.0
        last_exc: requests.RequestException | None = None

        for attempt in range(self.max_retries):
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code in (429, 500, 502, 503, 504):
                last_exc = requests.HTTPError(
                    f"{response.status_code} de {url}", response=response
                )
                if attempt < self.max_retries - 1:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if (retry_after or "").isdigit() else delay
                    log.debug("HTTP %s; nova tentativa em %.1fs", response.status_code, wait)
                    time.sleep(min(wait, 10.0))
                    delay *= 2
                    continue
                raise last_exc
            response.raise_for_status()
            return response.json()

        raise last_exc or requests.RequestException("falha ao consultar o provedor")


def build_ground_truth(candidates: list[Candidate], provider: str) -> GroundTruth:
    """Consolida varios registros do catalogo em uma unica distribuicao.

    Confiar em um unico registro e fragil: catalogos publicos misturam edicoes,
    adaptacoes e traducoes sob o mesmo titulo. Medido no Open Library,
    ``Frankenstein`` casava primeiro com uma adaptacao teatral e saia
    classificado como "Drama". Agregar os assuntos de todas as edicoes que
    passam do limiar, ponderando pela qualidade do casamento, dilui esse ruido:
    generos citados por muitas edicoes dominam os que aparecem em uma so.
    """
    eligible = [c for c in candidates if c.score >= MATCH_THRESHOLD]
    if not eligible:
        best = max((c.score for c in candidates), default=0.0)
        return GroundTruth(
            found=False,
            provider=provider,
            note=(
                "sem correspondencia confiavel "
                f"(melhor score {best:.0f} < {MATCH_THRESHOLD:.0f})"
            ),
        )

    eligible.sort(key=lambda c: c.score, reverse=True)
    best = eligible[0]

    dist = np.zeros(N_GENRES, dtype=np.float64)
    used_terms: list[str] = []
    seen: set[str] = set()
    n_mapped = 0

    for cand in eligible:
        mapped = map_external_terms(cand.terms)
        if not mapped:
            continue
        n_mapped += 1
        # Peso pelo score normalizado: edicao que casa melhor influencia mais.
        weight = cand.score / 100.0
        for gid, value in mapped.items():
            dist[GENRE_IDS.index(gid)] += weight * value
        for term in cand.terms:
            if term not in seen:
                seen.add(term)
                used_terms.append(term)

    if dist.sum() <= 0:
        return GroundTruth(
            found=False,
            provider=provider,
            matched_title=best.title,
            matched_author=best.author,
            match_score=best.score,
            raw_terms=used_terms[:40],
            note="registros encontrados, mas nenhuma categoria mapeou para a taxonomia interna",
        )

    dist = dist / dist.sum()
    return GroundTruth(
        found=True,
        provider=provider,
        matched_title=best.title,
        matched_author=best.author,
        match_score=best.score,
        raw_terms=used_terms[:40],
        distribution=dist,
        top_genre=GENRE_IDS[int(np.argmax(dist))],
        note=f"consolidado a partir de {n_mapped} registro(s) do catalogo",
    )


def from_cache_payload(payload: dict[str, Any]) -> GroundTruth:
    dist = None
    if payload.get("distribution"):
        dist = np.array([payload["distribution"].get(g, 0.0) for g in GENRE_IDS])
    return GroundTruth(
        found=payload["found"],
        provider=payload.get("provider"),
        matched_title=payload.get("matched_title"),
        matched_author=payload.get("matched_author"),
        match_score=payload.get("match_score", 0.0),
        raw_terms=payload.get("raw_terms", []),
        distribution=dist,
        top_genre=payload.get("top_genre"),
        note=payload.get("note"),
    )
