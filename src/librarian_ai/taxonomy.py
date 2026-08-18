"""Taxonomia de generos literarios usada em todo o pipeline.

Cada genero tem:
  - ``id``: chave interna, estavel, usada nos vetores de probabilidade;
  - ``pt``: rotulo exibido em relatorios;
  - ``hypothesis``: frase em ingles entregue ao modelo zero-shot (os modelos
    NLI publicos sao treinados em ingles, entao a hipotese fica em ingles
    mesmo quando o texto do livro esta em portugues);
  - ``aliases``: termos que aparecem nas categorias/subjects das APIs externas
    e que devem ser mapeados para este genero.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Genre:
    id: str
    pt: str
    hypothesis: str
    aliases: tuple[str, ...] = field(default=())


GENRES: tuple[Genre, ...] = (
    Genre(
        "drama", "Drama",
        "a dramatic literary work about personal conflict and human relationships",
        ("drama", "dramatico", "domestic fiction", "family life", "social"),
    ),
    Genre(
        "tragedia", "Tragedia",
        "a tragedy, where the protagonist is destroyed by fate or by a fatal flaw",
        ("tragedy", "tragedia", "tragic"),
    ),
    Genre(
        "suspense", "Suspense/Thriller",
        "a suspense thriller full of tension, danger and unexpected turns",
        ("thriller", "suspense", "psychological thriller"),
    ),
    Genre(
        "misterio", "Misterio/Policial",
        "a mystery or detective story about investigating a crime",
        ("mystery", "detective", "crime", "policial", "noir", "whodunit"),
    ),
    Genre(
        "terror", "Terror/Horror",
        "a horror story meant to frighten the reader with supernatural or macabre events",
        ("horror", "terror", "gothic", "ghost", "supernatural", "macabre"),
    ),
    Genre(
        "acao", "Acao/Aventura",
        "an action adventure story about journeys, battles and physical danger",
        ("adventure", "action", "aventura", "sea stories", "war", "western"),
    ),
    Genre(
        "comedia", "Comedia/Humor",
        "a comedy, humorous and light-hearted, meant to amuse the reader",
        ("humor", "comedy", "comic", "comedia", "funny"),
    ),
    Genre(
        "satira", "Satira",
        "a satire that ridicules society, politics or human vice through irony",
        ("satire", "satira", "parody", "irony"),
    ),
    Genre(
        "romance", "Romance (amoroso)",
        "a romance about love, courtship and the relationship between lovers",
        ("romance", "love stories", "courtship", "romantic"),
    ),
    Genre(
        "ficcao_cientifica", "Ficcao Cientifica",
        "a science fiction story about technology, space travel or the future",
        ("science fiction", "sci-fi", "ficcao cientifica", "dystopia", "utopia", "futuristic"),
    ),
    Genre(
        "fantasia", "Fantasia",
        "a fantasy story set in an imaginary world with magic and mythical creatures",
        ("fantasy", "fantasia", "magic", "mythology", "fairy tales", "legends"),
    ),
    Genre(
        "historico", "Historico",
        "a historical novel set in a documented period of the past",
        ("historical", "historico", "history", "historical fiction"),
    ),
    Genre(
        "filosofico", "Filosofico/Existencial",
        "a philosophical novel exploring existence, morality and the meaning of life",
        ("philosophy", "philosophical", "filosofia", "existential", "religion", "moral"),
    ),
    Genre(
        "realismo", "Realismo/Costumes",
        "a realistic novel depicting everyday life and the manners of a society",
        ("realism", "realistic", "manners", "literary", "classics", "bildungsroman", "coming of age"),
    ),
)

GENRE_IDS: tuple[str, ...] = tuple(g.id for g in GENRES)
BY_ID: dict[str, Genre] = {g.id: g for g in GENRES}
N_GENRES: int = len(GENRES)
INDEX: dict[str, int] = {g.id: i for i, g in enumerate(GENRES)}


def hypotheses() -> list[str]:
    """Rotulos candidatos na ordem canonica, para o classificador zero-shot."""
    return [g.hypothesis for g in GENRES]


def label_pt(genre_id: str) -> str:
    return BY_ID[genre_id].pt if genre_id in BY_ID else genre_id


def _normalize(text: str) -> str:
    """Minusculas, sem acento, sem pontuacao -- para casar aliases das APIs."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def map_external_terms(terms: list[str]) -> dict[str, float]:
    """Converte categorias externas (BISAC do Google Books, subjects do Open
    Library) em um vetor de scores sobre a taxonomia interna.

    Cada termo pode citar varios generos ("Fiction / Horror / Ghost"), entao a
    contagem e feita por alias encontrado e normalizada no fim.
    """
    scores: dict[str, float] = {}
    for term in terms:
        norm = _normalize(term)
        if not norm:
            continue
        for genre in GENRES:
            for alias in genre.aliases:
                if re.search(rf"\b{re.escape(_normalize(alias))}\b", norm):
                    scores[genre.id] = scores.get(genre.id, 0.0) + 1.0
                    break
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    return scores
