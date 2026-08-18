"""Backend classico e leve: TF-IDF sobre lexico de genero (sem modelos pesados).

Duas modalidades:

* **lexical** (padrao, nao supervisionado) -- pontua cada genero pela massa
  TF-IDF dos termos do seu lexico presentes no texto amostrado. Nao precisa de
  dados rotulados e roda em milissegundos, servindo de *baseline* honesto para
  medir quanto o zero-shot realmente adiciona.
* **supervisionada** -- se ``model_path`` apontar para um pipeline sklearn
  serializado (joblib) com ``predict_proba`` e ``classes_`` na taxonomia, ele
  e usado no lugar.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..taxonomy import GENRE_IDS, N_GENRES
from .base import ClassificationResult, GenreClassifier

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-zA-ZáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]{3,}")

# Lexico bilingue (PT/EN) por genero. Sinais grosseiros, mas suficientes para
# um baseline -- a intencao aqui e ser transparente e reproduzivel, nao otimo.
LEXICON: dict[str, tuple[str, ...]] = {
    "drama": ("familia", "casamento", "sofrimento", "lagrimas", "destino", "honra",
              "family", "marriage", "grief", "tears", "suffering"),
    "tragedia": ("morte", "ruina", "fatal", "condenado", "vinganca", "maldicao",
                 "death", "ruin", "doom", "fate", "revenge", "curse"),
    "suspense": ("segredo", "perseguicao", "ameaca", "escuridao", "fuga", "perigo",
                 "secret", "chase", "threat", "danger", "escape", "shadow"),
    "misterio": ("crime", "assassinato", "detetive", "pista", "investigacao", "suspeito",
                 "murder", "detective", "clue", "inspector", "evidence", "suspect"),
    "terror": ("horror", "sangue", "fantasma", "cadaver", "monstro", "pavor", "tumulo",
               "blood", "ghost", "corpse", "monster", "terror", "grave", "vampire"),
    "acao": ("batalha", "espada", "navio", "cavalo", "combate", "viagem", "inimigo",
             "battle", "sword", "ship", "horse", "fight", "journey", "enemy"),
    "comedia": ("riso", "gargalhada", "piada", "ridiculo", "tolo", "engracado",
                "laughter", "joke", "absurd", "foolish", "merry", "jest"),
    "satira": ("ironia", "hipocrisia", "costumes", "sociedade", "critica", "vaidade",
               "irony", "hypocrisy", "society", "vanity", "mock", "folly"),
    "romance": ("amor", "beijo", "paixao", "coracao", "namorada", "ciume", "noiva",
                "love", "kiss", "passion", "heart", "beloved", "jealousy"),
    "ficcao_cientifica": ("maquina", "futuro", "planeta", "cientista", "experimento",
                          "machine", "future", "planet", "scientist", "space", "robot"),
    "fantasia": ("magia", "feitico", "reino", "dragao", "bruxa", "encantado",
                 "magic", "spell", "kingdom", "dragon", "witch", "enchanted"),
    "historico": ("rei", "imperio", "guerra", "revolucao", "seculo", "batalha",
                  "king", "empire", "war", "revolution", "century", "throne"),
    "filosofico": ("alma", "verdade", "existencia", "razao", "consciencia", "moral",
                   "soul", "truth", "existence", "reason", "conscience", "virtue"),
    "realismo": ("cidade", "rua", "trabalho", "dinheiro", "casa", "vizinho", "jantar",
                 "town", "street", "work", "money", "house", "neighbour", "dinner"),
}


class TfidfClassifier(GenreClassifier):
    name = "tfidf"

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._supervised = False

    def warmup(self) -> None:
        if self._model is not None or self.model_path is None:
            return
        if not self.model_path.exists():
            log.info("modelo supervisionado ausente (%s); usando modo lexical",
                     self.model_path)
            return
        try:
            import joblib
            self._model = joblib.load(self.model_path)
            self._supervised = True
            log.info("modelo TF-IDF supervisionado carregado de %s", self.model_path)
        except Exception as exc:
            log.warning("falha ao carregar %s (%s); usando modo lexical",
                        self.model_path, exc)

    def classify(self, pages: list[str]) -> ClassificationResult:
        self.warmup()
        text = "\n".join(pages).strip()
        if not text:
            return ClassificationResult(self.uniform(), 0, self.name, {"empty": True})

        if self._supervised:
            return self._classify_supervised(text)
        return self._classify_lexical(text)

    def _classify_supervised(self, text: str) -> ClassificationResult:
        proba = self._model.predict_proba([text])[0]
        classes = list(self._model.classes_)
        vec = np.zeros(N_GENRES, dtype=np.float64)
        for cls, p in zip(classes, proba):
            if cls in GENRE_IDS:
                vec[GENRE_IDS.index(cls)] = float(p)
        total = vec.sum()
        dist = vec / total if total > 0 else self.uniform()
        return ClassificationResult(dist, 1, self.name, {"mode": "supervised"})

    def _classify_lexical(self, text: str) -> ClassificationResult:
        tokens = [t.lower() for t in _TOKEN.findall(text)]
        if not tokens:
            return ClassificationResult(self.uniform(), 0, self.name, {"empty": True})

        counts = Counter(tokens)
        n_tokens = len(tokens)
        vec = np.zeros(N_GENRES, dtype=np.float64)

        for i, gid in enumerate(GENRE_IDS):
            score = 0.0
            for term in LEXICON.get(gid, ()):  # pesos sublineares: evita que
                tf = counts.get(term, 0)       # um termo repetido domine o genero
                if tf:
                    score += (1.0 + math.log(tf)) / math.log(2 + n_tokens)
            vec[i] = score

        if vec.sum() <= 0:
            return ClassificationResult(self.uniform(), 1, self.name,
                                        {"mode": "lexical", "no_hits": True})

        # Softmax temperado: converte scores brutos em distribuicao sem que um
        # unico genero absorva toda a massa.
        z = vec / max(vec.max(), 1e-9) * 3.0
        e = np.exp(z - z.max())
        return ClassificationResult(e / e.sum(), 1, self.name, {"mode": "lexical"})
