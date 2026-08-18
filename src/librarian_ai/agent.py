"""Agente de aprendizado por reforco que decide QUE paginas ler.

Formulacao
----------
O espaco de acoes "subconjunto de paginas" e combinatorio (C(P, n)), entao ele
e fatorado: o livro e dividido em ``K`` segmentos contiguos (inicio, meio, fim
...), que sao os bracos do bandit. Uma geracao amostra ``n`` paginas sorteando
repetidamente um segmento pela politica e, dentro dele, uma pagina uniforme e
ainda nao usada.

  - Politica:    pi = softmax(theta / tau),  theta em R^K
  - Acao:        n paginas ~ Categorical(pi) por segmento, sem reposicao
  - Recompensa:  qualidade da classificacao produzida por aquela amostra
  - Update:      REINFORCE com baseline (ou EXP3)

O gradiente do log-verossimilhanca de sortear ``count_k`` vezes o segmento k a
partir de ``n`` sorteios categoricos e ``count_k - n * pi_k``; por isso o passo
de REINFORCE abaixo usa exatamente esse termo, normalizado por ``n``.

Isso e o que da sentido ao "aprendizado": o agente descobre, por livro, que
certas regioes (tipicamente o miolo -- prefacios e apendices atrapalham)
rendem classificacoes mais consistentes, e passa a amostra-las mais.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SampleAction:
    """Uma amostragem de paginas concreta (uma geracao)."""

    pages: list[int]
    rate: float
    segment_counts: np.ndarray
    policy: np.ndarray

    @property
    def n_pages(self) -> int:
        return len(self.pages)


@dataclass
class PageSelectionAgent:
    """Bandit sobre segmentos do livro.

    Parameters
    ----------
    n_pages
        Total de paginas utilizaveis do livro.
    n_segments
        Numero de bracos. Reduzido automaticamente se o livro for curto.
    strategy
        ``"bandit"`` aprende a politica; ``"uniform"`` e o baseline sem
        aprendizado exigido para a comparacao no relatorio.
    """

    n_pages: int
    n_segments: int = 10
    strategy: str = "bandit"
    algorithm: str = "reinforce"
    learning_rate: float = 0.35
    temperature: float = 1.0
    baseline_decay: float = 0.8
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    theta: np.ndarray = field(init=False)
    baseline: float = field(init=False, default=0.0)
    _seen_reward: bool = field(init=False, default=False)
    history: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.n_pages < 1:
            raise ValueError("livro sem paginas utilizaveis")
        # Nao faz sentido ter mais segmentos que paginas.
        self.n_segments = max(1, min(self.n_segments, self.n_pages))
        self.theta = np.zeros(self.n_segments, dtype=np.float64)
        self._bounds = self._segment_bounds()

    # -- politica ----------------------------------------------------------
    def _segment_bounds(self) -> list[tuple[int, int]]:
        """Fronteiras [inicio, fim) de cada segmento, distribuindo o resto."""
        edges = np.linspace(0, self.n_pages, self.n_segments + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(self.n_segments)]

    def policy(self) -> np.ndarray:
        """Distribuicao atual sobre segmentos."""
        if self.strategy == "uniform":
            return np.full(self.n_segments, 1.0 / self.n_segments)
        z = self.theta / max(self.temperature, 1e-8)
        z = z - z.max()  # estabilidade numerica
        e = np.exp(z)
        return e / e.sum()

    # -- acao --------------------------------------------------------------
    def sample_rate(self, min_rate: float, max_rate: float, fixed_rate: float | None) -> float:
        if fixed_rate is not None:
            return float(fixed_rate)
        if max_rate <= min_rate:
            return float(min_rate)
        return float(self.rng.uniform(min_rate, max_rate))

    def select_pages(self, rate: float) -> SampleAction:
        """Sorteia as paginas de uma geracao segundo a politica atual."""
        n = int(round(rate * self.n_pages))
        n = max(1, min(n, self.n_pages))

        pi = self.policy()
        # Paginas ainda disponiveis por segmento, para amostrar sem reposicao.
        pools = [list(range(lo, hi)) for lo, hi in self._bounds]
        for pool in pools:
            self.rng.shuffle(pool)

        counts = np.zeros(self.n_segments, dtype=np.int64)
        chosen: list[int] = []
        weights = pi.copy()

        while len(chosen) < n:
            live = np.array([len(p) > 0 for p in pools])
            if not live.any():
                break
            w = weights * live
            total = w.sum()
            # Politica concentrada em segmentos ja esgotados -> cai para uniforme
            # entre os que ainda tem pagina.
            probs = w / total if total > 1e-12 else live / live.sum()
            k = int(self.rng.choice(self.n_segments, p=probs))
            chosen.append(pools[k].pop())
            counts[k] += 1

        chosen.sort()
        return SampleAction(pages=chosen, rate=rate, segment_counts=counts, policy=pi)

    # -- aprendizado -------------------------------------------------------
    def update(self, action: SampleAction, reward: float) -> dict:
        """Aplica um passo de atualizacao da politica e devolve o diagnostico."""
        advantage = 0.0
        if self.strategy == "bandit":
            if not self._seen_reward:
                # Primeira recompensa vira o baseline: sem isso o primeiro passo
                # empurra a politica so pelo sinal absoluto de r, que e arbitrario.
                self.baseline = reward
                self._seen_reward = True
            advantage = reward - self.baseline

            if self.algorithm == "reinforce":
                n = max(1, action.n_pages)
                grad = (action.segment_counts - n * action.policy) / n
                self.theta += self.learning_rate * advantage * grad
            elif self.algorithm == "exp3":
                # Estimador de importancia: credita so os bracos efetivamente usados.
                n = max(1, action.n_pages)
                share = action.segment_counts / n
                probs = np.maximum(action.policy, 1e-6)
                self.theta += self.learning_rate * advantage * share / probs
            else:  # pragma: no cover - barrado na validacao da config
                raise ValueError(f"algoritmo desconhecido: {self.algorithm}")

            self.theta -= self.theta.mean()  # softmax e invariante a shift; evita drift
            np.clip(self.theta, -20.0, 20.0, out=self.theta)
            self.baseline = (
                self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
            )

        record = {
            "reward": float(reward),
            "baseline": float(self.baseline),
            "advantage": float(advantage),
            "policy": action.policy.tolist(),
            "new_policy": self.policy().tolist(),
            "segment_counts": action.segment_counts.tolist(),
            "rate": action.rate,
            "n_pages_sampled": action.n_pages,
        }
        self.history.append(record)
        return record

    # -- diagnostico -------------------------------------------------------
    def policy_entropy(self) -> float:
        """Entropia normalizada da politica: 1.0 = uniforme, 0.0 = concentrada."""
        if self.n_segments == 1:
            return 0.0
        pi = np.maximum(self.policy(), 1e-12)
        return float(-(pi * np.log(pi)).sum() / np.log(self.n_segments))

    def segment_ranges(self) -> list[tuple[int, int]]:
        return list(self._bounds)
