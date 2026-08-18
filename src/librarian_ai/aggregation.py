"""Interpolacao das classificacoes ao longo das geracoes e metricas de suporte."""

from __future__ import annotations

import numpy as np

from .taxonomy import GENRE_IDS, N_GENRES


def normalize(vec: np.ndarray) -> np.ndarray:
    """Projeta um vetor de scores no simplexo (soma 1, sem negativos)."""
    v = np.clip(np.asarray(vec, dtype=np.float64), 0.0, None)
    total = v.sum()
    if total <= 0:
        return np.full(v.shape, 1.0 / v.size)
    return v / total


def entropy(dist: np.ndarray, normalized: bool = True) -> float:
    """Entropia de Shannon; ``normalized`` divide por log(N) -> [0, 1]."""
    p = np.maximum(normalize(dist), 1e-12)
    h = float(-(p * np.log(p)).sum())
    return h / np.log(len(p)) if normalized and len(p) > 1 else h


def confidence(dist: np.ndarray) -> float:
    """1 - entropia normalizada. Distribuicao concentrada -> confianca alta."""
    return 1.0 - entropy(dist, normalized=True)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Divergencia de Jensen-Shannon em base 2, em [0, 1]."""
    p, q = normalize(p), normalize(q)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


def aggregate(
    distributions: list[np.ndarray],
    *,
    method: str = "weighted_mean",
    weight_by_confidence: bool = True,
    ema_alpha: float | None = None,
) -> np.ndarray:
    """Interpola as classificacoes de todas as geracoes em uma unica.

    ``weighted_mean``
        Media aritmetica ponderada pela confianca de cada geracao -- geracoes
        que produziram um veredito difuso pesam menos.
    ``mean``
        Media simples (baseline).
    ``log_pool``
        Pooling logaritmico (media geometrica normalizada). E mais rigoroso:
        um genero so sobrevive se nenhuma geracao o descartou. Bom para
        contrastar com a media no relatorio.

    ``ema_alpha`` aplica media movel exponencial na ordem das geracoes, dando
    mais peso as ultimas -- as que ja se beneficiaram da politica aprendida.
    """
    if not distributions:
        return np.full(N_GENRES, 1.0 / N_GENRES)

    mat = np.vstack([normalize(d) for d in distributions])

    if ema_alpha is not None:
        acc = mat[0].copy()
        for row in mat[1:]:
            acc = ema_alpha * row + (1 - ema_alpha) * acc
        return normalize(acc)

    if method == "log_pool":
        weights = _weights(mat, weight_by_confidence)
        logs = np.log(np.maximum(mat, 1e-12))
        return normalize(np.exp(weights @ logs))

    weights = _weights(mat, weight_by_confidence and method == "weighted_mean")
    return normalize(weights @ mat)


def _weights(mat: np.ndarray, by_confidence: bool) -> np.ndarray:
    if not by_confidence:
        return np.full(len(mat), 1.0 / len(mat))
    w = np.array([confidence(row) for row in mat])
    return normalize(w) if w.sum() > 1e-12 else np.full(len(mat), 1.0 / len(mat))


def top_k(dist: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Os ``k`` generos mais provaveis, como pares (id, probabilidade)."""
    p = normalize(dist)
    order = np.argsort(p)[::-1][:k]
    return [(GENRE_IDS[i], float(p[i])) for i in order]


def convergence_curve(distributions: list[np.ndarray], **agg_kwargs) -> list[float]:
    """JSD entre a agregacao ate g e a agregacao ate g-1, para cada geracao.

    Valores caindo para perto de zero indicam que geracoes adicionais deixaram
    de mudar o veredito -- e o criterio de parada antecipada.
    """
    curve: list[float] = []
    prev: np.ndarray | None = None
    for g in range(1, len(distributions) + 1):
        current = aggregate(distributions[:g], **agg_kwargs)
        curve.append(jensen_shannon(prev, current) if prev is not None else 1.0)
        prev = current
    return curve
