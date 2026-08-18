"""Orquestracao do experimento: geracoes, recompensa, interpolacao, avaliacao.

Fluxo por livro::

    para g em 1..N:
        taxa       <- U(min_rate, max_rate)          # 15% a 30% das paginas
        paginas    <- agente.select_pages(taxa)      # politica do bandit
        dist_g     <- classificador.classify(paginas)
        recompensa <- qualidade(dist_g)
        agente.update(...)                           # aprendizado por reforco
        agregado   <- interpola(dist_1..dist_g)
        se convergiu: para

    ground_truth <- API externa (Google Books / Open Library)
    metricas     <- compara(agregado, ground_truth)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agent import PageSelectionAgent
from .aggregation import (
    aggregate,
    confidence,
    convergence_curve,
    entropy,
    jensen_shannon,
    top_k,
)
from .classifiers.base import GenreClassifier
from .config import Config
from .corpus import Book
from .ground_truth import GroundTruth, GroundTruthClient
from .taxonomy import GENRE_IDS, label_pt

log = logging.getLogger(__name__)


@dataclass
class GenerationRecord:
    """Resultado de uma unica geracao."""

    index: int
    rate: float
    n_pages_sampled: int
    pages: list[int]
    distribution: np.ndarray
    top_genre: str
    confidence: float
    reward: float
    reward_mode: str
    policy_before: list[float]
    policy_after: list[float]
    segment_counts: list[int]
    running_top_genre: str
    running_jsd: float
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.index,
            "rate": round(self.rate, 4),
            "n_pages_sampled": self.n_pages_sampled,
            "pages": self.pages,
            "distribution": {g: float(p) for g, p in zip(GENRE_IDS, self.distribution)},
            "top_genre": self.top_genre,
            "confidence": round(self.confidence, 4),
            "reward": round(self.reward, 4),
            "reward_mode": self.reward_mode,
            "policy_before": [round(p, 4) for p in self.policy_before],
            "policy_after": [round(p, 4) for p in self.policy_after],
            "segment_counts": self.segment_counts,
            "running_top_genre": self.running_top_genre,
            "running_jsd": round(self.running_jsd, 5),
            "elapsed_s": round(self.elapsed_s, 2),
        }


@dataclass
class BookResult:
    """Resultado consolidado de um livro."""

    book_id: str
    title: str
    author: str | None
    n_pages_total: int
    n_segments: int
    generations: list[GenerationRecord] = field(default_factory=list)
    final_distribution: np.ndarray | None = None
    predicted_genre: str | None = None
    ground_truth: GroundTruth | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    stopped_early: bool = False
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "book_id": self.book_id,
            "title": self.title,
            "author": self.author,
            "n_pages_total": self.n_pages_total,
            "n_segments": self.n_segments,
            "n_generations_run": len(self.generations),
            "stopped_early": self.stopped_early,
            "elapsed_s": round(self.elapsed_s, 2),
            "predicted_genre": self.predicted_genre,
            "predicted_genre_pt": label_pt(self.predicted_genre) if self.predicted_genre else None,
            "final_distribution": (
                {g: float(p) for g, p in zip(GENRE_IDS, self.final_distribution)}
                if self.final_distribution is not None
                else None
            ),
            "top3": (
                [
                    {"genre": g, "genre_pt": label_pt(g), "p": round(p, 4)}
                    for g, p in top_k(self.final_distribution, 3)
                ]
                if self.final_distribution is not None
                else []
            ),
            "ground_truth": self.ground_truth.as_dict() if self.ground_truth else None,
            "metrics": self.metrics,
            "generations": [g.as_dict() for g in self.generations],
        }


def compute_reward(
    dist: np.ndarray,
    *,
    mode: str,
    consensus: np.ndarray | None,
    ground_truth: np.ndarray | None,
    confidence_weight: float,
) -> tuple[float, str]:
    """Recompensa da geracao atual, em [0, 1]. Devolve (valor, modo_efetivo).

    ``supervised``
        Concordancia com a classificacao externa: ``1 - JSD``. So disponivel
        quando o ground truth foi buscado ANTES do loop, o que ``run_book`` faz
        apenas com ``--reward-mode supervised``.
    ``consensus``
        Concordancia com a interpolacao das geracoes anteriores. Nao usa rotulo
        externo: recompensa amostras *representativas* do livro, penalizando as
        regioes que produzem vereditos destoantes (prefacios, notas, apendices).

    Em ambos os casos soma-se um termo de confianca (baixa entropia), com peso
    ``confidence_weight``: entre duas amostras igualmente concordantes, a que
    produz um veredito mais decidido e preferida.
    """
    conf = confidence(dist)

    if mode == "supervised" and ground_truth is not None:
        base = 1.0 - jensen_shannon(dist, ground_truth)
        effective = "supervised"
    elif consensus is not None:
        base = 1.0 - jensen_shannon(dist, consensus)
        effective = "consensus"
    else:
        # Primeira geracao: nao ha com o que comparar, so a confianca informa.
        base = conf
        effective = "bootstrap"

    w = max(0.0, min(1.0, confidence_weight))
    return float((1.0 - w) * base + w * conf), effective


def evaluate(final: np.ndarray, gt: GroundTruth | None) -> dict[str, Any]:
    """Metricas de comparacao entre o veredito do modelo e o ground truth."""
    metrics: dict[str, Any] = {
        "final_confidence": round(confidence(final), 4),
        "final_entropy": round(entropy(final), 4),
    }
    if gt is None or not gt.found or gt.distribution is None:
        metrics["ground_truth_available"] = False
        return metrics

    predicted = GENRE_IDS[int(np.argmax(final))]
    ranked = [g for g, _ in top_k(final, len(GENRE_IDS))]
    gt_top = gt.top_genre
    gt_support = {g for g, p in zip(GENRE_IDS, gt.distribution) if p > 0.0}

    metrics.update(
        {
            "ground_truth_available": True,
            "ground_truth_genre": gt_top,
            "ground_truth_genre_pt": label_pt(gt_top) if gt_top else None,
            "top1_correct": predicted == gt_top,
            "top3_correct": gt_top in ranked[:3],
            # Acerto "solto": basta o genero previsto constar entre os que o
            # catalogo atribui ao livro. Classicos sao legitimamente
            # multi-genero, entao o top-1 estrito subestima o desempenho.
            "predicted_in_gt_support": predicted in gt_support,
            "jsd_to_ground_truth": round(jensen_shannon(final, gt.distribution), 4),
            "agreement": round(1.0 - jensen_shannon(final, gt.distribution), 4),
            "rank_of_ground_truth": (ranked.index(gt_top) + 1) if gt_top in ranked else None,
            "p_assigned_to_ground_truth": round(
                float(final[GENRE_IDS.index(gt_top)]) if gt_top in GENRE_IDS else 0.0, 4
            ),
        }
    )
    return metrics


def run_book(
    book: Book,
    classifier: GenreClassifier,
    cfg: Config,
    *,
    gt_client: GroundTruthClient | None = None,
    seed: int | None = None,
) -> BookResult:
    """Executa o ciclo completo de geracoes para um livro."""
    started = time.perf_counter()

    agg_kwargs = {
        "method": cfg.get("aggregation.method", "weighted_mean"),
        "weight_by_confidence": bool(cfg.get("aggregation.weight_by_confidence", True)),
        "ema_alpha": cfg.get("aggregation.ema_alpha"),
    }

    rng = np.random.default_rng(seed if seed is not None else cfg.get("generations.seed", 42))
    agent = PageSelectionAgent(
        n_pages=book.n_pages,
        n_segments=int(cfg.get("sampling.segments", 10)),
        strategy=cfg.get("agent.strategy", "bandit"),
        algorithm=cfg.get("agent.algorithm", "reinforce"),
        learning_rate=float(cfg.get("agent.learning_rate", 0.35)),
        temperature=float(cfg.get("agent.temperature", 1.0)),
        baseline_decay=float(cfg.get("agent.baseline_decay", 0.8)),
        normalize_advantage=bool(cfg.get("agent.normalize_advantage", True)),
        rng=rng,
    )

    result = BookResult(
        book_id=book.book_id,
        title=book.title,
        author=book.author,
        n_pages_total=book.n_pages,
        n_segments=agent.n_segments,
    )

    # Recompensa supervisionada exige o ground truth ANTES do loop. E uma opcao
    # de estudo (mede o teto do agente), nao o modo padrao: usar o rotulo
    # externo durante o treino e depois avaliar contra ele mede o proprio
    # gabarito. O padrao 'auto' cai em 'consensus', que nao olha o rotulo.
    reward_mode = cfg.get("agent.reward.mode", "auto")
    gt: GroundTruth | None = None
    if reward_mode == "supervised" and gt_client is not None:
        gt = gt_client.lookup(book.title, book.author)
        if not gt.found:
            log.warning(
                "%s: ground truth indisponivel; recompensa cai para 'consensus'", book.title
            )

    n_gen = int(cfg.require("generations.n"))
    min_rate = float(cfg.get("sampling.min_rate", 0.15))
    max_rate = float(cfg.get("sampling.max_rate", 0.30))
    fixed_rate = cfg.get("sampling.fixed_rate")
    conf_weight = float(cfg.get("agent.reward.confidence_weight", 0.3))

    es = cfg.get("generations.early_stop", {}) or {}
    es_enabled = bool(es.get("enabled", False))
    es_min = int(es.get("min_generations", 5))
    es_thresh = float(es.get("jsd_threshold", 0.01))
    es_patience = int(es.get("patience", 3))
    stable_streak = 0

    distributions: list[np.ndarray] = []
    running: np.ndarray | None = None

    for g in range(1, n_gen + 1):
        gen_start = time.perf_counter()

        rate = agent.sample_rate(min_rate, max_rate, fixed_rate)
        action = agent.select_pages(rate)
        pages_text = [book.pages[i] for i in action.pages]

        outcome = classifier.classify(pages_text)
        dist = outcome.distribution

        reward, effective_mode = compute_reward(
            dist,
            mode=reward_mode if reward_mode != "auto" else ("supervised" if gt and gt.found else "consensus"),
            consensus=running,
            ground_truth=gt.distribution if (gt and gt.found) else None,
            confidence_weight=conf_weight,
        )
        update = agent.update(action, reward)

        distributions.append(dist)
        previous = running
        running = aggregate(distributions, **agg_kwargs)
        jsd = jensen_shannon(previous, running) if previous is not None else 1.0

        record = GenerationRecord(
            index=g,
            rate=rate,
            n_pages_sampled=action.n_pages,
            pages=action.pages,
            distribution=dist,
            top_genre=GENRE_IDS[int(np.argmax(dist))],
            confidence=confidence(dist),
            reward=reward,
            reward_mode=effective_mode,
            policy_before=update["policy"],
            policy_after=update["new_policy"],
            segment_counts=update["segment_counts"],
            running_top_genre=GENRE_IDS[int(np.argmax(running))],
            running_jsd=jsd,
            elapsed_s=time.perf_counter() - gen_start,
        )
        result.generations.append(record)

        log.info(
            "  g%02d/%02d taxa=%.1f%% paginas=%d -> %-22s r=%.3f | acumulado=%-22s JSD=%.4f",
            g, n_gen, rate * 100, action.n_pages, label_pt(record.top_genre),
            reward, label_pt(record.running_top_genre), jsd,
        )

        if es_enabled and g >= es_min:
            stable_streak = stable_streak + 1 if jsd < es_thresh else 0
            if stable_streak >= es_patience:
                log.info(
                    "  convergiu na geracao %d (JSD < %.3f por %d geracoes seguidas)",
                    g, es_thresh, es_patience,
                )
                result.stopped_early = True
                break

    result.final_distribution = running if running is not None else classifier.uniform()
    result.predicted_genre = GENRE_IDS[int(np.argmax(result.final_distribution))]

    # Busca o ground truth depois do loop, salvo quando ja foi buscado acima.
    if gt is None and gt_client is not None:
        gt = gt_client.lookup(book.title, book.author)
    result.ground_truth = gt

    result.metrics = evaluate(result.final_distribution, gt)
    result.metrics["convergence_curve"] = [
        round(v, 5) for v in convergence_curve(distributions, **agg_kwargs)
    ]
    result.metrics["policy_entropy_final"] = round(agent.policy_entropy(), 4)
    result.metrics["final_policy"] = [round(p, 4) for p in agent.policy().tolist()]
    result.metrics["segment_ranges"] = agent.segment_ranges()
    result.metrics["mean_reward"] = round(
        float(np.mean([r.reward for r in result.generations])), 4
    )
    result.elapsed_s = time.perf_counter() - started
    return result


def run_corpus(
    books: list[Book],
    classifier: GenreClassifier,
    cfg: Config,
    *,
    gt_client: GroundTruthClient | None = None,
) -> list[BookResult]:
    """Roda o experimento sobre todos os livros do dataset."""
    results: list[BookResult] = []
    base_seed = int(cfg.get("generations.seed", 42))

    for i, book in enumerate(books, start=1):
        log.info("[%d/%d] %s -- %s (%d paginas uteis)",
                 i, len(books), book.title, book.author or "autor desconhecido", book.n_pages)
        try:
            # Semente derivada do indice: cada livro tem um fluxo aleatorio
            # proprio, mas a rodada inteira continua reproduzivel.
            results.append(
                run_book(book, classifier, cfg, gt_client=gt_client, seed=base_seed + i)
            )
        except Exception:
            log.exception("falha ao processar %s; seguindo para o proximo", book.title)
    return results


def corpus_summary(results: list[BookResult]) -> dict[str, Any]:
    """Metricas agregadas do dataset inteiro."""
    evaluated = [r for r in results if r.metrics.get("ground_truth_available")]
    summary: dict[str, Any] = {
        "n_books": len(results),
        "n_with_ground_truth": len(evaluated),
        "total_generations": sum(len(r.generations) for r in results),
        "total_elapsed_s": round(sum(r.elapsed_s for r in results), 2),
    }
    if not evaluated:
        summary["note"] = "nenhum livro teve ground truth resolvido; acuracia indisponivel"
        return summary

    n = len(evaluated)
    summary.update(
        {
            "top1_accuracy": round(sum(r.metrics["top1_correct"] for r in evaluated) / n, 4),
            "top3_accuracy": round(sum(r.metrics["top3_correct"] for r in evaluated) / n, 4),
            "support_accuracy": round(
                sum(r.metrics["predicted_in_gt_support"] for r in evaluated) / n, 4
            ),
            "mean_agreement": round(
                float(np.mean([r.metrics["agreement"] for r in evaluated])), 4
            ),
            "mean_jsd": round(
                float(np.mean([r.metrics["jsd_to_ground_truth"] for r in evaluated])), 4
            ),
        }
    )
    return summary
