#!/usr/bin/env python
"""Varredura da forca de calibracao (alpha) contra o ground truth do catalogo.

Divide-se a saida do modelo por ``prior ** alpha``. Com ``alpha=0`` nao ha
calibracao (o vies do modelo passa inteiro, e "Drama" vence quase sempre); com
``alpha=1`` a calibracao e total (e generos de prior muito baixo, como
"Tragedia", ficam superamplificados). Este script mede o meio-termo.

    python scripts/tune_calibration.py

As distribuicoes brutas sao classificadas uma unica vez por livro e a
calibracao e aplicada em pos-processamento, entao a varredura inteira custa uma
passada de GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from librarian_ai.agent import PageSelectionAgent  # noqa: E402
from librarian_ai.aggregation import aggregate, jensen_shannon  # noqa: E402
from librarian_ai.classifiers.zeroshot import ZeroShotClassifier  # noqa: E402
from librarian_ai.corpus import load_corpus  # noqa: E402
from librarian_ai.ground_truth import GroundTruthClient  # noqa: E402
from librarian_ai.taxonomy import GENRE_IDS, label_pt  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=3,
                        help="amostragens independentes por livro")
    parser.add_argument("--rate", type=float, default=0.25)
    parser.add_argument("--sharpen", type=float, default=2.0)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    books = load_corpus(PROJECT_ROOT / "data" / "raw",
                        cache_dir=PROJECT_ROOT / "data" / "cache")
    gt_client = GroundTruthClient(cache_dir=PROJECT_ROOT / "data" / "cache" / "ground_truth")

    # Sem calibracao e sem sharpen: queremos a saida crua do modelo.
    clf = ZeroShotClassifier(batch_size=8, calibrate=False, sharpen=1.0)
    clf.warmup()

    print(f"classificando {len(books)} livro(s) x {args.samples} amostra(s)...")
    raw: dict[str, list[np.ndarray]] = {}
    truth: dict[str, np.ndarray] = {}

    for book in books:
        gt = gt_client.lookup(book.title, book.author)
        if not gt.found:
            print(f"  [skip] {book.title}: sem ground truth")
            continue
        truth[book.title] = gt.distribution

        agent = PageSelectionAgent(n_pages=book.n_pages, rng=np.random.default_rng(11))
        dists = []
        for _ in range(args.samples):
            action = agent.select_pages(args.rate)
            dists.append(clf.classify([book.pages[i] for i in action.pages]).distribution)
        raw[book.title] = dists
        print(f"  [ok] {book.title}")

    # Prior medido no mesmo modelo, com a calibracao desligada.
    prior = np.maximum(_measure_prior(clf), 1e-3)

    if not raw:
        print("nenhum livro com ground truth; nada a ajustar")
        return 1

    print("\nalpha |  top-1  |  top-3  |  JSD medio | vencedores")
    print("-" * 78)

    best = (None, -1.0, 1.0)
    for alpha in args.alphas:
        top1 = top3 = 0
        jsds = []
        winners = []
        for title, dists in raw.items():
            calibrated = []
            for d in dists:
                c = d / (prior ** alpha)
                c = c / c.sum()
                c = np.power(c, args.sharpen)
                calibrated.append(c / c.sum())
            final = aggregate(calibrated, method="weighted_mean")

            order = np.argsort(final)[::-1]
            predicted = GENRE_IDS[int(order[0])]
            gt_top = GENRE_IDS[int(np.argmax(truth[title]))]
            top1 += predicted == gt_top
            top3 += gt_top in [GENRE_IDS[i] for i in order[:3]]
            jsds.append(jensen_shannon(final, truth[title]))
            winners.append(label_pt(predicted)[:12])

        n = len(raw)
        acc1, acc3, mjsd = top1 / n, top3 / n, float(np.mean(jsds))
        print(f" {alpha:.2f} |  {acc1:5.0%}  |  {acc3:5.0%}  |   {mjsd:.3f}    | "
              + ", ".join(winners))
        # Criterio: top-1 primeiro, JSD como desempate.
        if (acc1, -mjsd) > (best[1], -best[2]):
            best = (alpha, acc1, mjsd)

    print("-" * 78)
    print(f"\nmelhor alpha: {best[0]:.2f}  (top-1 {best[1]:.0%}, JSD medio {best[2]:.3f})")
    print("\nAviso: com poucos livros esta escolha e indicativa, nao conclusiva.")
    print("Rode com um dataset maior antes de fixar o valor no relatorio.")
    clf.close()
    return 0


def _measure_prior(clf: ZeroShotClassifier) -> np.ndarray:
    from librarian_ai.classifiers.zeroshot import CALIBRATION_TEXTS

    return clf.classify(list(CALIBRATION_TEXTS)).distribution


if __name__ == "__main__":
    raise SystemExit(main())
