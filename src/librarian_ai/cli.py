"""Interface de linha de comando do Librarian AI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .classifiers import build_classifier
from .config import ConfigError, load_config
from .corpus import load_corpus
from .ground_truth import GroundTruthClient
from .pipeline import corpus_summary, run_corpus
from .report import write_reports
from .taxonomy import label_pt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="librarian-ai",
        description="Classificacao de genero de classicos da literatura por "
                    "amostragem de paginas guiada por aprendizado por reforco.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", type=Path, default=None,
                   help="arquivo YAML de configuracao")
    p.add_argument("-i", "--input-dir", type=Path, default=None,
                   help="diretorio com os PDFs (sobrescreve o YAML)")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="diretorio de saida (sobrescreve o YAML)")

    g = p.add_argument_group("parametros do experimento")
    g.add_argument("-n", "--generations", type=int, default=None,
                   help="numero de geracoes N")
    g.add_argument("--min-rate", type=float, default=None,
                   help="fracao minima de paginas amostradas (ex.: 0.15)")
    g.add_argument("--max-rate", type=float, default=None,
                   help="fracao maxima de paginas amostradas (ex.: 0.30)")
    g.add_argument("--fixed-rate", type=float, default=None,
                   help="fixa a fracao de paginas, ignorando o intervalo")
    g.add_argument("--segments", type=int, default=None,
                   help="numero de segmentos/bracos do bandit")
    g.add_argument("--seed", type=int, default=None, help="semente aleatoria")

    a = p.add_argument_group("agente")
    a.add_argument("--strategy", choices=["bandit", "uniform"], default=None,
                   help="bandit aprende a politica; uniform e o baseline")
    a.add_argument("--algorithm", choices=["reinforce", "exp3"], default=None)
    a.add_argument("--learning-rate", type=float, default=None)
    a.add_argument("--reward-mode", choices=["auto", "consensus", "supervised"], default=None)

    m = p.add_argument_group("modelo")
    m.add_argument("--backend", choices=["zeroshot", "llm", "tfidf"], default=None)
    m.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    m.add_argument("--batch-size", type=int, default=None)
    m.add_argument("--aggregation", choices=["weighted_mean", "mean", "log_pool"], default=None)

    o = p.add_argument_group("saida")
    o.add_argument("--no-ground-truth", action="store_true",
                   help="pula a consulta as APIs externas (modo offline)")
    o.add_argument("--no-plots", action="store_true", help="nao gera graficos")
    o.add_argument("-v", "--verbose", action="store_true", help="log em nivel DEBUG")
    o.add_argument("-q", "--quiet", action="store_true", help="so avisos e erros")
    return p


def _overrides(args: argparse.Namespace) -> dict:
    ov = {
        "corpus.input_dir": str(args.input_dir) if args.input_dir else None,
        "output.dir": str(args.output_dir) if args.output_dir else None,
        "generations.n": args.generations,
        "generations.seed": args.seed,
        "sampling.min_rate": args.min_rate,
        "sampling.max_rate": args.max_rate,
        "sampling.fixed_rate": args.fixed_rate,
        "sampling.segments": args.segments,
        "agent.strategy": args.strategy,
        "agent.algorithm": args.algorithm,
        "agent.learning_rate": args.learning_rate,
        "agent.reward.mode": args.reward_mode,
        "classifier.backend": args.backend,
        "classifier.zeroshot.device": args.device,
        "classifier.zeroshot.batch_size": args.batch_size,
        "aggregation.method": args.aggregation,
    }
    if args.no_ground_truth:
        ov["ground_truth.enabled"] = False
    if args.no_plots:
        ov["output.save_plots"] = False
    return ov


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)
    # Bibliotecas de rede/ML sao ruidosas em INFO; so interessam se -v.
    for noisy in ("httpx", "urllib3", "transformers", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING if not args.verbose else logging.INFO)

    log = logging.getLogger("librarian_ai")

    try:
        cfg = load_config(args.config, _overrides(args))
    except ConfigError as exc:
        print(f"erro de configuracao: {exc}", file=sys.stderr)
        return 2

    try:
        books = load_corpus(
            cfg.resolve_path("corpus.input_dir"),
            cache_dir=cfg.resolve_path("corpus.cache_dir"),
            min_chars_per_page=int(cfg.get("corpus.min_chars_per_page", 120)),
            max_pages=cfg.get("corpus.max_pages"),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"erro ao carregar o dataset: {exc}", file=sys.stderr)
        return 1

    log.info("dataset: %d livro(s), %d paginas uteis no total",
             len(books), sum(b.n_pages for b in books))

    gt_client = None
    if cfg.get("ground_truth.enabled", True):
        gt_client = GroundTruthClient(
            providers=cfg.get("ground_truth.providers", ["google_books", "open_library"]),
            cache_dir=cfg.resolve_path("ground_truth.cache_dir"),
            timeout=int(cfg.get("ground_truth.timeout", 15)),
            google_books_api_key=cfg.get("ground_truth.google_books_api_key"),
        )

    classifier = build_classifier(cfg)
    try:
        classifier.warmup()
    except RuntimeError as exc:
        print(f"erro ao inicializar o classificador: {exc}", file=sys.stderr)
        return 1

    try:
        results = run_corpus(books, classifier, cfg, gt_client=gt_client)
    finally:
        classifier.close()

    if not results:
        print("nenhum livro pode ser processado", file=sys.stderr)
        return 1

    run_dir = write_reports(results, cfg, output_dir=args.output_dir)
    _print_summary(results, run_dir)
    return 0


def _print_summary(results, run_dir: Path) -> None:
    summary = corpus_summary(results)
    print("\n" + "=" * 74)
    print("RESULTADO FINAL")
    print("=" * 74)
    for r in results:
        m = r.metrics
        gt = m.get("ground_truth_genre_pt") or "nao resolvido"
        status = ""
        if m.get("ground_truth_available"):
            status = "  [top-1 OK]" if m["top1_correct"] else (
                "  [top-3 OK]" if m["top3_correct"] else "  [divergente]"
            )
        print(f"\n{r.title}")
        print(f"  previsto     : {label_pt(r.predicted_genre)}{status}")
        print(f"  ground truth : {gt}")
        if m.get("ground_truth_available"):
            print(f"  concordancia : {m['agreement']:.3f}   (JSD {m['jsd_to_ground_truth']:.3f})")
        print(f"  geracoes     : {len(r.generations)}"
              + (" (convergiu antes do fim)" if r.stopped_early else ""))

    print("\n" + "-" * 74)
    if summary.get("n_with_ground_truth"):
        print(f"Acuracia top-1 : {summary['top1_accuracy']:.1%}")
        print(f"Acuracia top-3 : {summary['top3_accuracy']:.1%}")
        print(f"Concordancia   : {summary['mean_agreement']:.3f}")
    else:
        print(summary.get("note", "sem ground truth disponivel"))
    print(f"\nRelatorios em: {run_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
