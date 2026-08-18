"""Geracao dos artefatos de saida: JSON, CSV, graficos e resumo em Markdown."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config
from .pipeline import BookResult, corpus_summary
from .taxonomy import GENRE_IDS, label_pt

log = logging.getLogger(__name__)


def write_reports(
    results: list[BookResult],
    cfg: Config,
    output_dir: Path | None = None,
) -> Path:
    """Grava todos os artefatos em ``results/run_<timestamp>/`` e devolve o caminho."""
    base = Path(output_dir) if output_dir else cfg.resolve_path("output.dir")
    run_dir = base / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = corpus_summary(results)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.as_dict(),
        "summary": summary,
        "books": [r.as_dict() for r in results],
    }
    (run_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_books_csv(run_dir / "books.csv", results)
    _write_generations_csv(run_dir / "generations.csv", results)
    (run_dir / "README.md").write_text(_markdown(results, summary, cfg), encoding="utf-8")

    if cfg.get("output.save_plots", True):
        try:
            _plots(run_dir, results)
        except Exception as exc:  # matplotlib e opcional para o resultado
            log.warning("nao foi possivel gerar os graficos: %s", exc)

    return run_dir


def _write_books_csv(path: Path, results: list[BookResult]) -> None:
    columns = [
        "book_id", "title", "author", "n_pages_total", "n_generations_run",
        "stopped_early", "predicted_genre", "predicted_genre_pt", "ground_truth_genre",
        "top1_correct", "top3_correct", "predicted_in_gt_support", "agreement",
        "jsd_to_ground_truth", "final_confidence", "mean_reward", "elapsed_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.as_dict()
            row.update(r.metrics)
            writer.writerow(row)


def _write_generations_csv(path: Path, results: list[BookResult]) -> None:
    columns = [
        "book_id", "generation", "rate", "n_pages_sampled", "top_genre",
        "confidence", "reward", "reward_mode", "running_top_genre", "running_jsd",
        "elapsed_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            for g in r.generations:
                row = g.as_dict()
                row["book_id"] = r.book_id
                writer.writerow(row)


def _markdown(results: list[BookResult], summary: dict[str, Any], cfg: Config) -> str:
    lines: list[str] = [
        "# Librarian AI -- resultados",
        "",
        f"Gerado em {datetime.now():%d/%m/%Y %H:%M}",
        "",
        "## Configuracao",
        "",
        f"- Backend de classificacao: `{cfg.get('classifier.backend')}`",
        f"- Estrategia do agente: `{cfg.get('agent.strategy')}` / `{cfg.get('agent.algorithm')}`",
        f"- Geracoes: {cfg.get('generations.n')}",
        f"- Taxa de amostragem: {cfg.get('sampling.min_rate'):.0%} a {cfg.get('sampling.max_rate'):.0%}",
        f"- Segmentos (bracos): {cfg.get('sampling.segments')}",
        f"- Interpolacao: `{cfg.get('aggregation.method')}`",
        "",
        "## Resumo",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")

    lines += ["", "## Por livro", "",
              "| Livro | Previsto | Ground truth | Top-1 | Top-3 | Concordancia |",
              "|---|---|---|:--:|:--:|--:|"]

    for r in results:
        m = r.metrics
        gt = m.get("ground_truth_genre_pt") or "--"
        mark = lambda v: "OK" if v else "X"  # noqa: E731
        top1 = mark(m["top1_correct"]) if m.get("ground_truth_available") else "--"
        top3 = mark(m["top3_correct"]) if m.get("ground_truth_available") else "--"
        agree = f"{m['agreement']:.3f}" if m.get("ground_truth_available") else "--"
        lines.append(
            f"| {r.title} | {label_pt(r.predicted_genre)} | {gt} | {top1} | {top3} | {agree} |"
        )

    lines += ["", "## Detalhe por livro", ""]
    for r in results:
        lines += [f"### {r.title}", ""]
        if r.author:
            lines.append(f"*{r.author}*")
            lines.append("")
        lines.append(
            f"{r.n_pages_total} paginas uteis, {len(r.generations)} geracoes"
            + (" (parada antecipada por convergencia)" if r.stopped_early else "")
        )
        lines += ["", "Distribuicao final (top 5):", ""]
        pairs = sorted(
            zip(GENRE_IDS, r.final_distribution), key=lambda kv: kv[1], reverse=True
        )[:5]
        for gid, p in pairs:
            bar = "#" * int(round(p * 40))
            lines.append(f"- `{label_pt(gid):26s}` {p:.3f} {bar}")

        gt = r.ground_truth
        lines += ["", "Ground truth:", ""]
        if gt and gt.found:
            lines += [
                f"- Provedor: `{gt.provider}` (score de casamento {gt.match_score:.0f})",
                f"- Registro: {gt.matched_title} -- {gt.matched_author or 'autor n/d'}",
                f"- Genero: **{label_pt(gt.top_genre)}**",
                f"- Termos brutos: {', '.join(gt.raw_terms[:12])}",
            ]
        else:
            lines.append(f"- Nao resolvido: {gt.note if gt else 'consulta desabilitada'}")
        lines.append("")

    return "\n".join(lines)


def _plots(run_dir: Path, results: list[BookResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: nao tenta abrir janela
    import matplotlib.pyplot as plt

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Convergencia: JSD entre agregacoes consecutivas.
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        curve = r.metrics.get("convergence_curve", [])[1:]  # 1a geracao e sempre 1.0
        if curve:
            ax.plot(range(2, len(curve) + 2), curve, marker="o", label=r.title[:28])
    ax.set_xlabel("Geracao")
    ax.set_ylabel("JSD(agregado_g, agregado_g-1)")
    ax.set_title("Convergencia da interpolacao entre geracoes")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(alpha=0.3)
    if results:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "convergencia.png", dpi=130)
    plt.close(fig)

    # Recompensa por geracao: evidencia do aprendizado do agente.
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        rewards = [g.reward for g in r.generations]
        if rewards:
            ax.plot(range(1, len(rewards) + 1), rewards, marker="s", label=r.title[:28])
    ax.set_xlabel("Geracao")
    ax.set_ylabel("Recompensa")
    ax.set_title("Recompensa por geracao (aprendizado do agente)")
    ax.grid(alpha=0.3)
    if results:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "recompensa.png", dpi=130)
    plt.close(fig)

    # Politica final por livro: quais regioes o agente aprendeu a preferir.
    for r in results:
        policy = r.metrics.get("final_policy")
        if not policy:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, len(policy) + 1), policy)
        ax.axhline(1 / len(policy), color="crimson", ls="--", lw=1,
                   label="politica uniforme")
        ax.set_xlabel("Segmento do livro (inicio -> fim)")
        ax.set_ylabel("P(amostrar)")
        ax.set_title(f"Politica aprendida -- {r.title[:44]}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in r.book_id)[:60]
        fig.savefig(plots_dir / f"politica_{safe}.png", dpi=130)
        plt.close(fig)
