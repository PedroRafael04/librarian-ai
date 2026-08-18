"""Carga e validacao da configuracao do experimento."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Configuracao invalida ou incoerente."""


@dataclass
class Config:
    """Wrapper fino sobre o dicionario do YAML.

    Mantemos o dicionario cru (em vez de uma arvore de dataclasses) porque o
    relatorio final grava a configuracao inteira junto dos resultados, e assim
    qualquer chave nova aparece la sem codigo extra.
    """

    data: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # -- acesso ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"chave obrigatoria ausente na configuracao: {dotted}")
        return value

    def resolve_path(self, dotted: str) -> Path:
        """Caminhos do YAML sao relativos a raiz do projeto."""
        raw = self.require(dotted)
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


_MISSING = object()


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Le o YAML e aplica overrides no formato ``{"generations.n": 20}``."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise ConfigError(f"arquivo de configuracao nao encontrado: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    cfg = Config(data=data, path=cfg_path)
    for dotted, value in (overrides or {}).items():
        if value is not None:
            cfg.set(dotted, value)

    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Falha cedo em configuracoes incoerentes -- roda antes de abrir PDFs."""
    fixed = cfg.get("sampling.fixed_rate")
    if fixed is None:
        lo = float(cfg.require("sampling.min_rate"))
        hi = float(cfg.require("sampling.max_rate"))
        if not 0.0 < lo <= hi <= 1.0:
            raise ConfigError(
                f"intervalo de amostragem invalido: min_rate={lo}, max_rate={hi} "
                "(esperado 0 < min <= max <= 1)"
            )
    elif not 0.0 < float(fixed) <= 1.0:
        raise ConfigError(f"sampling.fixed_rate deve estar em (0, 1], recebido {fixed}")

    n_gen = int(cfg.require("generations.n"))
    if n_gen < 1:
        raise ConfigError(f"generations.n deve ser >= 1, recebido {n_gen}")

    segments = int(cfg.require("sampling.segments"))
    if segments < 1:
        raise ConfigError(f"sampling.segments deve ser >= 1, recebido {segments}")

    backend = cfg.require("classifier.backend")
    if backend not in {"zeroshot", "llm", "tfidf"}:
        raise ConfigError(f"classifier.backend desconhecido: {backend!r}")

    strategy = cfg.get("agent.strategy", "bandit")
    if strategy not in {"bandit", "uniform"}:
        raise ConfigError(f"agent.strategy desconhecido: {strategy!r}")

    algorithm = cfg.get("agent.algorithm", "reinforce")
    if algorithm not in {"reinforce", "exp3"}:
        raise ConfigError(f"agent.algorithm desconhecido: {algorithm!r}")

    method = cfg.get("aggregation.method", "weighted_mean")
    if method not in {"weighted_mean", "mean", "log_pool"}:
        raise ConfigError(f"aggregation.method desconhecido: {method!r}")

    reward_mode = cfg.get("agent.reward.mode", "auto")
    if reward_mode not in {"auto", "consensus", "supervised"}:
        raise ConfigError(f"agent.reward.mode desconhecido: {reward_mode!r}")
