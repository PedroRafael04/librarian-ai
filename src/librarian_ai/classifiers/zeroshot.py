"""Classificacao zero-shot por inferencia textual (NLI), acelerada por CUDA.

O modelo (por padrao ``facebook/bart-large-mnli``) julga, para cada bloco de
texto, o quanto ele acarreta a hipotese "This text is <genero>". Nao ha treino
supervisionado: a taxonomia de ``taxonomy.py`` entra como rotulo candidato, o
que permite mexer nos generos sem reanotar nada.
"""

from __future__ import annotations

import logging

import numpy as np

from ..taxonomy import N_GENRES, hypotheses
from .base import ClassificationResult, GenreClassifier, chunk_pages

log = logging.getLogger(__name__)

# Textos deliberadamente sem sinal de genero. A distribuicao que o modelo
# produz sobre eles e, por construcao, o seu vies a priori sobre os rotulos.
CALIBRATION_TEXTS: tuple[str, ...] = (
    "N/A",
    "This is a text.",
    "The following pages contain the continuation of the previous chapter.",
    "Chapter twelve. It was a day. He said something and then she replied.",
    "The book was printed and bound. It has a cover, pages and a title.",
)


class ZeroShotClassifier(GenreClassifier):
    name = "zeroshot"

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str = "auto",
        fp16: bool = True,
        batch_size: int = 8,
        max_chars_per_chunk: int = 3500,
        multi_label: bool = False,
        sharpen: float = 1.0,
        calibrate: bool = True,
        calibration_strength: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.device_pref = device
        self.fp16 = fp16
        self.batch_size = batch_size
        self.max_chars_per_chunk = max_chars_per_chunk
        self.multi_label = multi_label
        self.sharpen = sharpen
        self.calibrate = calibrate
        self.calibration_strength = float(np.clip(calibration_strength, 0.0, 1.0))
        self._prior: np.ndarray | None = None
        self._calibrating = False
        self._pipe = None
        self._device: str | None = None
        self._labels = hypotheses()

    # -- ciclo de vida -----------------------------------------------------
    def _resolve_device(self) -> tuple[int, str]:
        """Devolve (device_index_para_pipeline, rotulo_legivel)."""
        import torch

        want = self.device_pref
        if want == "cpu":
            return -1, "cpu"
        if want in ("auto", "cuda"):
            if torch.cuda.is_available():
                return 0, f"cuda:0 ({torch.cuda.get_device_name(0)})"
            if want == "cuda":
                raise RuntimeError(
                    "device='cuda' pedido, mas torch.cuda.is_available() e False. "
                    "Instale o torch com CUDA: "
                    "pip install -r requirements-cuda.txt"
                )
            log.warning("CUDA indisponivel; caindo para CPU (bem mais lento)")
            return -1, "cpu"
        raise ValueError(f"device invalido: {self.device_pref!r}")

    def warmup(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        device_idx, device_label = self._resolve_device()
        self._device = device_label
        dtype = torch.float16 if (self.fp16 and device_idx >= 0) else torch.float32

        log.info("carregando %s em %s (dtype=%s)", self.model_name, device_label, dtype)
        kwargs = {
            "model": self.model_name,
            "device": device_idx,
            **self._dtype_kwarg(dtype),
        }
        self._pipe = pipeline("zero-shot-classification", **kwargs)

        if self.calibrate:
            self._fit_prior()

    def _fit_prior(self) -> None:
        """Estima o vies do modelo sobre os rotulos usando texto sem conteudo.

        Modelos NLI zero-shot nao tratam os rotulos de forma equanime: medido
        neste projeto com bart-large-mnli, texto totalmente neutro ja recebia
        Drama=0.176 contra Terror=0.042 -- um vies de 4.2x que fazia Dracula e
        Frankenstein serem classificados como Drama. Dividir a saida por esse
        prior (calibracao contextual, Zhao et al. 2021, "Calibrate Before Use")
        remove o efeito sem precisar de dados rotulados.
        """
        self._calibrating = True
        try:
            prior = self.classify(list(CALIBRATION_TEXTS)).distribution
        finally:
            self._calibrating = False

        # Piso evita divisao explosiva num rotulo que o modelo quase nunca usa.
        self._prior = np.maximum(prior, 1e-3)
        log.debug("prior calibrado: %s", np.round(self._prior, 4).tolist())

    @staticmethod
    def _dtype_kwarg(dtype) -> dict:
        """transformers >= 5 renomeou ``torch_dtype`` para ``dtype``."""
        import inspect

        from transformers import pipeline

        params = inspect.signature(pipeline).parameters
        if "dtype" in params:
            return {"dtype": dtype}
        if "torch_dtype" in params:
            return {"torch_dtype": dtype}
        return {}

    def close(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # -- inferencia --------------------------------------------------------
    def classify(self, pages: list[str]) -> ClassificationResult:
        self.warmup()
        chunks = chunk_pages(pages, self.max_chars_per_chunk)
        if not chunks:
            return ClassificationResult(self.uniform(), 0, self.name, {"empty": True})

        scores = np.zeros(N_GENRES, dtype=np.float64)
        weights = 0.0
        processed = 0

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            try:
                outputs = self._pipe(
                    batch, candidate_labels=self._labels, multi_label=self.multi_label
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self.batch_size > 1:
                    # 8GB de VRAM: com chunks longos o batch pode estourar.
                    # Reduzir e seguir e melhor que abortar o experimento.
                    self.batch_size = max(1, self.batch_size // 2)
                    log.warning("VRAM insuficiente; batch_size -> %d", self.batch_size)
                    self._empty_cache()
                    return self.classify(pages)
                raise

            if isinstance(outputs, dict):
                outputs = [outputs]

            for chunk, out in zip(batch, outputs):
                vec = self._to_vector(out)
                # Chunks maiores carregam mais evidencia -> pesam mais na media.
                w = float(len(chunk))
                scores += w * vec
                weights += w
                processed += 1

        dist = scores / weights if weights > 0 else self.uniform()
        total = dist.sum()
        dist = dist / total if total > 0 else self.uniform()

        if self._prior is not None and not self._calibrating and self.calibration_strength > 0:
            # alpha=0 nao calibra (o vies de "Drama" passa inteiro); alpha=1
            # calibra totalmente, mas superamplifica rotulos de prior muito
            # baixo -- medido, "Tragedia" (prior 0.030) passava a vencer quase
            # tudo. O expoente interpola entre os dois extremos.
            dist = dist / np.power(self._prior, self.calibration_strength)
            dist = dist / dist.sum()

        if self.sharpen and self.sharpen != 1.0:
            # Transformacao de potencia: monotonica, portanto NAO altera o
            # ranking de generos. Serve so para dar contraste a distribuicao,
            # que sai bem achatada do NLI sobre 14 rotulos -- sem isso o peso
            # por confianca na agregacao fica praticamente uniforme.
            dist = np.power(dist, self.sharpen)
            dist = dist / dist.sum()

        return ClassificationResult(
            distribution=dist,
            n_chunks=processed,
            backend=self.name,
            meta={"device": self._device, "model": self.model_name,
                  "batch_size": self.batch_size, "sharpen": self.sharpen,
                  "calibrated": self._prior is not None,
                  "calibration_strength": self.calibration_strength,
                  "multi_label": self.multi_label},
        )

    def _to_vector(self, output: dict) -> np.ndarray:
        """Reordena a saida do pipeline (ordenada por score) na ordem canonica."""
        by_label = dict(zip(output["labels"], output["scores"]))
        vec = np.array([by_label.get(h, 0.0) for h in self._labels], dtype=np.float64)
        total = vec.sum()
        # Em multi_label os scores sao sigmoides independentes e nao somam 1.
        return vec / total if total > 0 else self.uniform()

    @staticmethod
    def _empty_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
