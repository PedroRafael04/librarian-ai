#!/usr/bin/env python
"""Ponto de entrada do projeto: `python run.py [opcoes]`.

Adiciona ``src/`` ao path para que o projeto rode sem instalacao previa
(``pip install -e .``), o que simplifica a avaliacao do trabalho.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from librarian_ai.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
