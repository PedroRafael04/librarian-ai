"""Testes das partes deterministicas do pipeline (sem rede e sem GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from librarian_ai.aggregation import (
    aggregate,
    confidence,
    convergence_curve,
    entropy,
    jensen_shannon,
    normalize,
    top_k,
)
from librarian_ai.agent import PageSelectionAgent
from librarian_ai.classifiers.base import chunk_pages
from librarian_ai.corpus import clean_page_text, parse_filename
from librarian_ai.ground_truth import Candidate, build_ground_truth, score_match
from librarian_ai.pipeline import compute_reward, evaluate
from librarian_ai.taxonomy import GENRE_IDS, N_GENRES, map_external_terms

from pathlib import Path


# -- taxonomia -------------------------------------------------------------

def test_taxonomia_tem_ids_unicos():
    assert len(set(GENRE_IDS)) == N_GENRES


@pytest.mark.parametrize(
    "termos,esperado",
    [
        (["Fiction / Horror / Ghost"], "terror"),
        (["Detective and mystery stories"], "misterio"),
        (["Science fiction", "Dystopia"], "ficcao_cientifica"),
        (["Love stories"], "romance"),
    ],
)
def test_mapeia_categorias_externas(termos, esperado):
    scores = map_external_terms(termos)
    assert max(scores, key=scores.get) == esperado


def test_termos_sem_correspondencia_devolvem_vazio():
    assert map_external_terms(["Accessible book", "Protected DAISY"]) == {}


# -- corpus ----------------------------------------------------------------

def test_junta_palavra_hifenizada_entre_linhas():
    assert "profundamente" in clean_page_text("profunda-\nmente triste")


def test_extrai_titulo_e_autor_do_nome_do_arquivo():
    titulo, autor = parse_filename(Path("Dom Casmurro - Machado de Assis.pdf"))
    assert titulo == "Dom Casmurro"
    assert autor == "Machado de Assis"


def test_detecta_autor_no_lado_esquerdo():
    titulo, autor = parse_filename(Path("Bram Stoker - Dracula.pdf"))
    assert titulo == "Dracula"
    assert autor == "Bram Stoker"


def test_nome_sem_separador_vira_so_titulo():
    titulo, autor = parse_filename(Path("Moby Dick.pdf"))
    assert titulo == "Moby Dick"
    assert autor is None


# -- chunking --------------------------------------------------------------

def test_chunks_respeitam_o_limite_de_caracteres():
    paginas = ["a" * 400 for _ in range(10)]
    chunks = chunk_pages(paginas, max_chars=1000)
    assert chunks and all(len(c) <= 1000 for c in chunks)


def test_pagina_maior_que_o_limite_e_truncada():
    chunks = chunk_pages(["x" * 5000], max_chars=1000)
    assert len(chunks) == 1 and len(chunks[0]) == 1000


def test_paginas_vazias_sao_ignoradas():
    assert chunk_pages(["", "   ", "\n"], max_chars=100) == []


# -- agregacao -------------------------------------------------------------

def test_normalize_projeta_no_simplexo():
    v = normalize(np.array([2.0, -1.0, 3.0]))
    assert v.sum() == pytest.approx(1.0)
    assert (v >= 0).all()


def test_vetor_nulo_vira_uniforme():
    v = normalize(np.zeros(4))
    assert v == pytest.approx(np.full(4, 0.25))


def test_entropia_maxima_na_uniforme():
    assert entropy(np.full(N_GENRES, 1 / N_GENRES)) == pytest.approx(1.0)


def test_confianca_maxima_na_distribuicao_degenerada():
    d = np.zeros(N_GENRES)
    d[0] = 1.0
    assert confidence(d) == pytest.approx(1.0)


def test_jsd_e_zero_para_distribuicoes_iguais():
    d = normalize(np.arange(1, N_GENRES + 1, dtype=float))
    assert jensen_shannon(d, d) == pytest.approx(0.0, abs=1e-9)


def test_jsd_e_um_para_suportes_disjuntos():
    a, b = np.zeros(4), np.zeros(4)
    a[0], b[3] = 1.0, 1.0
    assert jensen_shannon(a, b) == pytest.approx(1.0)


def test_jsd_e_simetrico():
    a = normalize(np.array([0.7, 0.2, 0.1]))
    b = normalize(np.array([0.1, 0.5, 0.4]))
    assert jensen_shannon(a, b) == pytest.approx(jensen_shannon(b, a))


@pytest.mark.parametrize("metodo", ["mean", "weighted_mean", "log_pool"])
def test_agregacao_devolve_distribuicao_valida(metodo):
    dists = [normalize(np.random.default_rng(i).random(N_GENRES)) for i in range(5)]
    out = aggregate(dists, method=metodo)
    assert out.sum() == pytest.approx(1.0)
    assert (out >= 0).all()


def test_agregacao_pondera_a_favor_da_geracao_confiante():
    confiante = np.zeros(N_GENRES)
    confiante[0] = 1.0
    difusa = np.full(N_GENRES, 1 / N_GENRES)
    out = aggregate([confiante, difusa], method="weighted_mean", weight_by_confidence=True)
    simples = aggregate([confiante, difusa], method="mean")
    assert out[0] > simples[0]


def test_convergencia_cai_quando_as_geracoes_se_repetem():
    d = normalize(np.array([0.6] + [0.4 / (N_GENRES - 1)] * (N_GENRES - 1)))
    curva = convergence_curve([d] * 6)
    assert curva[-1] < 1e-6


def test_top_k_devolve_em_ordem_decrescente():
    d = normalize(np.arange(N_GENRES, dtype=float))
    pares = top_k(d, 3)
    assert [p for _, p in pares] == sorted([p for _, p in pares], reverse=True)


# -- agente ----------------------------------------------------------------

def test_amostra_a_quantidade_certa_de_paginas():
    agente = PageSelectionAgent(n_pages=200, n_segments=10)
    acao = agente.select_pages(0.25)
    assert acao.n_pages == 50
    assert len(set(acao.pages)) == 50  # sem repeticao
    assert all(0 <= p < 200 for p in acao.pages)


def test_taxa_sorteada_respeita_o_intervalo():
    agente = PageSelectionAgent(n_pages=100, rng=np.random.default_rng(0))
    taxas = [agente.sample_rate(0.15, 0.30, None) for _ in range(200)]
    assert all(0.15 <= t <= 0.30 for t in taxas)


def test_taxa_fixa_ignora_o_intervalo():
    agente = PageSelectionAgent(n_pages=100)
    assert agente.sample_rate(0.15, 0.30, 0.5) == 0.5


def test_segmentos_nao_excedem_as_paginas():
    agente = PageSelectionAgent(n_pages=3, n_segments=10)
    assert agente.n_segments == 3


def test_pedir_todas_as_paginas_devolve_todas():
    agente = PageSelectionAgent(n_pages=40, n_segments=7)
    assert agente.select_pages(1.0).n_pages == 40


def test_politica_uniforme_nao_aprende():
    agente = PageSelectionAgent(n_pages=100, strategy="uniform")
    antes = agente.policy().copy()
    agente.update(agente.select_pages(0.2), reward=1.0)
    assert agente.policy() == pytest.approx(antes)


def test_bandit_favorece_o_segmento_recompensado():
    """Recompensar repetidamente amostras concentradas no segmento 0 deve
    aumentar a probabilidade daquele segmento."""
    agente = PageSelectionAgent(
        n_pages=100, n_segments=5, learning_rate=0.5, rng=np.random.default_rng(7)
    )
    inicial = agente.policy()[0]
    for _ in range(30):
        acao = agente.select_pages(0.2)
        # Recompensa alta quando a amostra pegou muito do segmento 0.
        recompensa = acao.segment_counts[0] / max(1, acao.n_pages)
        agente.update(acao, recompensa)
    assert agente.policy()[0] > inicial


def test_politica_sempre_soma_um():
    agente = PageSelectionAgent(n_pages=100, n_segments=6, rng=np.random.default_rng(3))
    for _ in range(15):
        agente.update(agente.select_pages(0.2), reward=float(np.random.rand()))
        assert agente.policy().sum() == pytest.approx(1.0)


def test_entropia_da_politica_e_maxima_no_inicio():
    agente = PageSelectionAgent(n_pages=100, n_segments=8)
    assert agente.policy_entropy() == pytest.approx(1.0)


def test_livro_sem_paginas_e_rejeitado():
    with pytest.raises(ValueError):
        PageSelectionAgent(n_pages=0)


# -- recompensa e avaliacao ------------------------------------------------

def test_recompensa_inicial_usa_bootstrap():
    d = np.full(N_GENRES, 1 / N_GENRES)
    _, modo = compute_reward(d, mode="consensus", consensus=None,
                             ground_truth=None, confidence_weight=0.3)
    assert modo == "bootstrap"


def test_concordar_com_o_consenso_rende_mais_que_divergir():
    consenso = np.zeros(N_GENRES)
    consenso[0] = 1.0
    igual = consenso.copy()
    oposto = np.zeros(N_GENRES)
    oposto[5] = 1.0
    r_igual, _ = compute_reward(igual, mode="consensus", consensus=consenso,
                                ground_truth=None, confidence_weight=0.3)
    r_oposto, _ = compute_reward(oposto, mode="consensus", consensus=consenso,
                                 ground_truth=None, confidence_weight=0.3)
    assert r_igual > r_oposto


def test_recompensa_fica_no_intervalo_unitario():
    rng = np.random.default_rng(1)
    for _ in range(50):
        d = normalize(rng.random(N_GENRES))
        c = normalize(rng.random(N_GENRES))
        r, _ = compute_reward(d, mode="consensus", consensus=c,
                              ground_truth=None, confidence_weight=0.3)
        assert 0.0 <= r <= 1.0


def test_avaliacao_sem_ground_truth_nao_quebra():
    m = evaluate(np.full(N_GENRES, 1 / N_GENRES), None)
    assert m["ground_truth_available"] is False


def test_avaliacao_detecta_acerto_top1():
    gt = build_ground_truth([Candidate(terms=["Fiction / Horror"], score=100.0)], "teste")
    final = np.zeros(N_GENRES)
    final[GENRE_IDS.index("terror")] = 1.0
    m = evaluate(final, gt)
    assert m["top1_correct"] and m["rank_of_ground_truth"] == 1


# -- ground truth ----------------------------------------------------------

def test_score_de_casamento_alto_para_titulo_identico():
    assert score_match("Dracula", "Bram Stoker", "Dracula", ["Bram Stoker"]) > 95


def test_score_baixo_para_livro_diferente():
    assert score_match("Dracula", "Bram Stoker", "Moby Dick", ["Herman Melville"]) < 50


def test_termos_sem_mapeamento_nao_viram_ground_truth():
    gt = build_ground_truth([Candidate(terms=["Accessible book"], score=100.0)], "teste")
    assert not gt.found


def test_candidato_abaixo_do_limiar_e_rejeitado():
    gt = build_ground_truth([Candidate(terms=["Horror"], score=10.0)], "teste")
    assert not gt.found


def test_agregacao_de_edicoes_dilui_registro_destoante():
    """Uma adaptacao teatral isolada nao deve sobrepor varias edicoes de terror
    -- foi exatamente o caso do Frankenstein no Open Library."""
    candidatos = [
        Candidate(terms=["Drama"], score=100.0),
        Candidate(terms=["Horror tales"], score=95.0),
        Candidate(terms=["Gothic fiction"], score=93.0),
        Candidate(terms=["Horror"], score=90.0),
    ]
    gt = build_ground_truth(candidatos, "teste")
    assert gt.found and gt.top_genre == "terror"


def test_sem_candidatos_devolve_nao_encontrado():
    assert not build_ground_truth([], "teste").found


@pytest.mark.parametrize(
    "nome,titulo,autor",
    [
        ("Dom Casmurro - Machado de Assis", "Dom Casmurro", "Machado de Assis"),
        ("Bram Stoker - Dracula", "Dracula", "Bram Stoker"),
        ("Pride and Prejudice - Jane Austen", "Pride and Prejudice", "Jane Austen"),
        ("The Adventures of Sherlock Holmes - Arthur Conan Doyle",
         "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
        ("Mary Shelley - Frankenstein", "Frankenstein", "Mary Shelley"),
        ("O Cortico - Aluisio Azevedo", "O Cortico", "Aluisio Azevedo"),
        ("Ludwig van Beethoven - Cartas", "Cartas", "Ludwig van Beethoven"),
    ],
)
def test_convencoes_de_nome_de_arquivo(nome, titulo, autor):
    assert parse_filename(Path(f"{nome}.pdf")) == (titulo, autor)
