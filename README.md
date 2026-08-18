# Librarian AI

Classificação de gênero literário a partir de **amostragem parcial de páginas**,
guiada por um agente de **aprendizado por reforço**, com validação contra
catálogos bibliográficos públicos.

Dado um dataset de PDFs de clássicos da literatura, o sistema:

1. sorteia entre **15% e 30%** das páginas de cada livro (faixa parametrizável);
2. aplica **PLN** sobre essa amostra e produz uma classificação de gênero — isso
   é uma **geração**;
3. repete por **N gerações**, **interpolando** as classificações para refinar o
   veredito;
4. compara o resultado final com a classificação real obtida via **API do
   Google Books / Open Library**.

O diferencial em relação a uma amostragem puramente aleatória é o passo 1: um
bandit aprende, ao longo das gerações, **quais regiões do livro vale a pena
ler**.

---

## Instalação

Requer Python 3.11+ (testado em 3.14) e, para a GPU, uma placa NVIDIA.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt   # PyTorch + CUDA 12.8
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sem GPU, pule o `requirements-cuda.txt` e rode com `--device cpu` (bem mais
lento: o modelo tem 400M de parâmetros).

## Dataset

Coloque os PDFs em `data/raw/`, nomeados como `Título - Autor.pdf` (a convenção
`Autor - Título.pdf` também é reconhecida). O nome do arquivo é o que alimenta a
busca do ground truth.

Para gerar um dataset de exemplo com 8 clássicos de domínio público:

```powershell
.venv\Scripts\python.exe scripts\fetch_samples.py
```

O script baixa os textos do Project Gutenberg, remove o cabeçalho de licença e
renderiza PDFs paginados com camada de texto extraível.

> PDFs que são digitalizações sem OCR não têm camada de texto e serão
> descartados com um aviso.

## Uso

```powershell
.venv\Scripts\python.exe run.py                       # usa config.yaml
.venv\Scripts\python.exe run.py -n 20 --min-rate 0.2 --max-rate 0.35
.venv\Scripts\python.exe run.py --strategy uniform    # baseline sem aprendizado
.venv\Scripts\python.exe run.py --no-ground-truth     # modo offline
.venv\Scripts\python.exe run.py --backend tfidf       # baseline leve, sem GPU
```

Cada execução grava em `results/run_<timestamp>/`:

| Arquivo | Conteúdo |
|---|---|
| `results.json` | tudo: config, métricas, cada geração, páginas sorteadas |
| `books.csv` | uma linha por livro — pronto para planilha |
| `generations.csv` | uma linha por geração — para gráficos próprios |
| `README.md` | resumo legível com as distribuições finais |
| `plots/` | convergência, recompensa e política aprendida por livro |

---

## Como funciona

### 1. O agente de RL

O espaço de ações "qual subconjunto de páginas ler" é combinatório
(`C(P, n)`), então é fatorado: o livro é dividido em **K segmentos contíguos**,
que são os braços do bandit.

| Elemento | Definição |
|---|---|
| Política | `π = softmax(θ / τ)`, com `θ ∈ R^K` |
| Ação | `n` páginas, sorteando o segmento por `π` e a página uniformemente dentro dele, sem reposição |
| Recompensa | qualidade da classificação daquela amostra (abaixo) |
| Atualização | REINFORCE com baseline, ou EXP3 |

O gradiente do log-verossimilhança de sortear `count_k` vezes o segmento *k* em
`n` sorteios categóricos é `count_k − n·π_k`, e é exatamente esse termo que o
passo de REINFORCE usa.

**Recompensa.** Por padrão (`consensus`), a recompensa é a concordância da
geração com a interpolação das gerações anteriores, `1 − JSD`, somada a um termo
de confiança (baixa entropia). Isso **não usa o rótulo externo**: o agente
aprende a preferir regiões *representativas* do livro e a evitar as que geram
vereditos destoantes — prefácios, notas de rodapé, apêndices.

Existe também o modo `supervised`, que usa o ground truth como recompensa. Ele
mede o **teto** do agente, mas não deve ser usado para reportar acurácia: treinar
contra o gabarito e depois avaliar contra ele mede o próprio gabarito.

### 2. Classificação (PLN)

Três backends, selecionáveis por `classifier.backend`:

- **`zeroshot`** (padrão) — `facebook/bart-large-mnli` via inferência textual.
  Cada bloco de texto é julgado contra a hipótese "este texto é `<gênero>`".
  Roda em CUDA com fp16. Sem treino supervisionado: mexer na taxonomia não exige
  reanotar nada.
- **`llm`** — Claude via API. Melhor qualidade, exige `ANTHROPIC_API_KEY` e tem
  custo por token.
- **`tfidf`** — léxico bilíngue + TF-IDF. Instantâneo, sem GPU, serve de
  baseline honesto para medir quanto o zero-shot realmente adiciona.

**Calibração.** Modelos NLI zero-shot não tratam os rótulos de forma equânime.
Medido neste projeto, texto totalmente neutro ("N/A", "This is a text.") já
recebia `Drama = 0.176` contra `Terror = 0.042` — um viés de **4,2×** que fazia
*Drácula* e *Frankenstein* saírem como "Drama". A saída é dividida por
`prior ** alpha` (calibração contextual, Zhao et al. 2021). Ver
[Limitações](#limitações-conhecidas) sobre a escolha de `alpha`.

### 3. Interpolação entre gerações

`aggregation.method` controla como as N classificações viram uma só:

- `weighted_mean` (padrão) — média ponderada pela confiança de cada geração;
- `mean` — média simples (baseline);
- `log_pool` — pooling logarítmico; mais rigoroso, um gênero só sobrevive se
  nenhuma geração o descartou.

A **convergência** é medida pela JSD entre agregações consecutivas. Quando ela
fica abaixo de `jsd_threshold` por `patience` gerações seguidas, o loop para —
gerações adicionais deixaram de mudar o veredito.

### 4. Ground truth

Consulta, em ordem, a API do **Google Books** (campo `categories`, padrão BISAC)
e o **Open Library** (campo `subject`). Nenhuma exige autenticação, o que mantém
o experimento reproduzível.

O casamento título/autor usa similaridade fuzzy (título pesa 70%, porque nomes
de autor variam muito entre catálogos), com limiar de 72/100.

**Consolidação entre edições.** Confiar num único registro é frágil: catálogos
misturam edições, traduções e adaptações. *Frankenstein* casava primeiro com uma
adaptação teatral e saía como "Drama". O sistema agrega os assuntos de **todas**
as edições acima do limiar, ponderando pela qualidade do casamento — o que
corrigiu *Frankenstein* para Terror (0.42) e elevou *Drácula* para 0.66.

---

## Parâmetros principais

Tudo em `config.yaml`; os mais usados têm flag equivalente na CLI.

| Chave | Padrão | O que faz |
|---|---|---|
| `sampling.min_rate` / `max_rate` | 0.15 / 0.30 | faixa da fração de páginas por geração |
| `sampling.segments` | 10 | número de braços do bandit |
| `generations.n` | 12 | N gerações |
| `generations.early_stop.*` | ativo | parada por convergência |
| `agent.strategy` | `bandit` | `uniform` desliga o aprendizado (baseline) |
| `agent.algorithm` | `reinforce` | ou `exp3` |
| `agent.reward.mode` | `auto` | `consensus` ou `supervised` |
| `classifier.backend` | `zeroshot` | ou `llm`, `tfidf` |
| `aggregation.method` | `weighted_mean` | ou `mean`, `log_pool` |

## Testes

```powershell
.venv\Scripts\python.exe -m pytest -q
```

53 testes cobrindo taxonomia, parsing de PDF, chunking, métricas de agregação,
dinâmica do bandit, função de recompensa e consolidação do ground truth. Não
exigem rede nem GPU.

---

## Experimentos sugeridos para o relatório

O projeto foi estruturado para que estas comparações sejam uma flag de CLI:

1. **O RL ajuda?** `--strategy bandit` vs `--strategy uniform`, mesma semente.
   Compare a curva de recompensa e a geração em que cada um converge.
2. **Quantas páginas bastam?** Varie `--fixed-rate` de 0.05 a 1.0 e veja onde a
   acurácia satura — o argumento central a favor da amostragem parcial.
3. **Quantas gerações bastam?** A curva de convergência em
   `plots/convergencia.png` responde diretamente.
4. **Qual backend?** `--backend zeroshot` vs `tfidf` vs `llm`, mesmo dataset.
5. **Qual interpolação?** `--aggregation weighted_mean` vs `mean` vs `log_pool`.
6. **O que o agente aprendeu?** `plots/politica_*.png` mostra a política final
   por segmento contra a uniforme.

## Resultados medidos (dataset de exemplo, 8 livros)

| Backend | Top-1 | Top-3 | Concordância média |
|---|--:|--:|--:|
| `zeroshot` (bart-large-mnli, α=0.6) | 25% | 50% | 0.53 |
| `tfidf` (léxico, sem GPU) | **37,5%** | 75% | **0.62** |

Sim: o baseline léxico bate o modelo de 400M de parâmetros neste dataset. Não é
erro de implementação — é o resultado, e vale discuti-lo no relatório. Ver
abaixo.

## Limitações conhecidas

Documentadas de propósito — são material honesto para a seção de discussão.

- **O `bart-large-mnli` tem baixo poder discriminativo nesta tarefa.** O top-1
  fica preso em 2/8 em *todas* as configurações testadas: `alpha` de 0 a 1,
  hipóteses longas ou rótulos curtos. O padrão de falha é **colapso de modo** —
  um único gênero vence quase todos os livros, e a calibração só troca *qual*
  gênero é esse (α=0 → "Drama"; α=0.5 → "Tragédia"/"Sátira"). Diagnóstico
  reproduzível em `scripts/tune_calibration.py`.
- **Rótulos curtos não ajudam.** Testado: `"horror"`, `"romance"` etc. contra as
  hipóteses longas atuais. Os curtos ficaram iguais ou piores (1–2/8 vs 2/8),
  então as hipóteses descritivas foram mantidas.
- **Para melhorar a acurácia, o caminho mais provável é o backend `llm`**, ou um
  modelo NLI maior/multilíngue. A arquitetura já é plugável para isso.
- **O ground truth é ruidoso.** *Dom Casmurro* sai como "Histórico" no Open
  Library. A cobertura de literatura brasileira é bem pior que a anglófona, e o
  vocabulário de `subject` é inconsistente. Uma lista curada de rótulos para o
  seu dataset seria mais confiável que a API para fins de avaliação.
- **O Google Books limita por IP.** Sem chave de API, retorna HTTP 429 já na
  primeira consulta em rede compartilhada; na prática o Open Library é quem
  responde. Preencha `ground_truth.google_books_api_key` para melhorar isso.
- **Classificação multi-rótulo é intrinsecamente ambígua.** Clássicos são
  legitimamente multi-gênero — *Frankenstein* é terror, ficção científica e
  filosófico ao mesmo tempo. Por isso o pipeline reporta `top3_correct` e
  `predicted_in_gt_support` além do top-1 estrito; o top-1 sozinho subestima o
  desempenho.
- **`bart-large-mnli` é um modelo em inglês.** Livros em português são
  classificados por um modelo que não foi treinado na língua. Para um dataset
  majoritariamente em PT, vale testar um modelo multilíngue como
  `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` em
  `classifier.zeroshot.model`.
- **O ganho do RL é modesto neste dataset.** Com 120 páginas e 10 segmentos,
  uma amostra de 15–30% cobre todos os segmentos, então as gerações ficam
  parecidas demais e a recompensa de consenso satura — foi preciso padronizar a
  vantagem (`agent.normalize_advantage`) para a política sair da uniforme, e
  mesmo assim ela varia só de 0.093 a 0.107. O efeito deve crescer com livros
  mais longos, mais segmentos, taxas menores e mais gerações; a parada
  antecipada em ~7 gerações também limita o aprendizado. Vale rodar com
  `early_stop.enabled: false` ao estudar o comportamento do agente.

## Estrutura

```
src/librarian_ai/
  taxonomy.py       14 gêneros + mapeamento das categorias externas
  config.py         carga e validação do YAML
  corpus.py         PDF -> páginas de texto, com cache
  agent.py          bandit REINFORCE/EXP3 sobre segmentos do livro
  aggregation.py    interpolação entre gerações, JSD, convergência
  ground_truth.py   Google Books + Open Library
  pipeline.py       orquestração das gerações e avaliação
  report.py         JSON, CSV, gráficos, resumo em Markdown
  cli.py            interface de linha de comando
  classifiers/      zeroshot (CUDA) | llm (Claude) | tfidf (léxico)
scripts/
  fetch_samples.py     monta o dataset de exemplo
  tune_calibration.py  varredura da força de calibração
```
