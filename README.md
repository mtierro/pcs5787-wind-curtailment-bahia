# Wind curtailment prediction in Bahia — PCS5787

**Dados integrados, validação temporal e representações meteorológicas compactas.** Projeto do Grupo 9, PCS5787 — Ciência dos Dados e Big Data, Escola Politécnica da Universidade de São Paulo.

Matheus T. de Paula; Pedro A. M. A. Filho; Robert Caetano.

**English summary.** Reproducible research on wind curtailment in Bahia, Brazil, combining ONS operational records, INMET weather and SIGA/ANEEL geography (2022–2026). This release provides verified historical predictions, 24 final H2 models, frozen feature subsets and the approved self-contained teaching notebook. Results are retrospective; importance is not causal evidence.

[Reprodução](docs/REPRODUCTION.md) · [Dados](docs/DATA.md) · [Protocolos](docs/PROTOCOLS.md) · [Fontes e direitos](NOTICE.md) · [Citação](CITATION.cff)

## Perguntas e resultados

H1: previsibilidade pelo calendário/região. H2: contribuição adicional da meteorologia. H3: diferenças entre ENE, CNF e REL. H4: atualização do treino diante da mudança temporal. O aprofundamento H2 usa SHAP, permutação, ablações e seleção temporal, preservando essas quatro hipóteses.

Integração região × hora (332.928 linhas, oito regiões analíticas), janela móvel de três meses, reajustes mensais, meteorologia até t−2h e latência assumida de 168h após o fim da hora para os rótulos. Avaliação H2 ENE >2%: abr–set/2026, 34.811 casos, 11.521 positivos.

| Modelo | Entradas | AP | Brier |
|---|---:|---:|---:|
| Calendário A | 6 | 0,887727 | 0,087178 |
| Meteorologia completa B | 108 | 0,899426 | 0,079880 |
| Cinco variáveis C_top5 | 16 | 0,902156 | 0,079388 |
| Multissensor C_spatial | 34 | 0,900128 | 0,079164 |

A redução de 51 para cinco variáveis (90,2%) preservou desempenho segundo a margem operacional AP=0,01. As duas reduções passaram pelo critério; não há evidência robusta de superioridade. O ganho AP meteorologia–calendário permanece inconclusivo, enquanto Brier melhora. SHAP multissensor=88,48% da soma das magnitudes meteorológicas, não do desempenho explicado. Limiares de alerta diferentes limitam comparações de F1/recall. Ver tabelas e intervalos em `results/h2/`.

## Começar

Python **3.11**, Linux. O ambiente científico registrado usa Python 3.11.17 e LightGBM 4.7.0.

```bash
git clone https://github.com/mtierro/pcs5787-wind-curtailment-bahia.git
cd pcs5787-wind-curtailment-bahia
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/reproduction.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps .
```

Obtenha os dois arquivos de dados conforme [docs/DATA.md](docs/DATA.md), confira checksums e extraia em `data/`. Então:

```bash
.venv/bin/python -m pcs5787.release_tools validate --data-root data
.venv/bin/python -m pcs5787.release_tools metrics --data-root data --output metrics.csv
PCS5787_DATA=data .venv/bin/python -B -m unittest discover -s tests -v
```

Os testes verificam chaves, nulos, rótulos, relógio, ordem, reconstrução dos 51 atributos, SHAP e inferência dos 24 modelos. Não retreinam. [Procedimento de treino H2 e seus limites](docs/REPRODUCTION.md).

## Notebook aprovado

`notebooks/PCS5787_Projeto_Final.ipynb` é cópia exata do membro do pacote aprovado, com 232 células. Para executar, obtenha **todo** o ZIP `PCS5787_Notebook_H2_Resultados_Reforcados_20261125.zip` na release `v1.0.0`; o arquivo isolado no Git não contém seus insumos. SHA-256:

`3d628576f77e1f39d7e12ac969d12091952d9de46d33363de2ffcbba518809e8`

Extraia em outra pasta e siga o README interno. Com o ambiente acima, na pasta `PCS5787_Projeto`:

```bash
python codigo/verificar_pacote.py
python -B codigo/executar.py --root . --progresso
python -B -m unittest discover -s codigo -p 'testar*.py' -v
```

São 102 células de código e 44 testes. O notebook usa dados/resultados persistidos para reconstruir tabelas, figuras e demonstrações; não retreina todas as campanhas. A seção didática sobre intensidade permanece, sem integrar o escopo central dos datasets experimentais.

## Organização

`src/pcs5787/`: métodos históricos e interfaces portáteis; `experiments/`: scripts originais das três campanhas H2; `configs/`: protocolos e seleção congelada; `tests/`: verificações da distribuição; `docs/`: reprodução e proveniência; `results/`: tabelas/figuras compactas; `manifests/`: hashes e identidade das fontes. Dados, modelos e ZIP não entram no histórico Git.

Os scripts históricos de campanhas são referência metodológica e exigem a organização original descrita na documentação. As interfaces portáteis acima são o fluxo testável com os dois datasets. Não se promete reconstrução automática do acervo nacional ou de todas as campanhas V1–V4.

## Recursos, limites e crédito

Inferência e validação cabem em estação Linux com 4 núcleos e memória recomendada de 8 GB; o notebook previamente mediu pico inferior a 3 GB. Não é necessária GPU. Reserve espaço para dados, ambiente e extrações independentes. Cobertura/representatividade meteorológica, snapshot geográfico retrospectivo, latências assumidas, mudança de distribuição e sete blocos na sensibilidade 672h limitam a generalização.

Código próprio MIT; documentação original CC BY 4.0; bases derivadas ODbL com avisos de fontes preservados. ONS, INMET e SIGA/ANEEL devem receber crédito. [CITATION.cff](CITATION.cff) identifica autores/versão; identificadores de arquivo científico constarão em [docs/DATA.md](docs/DATA.md) quando efetivamente reservados/publicados. Nenhum DOI fictício é utilizado.
