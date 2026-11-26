# Produtos de dados e arquivo científico

Os três produtos foram publicados no Zenodo em **26/11/2026**, na versão **1.0.0**, com acesso público e DOIs resolvendo para os registros. A inclusão na comunidade [Big Data Analytics Research Group of Escola Politécnica da Universidade de São Paulo](https://zenodo.org/communities/bigdatagroup/) foi solicitada depois da publicação; os pedidos estavam pendentes de curadoria na conferência de 26/11/2026. Os registros já estão disponíveis independentemente dessa decisão.

| Produto | DOI publicado | Arquivo principal |
|---|---|---|
| Integrated Wind Curtailment and Meteorological Data for Bahia, Brazil (2022–2026) | [10.5281/zenodo.24549469](https://doi.org/10.5281/zenodo.24549469) | `pcs5787_observational_v1.0.0.zip` |
| Reproducible Machine Learning Experiments for Wind Curtailment Prediction in Bahia, Brazil | [10.5281/zenodo.24549473](https://doi.org/10.5281/zenodo.24549473) | `pcs5787_experiments_v1.0.0.zip` |
| Software e notebook aprovado | [10.5281/zenodo.24549475](https://doi.org/10.5281/zenodo.24549475) | código v1.0.0 e ZIP aprovado |

## Obter e conferir

A partir da raiz do clone, sem autenticação Zenodo:

```bash
.venv/bin/python -m pcs5787.obtain_data --destination data
.venv/bin/python -m pcs5787.release_tools validate --data-root data
```

O comando consulta somente registros públicos, valida o SHA-256 e o tamanho fixados em `configs/publication/products.json`, recusa caminhos inseguros e verifica todos os manifestos internos. Um rascunho ainda não publicado produz erro explícito; não utiliza credenciais ou finge disponibilidade pública.

Para arquivos já baixados ou disponibilizados localmente:

```bash
.venv/bin/python -m pcs5787.obtain_data --archives /caminho/dos/zips --destination data
```

A extração cria `data/observational` e `data/experiments`, sem sobrescrever uma pasta existente. Cada produto inclui README, dicionários, proveniência, avisos e manifesto. As duas bases são complementares; a projeção regional não equivale campo a campo à V2 integral de 177 colunas. O software evita duplicar dados no histórico Git.

O notebook autossuficiente está nos ativos da [release v1.0.0](https://github.com/mtierro/pcs5787-wind-curtailment-bahia/releases/tag/v1.0.0) e é incluído explicitamente no único registro Zenodo de software. Não foi ativada integração automática GitHub–Zenodo para duplicar a versão. Os datasets e o software referenciam os demais por seus identificadores reais.
