# Publicação científica verificada — v1.0.0

Os três registros foram publicados em **26/11/2026**, com acesso público. Os DOIs resolveram para as páginas corretas (HTTP 200). A publicação foi realizada nos mesmos registros e com os mesmos arquivos, hashes e metadados aprovados.

| Produto | DOI e registro público | Arquivos | Bytes |
|---|---|---:|---:|
| Dataset observacional | [10.5281/zenodo.24549469](https://doi.org/10.5281/zenodo.24549469) · [registro](https://zenodo.org/records/24549469) | 5 | 25701192 |
| Dataset experimental | [10.5281/zenodo.24549473](https://doi.org/10.5281/zenodo.24549473) · [registro](https://zenodo.org/records/24549473) | 6 | 91018226 |
| Software e notebook | [10.5281/zenodo.24549475](https://doi.org/10.5281/zenodo.24549475) · [registro](https://zenodo.org/records/24549475) | 6 | 113878740 |

Os dois datasets usam ODbL 1.0 para a organização das bases derivadas e CC BY 4.0 para documentação original. O registro de software/notebook também declara MIT para o código próprio. Os registros explicitam que essas licenças correspondem a componentes distintos; os avisos e créditos de ONS, INMET, SIGA/ANEEL e dependências permanecem aplicáveis.

## Comunidade

Os pedidos de inclusão na [Big Data Analytics Research Group of Escola Politécnica da Universidade de São Paulo](https://zenodo.org/communities/bigdatagroup/) foram enviados **depois** da publicação. Na conferência de 26/11/2026, os três pedidos estavam em `submitted`, pendentes de curadoria; isso não impede acesso aos registros e não equivale a aceitação na comunidade.

| Registro | Pedido de inclusão | Estado conferido |
|---|---|---|
| 24549469 | `06252eda-a1b5-4a80-bb31-f5711ba990fc` | `submitted` |
| 24549473 | `e64e8395-1be7-4a80-91aa-b066d5da574d` | `submitted` |
| 24549475 | `cc09fe94-6cba-4939-ae58-a8f14539649d` | `submitted` |

## Identidade do software e notebook

* [Repositório GitHub](https://github.com/mtierro/pcs5787-wind-curtailment-bahia); [release v1.0.0](https://github.com/mtierro/pcs5787-wind-curtailment-bahia/releases/tag/v1.0.0).
* Commit científico arquivado: `633b50dfdbcd900c6c3481ad96517bc1b1e44b94`. Os 94 arquivos rastreados de uma clonagem pública da tag coincidiram byte a byte com o ZIP de código depositado no Zenodo.
* O notebook no Git corresponde ao membro do ZIP aprovado. O ZIP autossuficiente aprovado foi preservado integralmente tanto na release quanto no Zenodo.
* Os quatro anexos GitHub foram baixados sem autenticação e seus SHA-256 conferidos; código e notebook são os mesmos do registro Zenodo. Nenhuma tag, arquivo anexado ou arquivo científico foi substituído.
* A documentação em `main` foi atualizada para informar a publicação. Isso não altera o conteúdo da tag nem do arquivo de código da versão 1.0.0. O registro Zenodo de software é único; não há webhook de integração automática duplicando a release.

## Verificação pública independente

Os **17 arquivos Zenodo, totalizando 230.598.158 bytes**, foram baixados sem autenticação e todos os hashes coincidiram com a conferência anterior à publicação. Além disso, uma clonagem independente da tag obteve os datasets usando `pcs5787.obtain_data` pela interface pública, conferiu seus manifestos e passou nos **11 testes existentes**, incluindo reconstrução dos 51 atributos, alinhamento SHAP, contratos científicos e inferência dos 24 modelos H2. Não houve retreinamento ou novos experimentos.

Ambiente: Python 3.11.17; NumPy 2.4.6; pandas 3.0.6; scikit-learn 1.9.1; LightGBM 4.7.0; PyArrow 26.0.0. Aquisição e extração: 123,716 s; processo dos testes: 17,911 s; recálculo de métricas: 0,864 s; pico observado do ensaio: aproximadamente 1,68 GiB.

Métricas H2 recalculadas das previsões publicadas, com 34.811 observações e 11.521 positivos para cada sistema:

| Sistema | AP | Brier |
|---|---:|---:|
| A | 0.887727 | 0.087178 |
| B | 0.899426 | 0.079880 |
| C_top5 | 0.902156 | 0.079388 |
| C_spatial | 0.900128 | 0.079164 |

A execução integral de 102 células e 44 testes do notebook foi validada antes da publicação. Nesta etapa sua identidade foi novamente confirmada por hash; não se apresenta essa conferência como uma nova execução integral do notebook. A auditoria final dos 94.127 arquivos históricos (30.484.323.988 bytes) não encontrou alterações.

## SHA-256 dos arquivos Zenodo

| Produto | Arquivo | Bytes | SHA-256 |
|---|---|---:|---|
| Dataset observacional | `README.md` | 3866 | `d2158a66584802535e358d795df1fa8486f920fcc20b0f7b403e33e8a994ad0b` |
| Dataset observacional | `NOTICE.md` | 2397 | `d1e0216e29f6e922cf6b5505d39929862cc68e9bccaab7e92781a7d2a5767a49` |
| Dataset observacional | `SHA256SUMS.txt` | 337 | `21144f69e2b6681f1f3e92b0e4274d5e753854918d8dd1920d94d4929033fcfc` |
| Dataset observacional | `data_dictionary.csv` | 54502 | `c79229f7838c86336462f0d1f90631d57a9d27c616376ea153c1097811d32679` |
| Dataset observacional | `pcs5787_observational_v1.0.0.zip` | 25640090 | `4d224cdfc578b571a865244b5d975956348a9d89dfa52304fe278b8e8e6d52ad` |
| Dataset experimental | `README.md` | 6452 | `8664913148fe4c61b22840504e4bfcd5564054d76aa115802f03487dddc8c2cf` |
| Dataset experimental | `NOTICE.md` | 2397 | `d1e0216e29f6e922cf6b5505d39929862cc68e9bccaab7e92781a7d2a5767a49` |
| Dataset experimental | `input_dictionary.csv` | 31116 | `5c0affb417b7da5be7535e57a8dfb54a532d9e99379162d9c70baae319ab44eb` |
| Dataset experimental | `feature_definitions.csv` | 21935 | `09c1889a17d43a1ede34f8d1b7d464322d887bf3623c5c589dcba51e3738db02` |
| Dataset experimental | `SHA256SUMS.txt` | 426 | `b5806740a4a219217ba5271991f6389db2512e51b4efa7dc37e1cc109800e038` |
| Dataset experimental | `pcs5787_experiments_v1.0.0.zip` | 90955900 | `f1d41fc83a9d2e86cadd0c4e0f9045b48f080bc70042d80afd8c58cc3f0b7354` |
| Software e notebook | `pcs5787_software_v1.0.0.zip` | 3265773 | `b60eac0213b0fb741fb97484ff6d348e947de67a0579986f1c8978a6f1a11d6f` |
| Software e notebook | `README.md` | 5728 | `83376102e7c1a1d9744c1f5bd349e91e0d70c5535f60598f2d12701d7bb850a9` |
| Software e notebook | `NOTICE.md` | 2397 | `d1e0216e29f6e922cf6b5505d39929862cc68e9bccaab7e92781a7d2a5767a49` |
| Software e notebook | `SOURCE_MANIFEST.json` | 967 | `fd77389cb224ddd70947917766922f8bc966e8a266d46e84922e32c52c63ea3f` |
| Software e notebook | `SHA256SUMS.txt` | 454 | `8e1f1d516c274a9d02ab4115471d9185ef01de778ee69b0f4406cb7a65c83331` |
| Software e notebook | `PCS5787_Notebook_H2_Resultados_Reforcados_20261125.zip` | 110603421 | `3d628576f77e1f39d7e12ac969d12091952d9de46d33363de2ffcbba518809e8` |

## Reproduzir e citar

Use a tag `v1.0.0` para a versão exata arquivada. Com as dependências instaladas conforme o README, os comandos abaixo acessam somente dados públicos e não treinam modelos:

```bash
python -m pcs5787.obtain_data --destination data
PCS5787_DATA=data python -B -m unittest discover -s tests -v
python -m pcs5787.release_tools metrics --data-root data --output metrics.csv
```

Para o notebook, obtenha o ZIP autossuficiente na release ou no registro de software, confira o checksum e siga seu README interno. Para retreinamento opcional, consulte [REPRODUCTION.md](REPRODUCTION.md); ele não é necessário para recalcular as métricas publicadas.

Cite o software e os datasets efetivamente utilizados. Autores de cada produto: **Matheus T. de Paula; Pedro A. M. A. Filho; Robert Caetano**. Afiliação: Escola Politécnica da Universidade de São Paulo. Referências recomendadas:

* Matheus T. de Paula; Pedro A. M. A. Filho; Robert Caetano. (2026). *Integrated Wind Curtailment and Meteorological Data for Bahia, Brazil (2022–2026)* (Version 1.0.0) [Dataset]. Zenodo. [https://doi.org/10.5281/zenodo.24549469](https://doi.org/10.5281/zenodo.24549469).
* Matheus T. de Paula; Pedro A. M. A. Filho; Robert Caetano. (2026). *Reproducible Machine Learning Experiments for Wind Curtailment Prediction in Bahia, Brazil* (Version 1.0.0) [Dataset]. Zenodo. [https://doi.org/10.5281/zenodo.24549473](https://doi.org/10.5281/zenodo.24549473).
* Matheus T. de Paula; Pedro A. M. A. Filho; Robert Caetano. (2026). *Wind Curtailment Prediction in Bahia: Reproducible Software and Approved Scientific Notebook* (Version 1.0.0) [Software]. Zenodo. [https://doi.org/10.5281/zenodo.24549475](https://doi.org/10.5281/zenodo.24549475).

Use [CITATION.cff](../CITATION.cff) para o software. Os registros preservam relações explícitas entre produtos: o Dataset 2 requer o Dataset 1, e os produtos se relacionam como suplementos. O crédito aos autores dos produtos não substitui a atribuição às três instituições de origem. Não há DOI de artigo associado nesta entrega.
