# Reproducible Machine Learning Experiments for Wind Curtailment Prediction in Bahia, Brazil

Versão 1.0.0 — Matheus T. de Paula; Pedro A. M. A. Filho; Robert Caetano. Escola Politécnica da Universidade de São Paulo.

Produto experimental associado ao Dataset observacional e ao [software](https://github.com/mtierro/pcs5787-wind-curtailment-bahia). Contém entradas, previsões, protocolos, resultados positivos/negativos e os 24 modelos finais H2. Não inclui V4, todos os caches de desenvolvimento ou novos experimentos.

| Componente | Conteúdo |
|---|---|
| features_h2.parquet | 332.928 × 58: duas chaves, cinco categorias adicionais e 51 valores meteorológicos; `region` é também a sexta categoria |
| protocols/ | relógio, representação, campanhas e parâmetros originais |
| dictionaries/ | nomes, unidades, operações, disponibilidade, famílias e ordem das entradas |
| frozen/ | seleção fixa, manifesto e evidências anteriores à avaliação; arquivos originais |
| selection/ | 15 origens jan/2025–mar/2026, rankings, ablações, previsões, intervalos e memberships |
| evaluation/ | seis origens abr–set/2026, 139.244 previsões (4 × 34.811), limiares e métricas |
| importance/ | SHAP, permutação e estabilidade; desenvolvimento e avaliação separados |
| models/ | 24 binários e model_registry.json (hashes, origens, entradas, ambiente, parâmetros) |
| predictions/ | previsões selecionadas H1/H3/H4 e previsões históricas de desenvolvimento F1–F3 |
| historical_reference/ | tabelas do notebook e registros históricos de calibração/composição |

## Representações congeladas

| Sistema | Meteorologia | Entradas efetivas | AP | Brier |
|---|---:|---:|---:|---:|
| A | 0 | 6 | 0,887727 | 0,087178 |
| B | 51 | 108 | 0,899426 | 0,079880 |
| C_top5 | 5 | 16 | 0,902156 | 0,079388 |
| C_spatial | 14 | 34 | 0,900128 | 0,079164 |

Todas utilizam os mesmos 34.811 casos, 11.521 positivos e oito regiões. A população depende do rótulo ENE >2% conhecido e de `sp_wind` disponível, inclusive quando não utilizado pelo modelo. C_top5, nesta ordem: `wind_mean72_partial`, `sp_gust`, `sp_temperature`, `sp_ramp6`, `sp_mean24`. C_spatial: lista exata em `frozen/selection.json`. Nenhuma lista varia durante a avaliação.

As seis categorias são `region, hour, weekday, month, region_hour, region_month`. Mês histórico começa em zero. Codificação ordinal ajustada por origem, categorias desconhecidas → NaN. Não corrigir `month`/`region_month` desconhecidas na avaliação. Cada variável meteorológica conserva valor nativo (NaN admitido) e indicador `missing::<nome> = ~isfinite(valor)`. Os indicadores são gerados pelo pipeline, sem duplicar 51 colunas no Parquet. Ordem: seis categorias, k valores, k indicadores.

LightGBM 4.7.0; 7 folhas, mínimo 200 por folha, 180 iterações, taxa 0,05, seed 5787; L2=40 em A e 10 nos demais. Demais parâmetros, paralelismo e determinismo constam no registro dos modelos. Janela de três meses de calendário; último alvo de treino O−170h; meteorologia até t−2h. Nenhuma nova seleção ou campanha foi executada para publicar estes arquivos.

## Chaves, partições e reprodução

Chave observacional: `(region,time_local)`. Previsões exigem também hipótese/campanha, origem, sistema/variante e população. `row_id` identifica apenas uma ordem local; não é chave universal. Memberships de seleção/avaliação acrescentam chaves explícitas, campanha e posição original. As 103.624 participações nos treinos finais se sobrepõem entre origens e não representam observações independentes. A validação histórica H2 F1–F3 não é a nova validação mensal de 15 origens.

`predictions/hypotheses.parquet` preserva populações distintas: H1=34.811 por sistema; H3=34.484 por motivo/sistema; H4=15.584 por política. Não combinar essas populações como se fossem uma comparação pareada única. H3 utiliza a campanha histórica e não herda automaticamente as métricas H2. Os pequenos ledgers históricos registram também contextos de calibração; use os campos de protocolo/população, não agregação indiscriminada.

`importance/*/shap_contributions.npz`, chave `phi`, contém float64 (avaliação 34.811×109; desenvolvimento 85.185×109). Linhas correspondem **na mesma ordem** a `observations.parquet`; as primeiras 108 colunas seguem `dictionaries/effective_columns.csv`, a última é valor-base. A soma é margem em log-odds; sigmoid da soma reproduz a probabilidade ENE. Valor + indicador formam a variável lógica. Média de |soma| difere da soma das médias de |contribuições|. Os modelos V3 de desenvolvimento preservados reproduzem probabilidades V2, sem prova de identidade dos SHAP dos binários V2 ausentes.

Permutação usa blocos 24/168h, 20 repetições, valores e indicadores juntos, populações fixas. As tabelas guardam degradações AP/Brier, inclusive negativas; não são o contraste B−A. Correlação e combinações perturbadas limitam interpretações físicas.

A política top-5 validada no desenvolvimento foi **adaptativa**; a lista final fixa não foi validada diretamente nas 15 origens. O semestre final já havia sido examinado: avaliação retrospectiva adicional. Ambas as reduções satisfazem a margem operacional de não inferioridade 0,01 AP, sem superioridade robusta. Bootstrap pareado sincronizado: 2.000 réplicas, blocos 168/672h, sementes 5955/6459; limites simultâneos para os dois contrastes reduzidos. A sensibilidade 672h tem só sete blocos. Incerteza é condicional aos modelos e não cobre todo o processo de seleção. Limiares iniciais de alerta diferem entre A/B e reduzidos; ganhos F1/recall não são atribuíveis exclusivamente à redução.

## Uso

Extraia lado a lado `data/observational/` e `data/experiments/`. Instale o software associado e execute:

```bash
python -m pcs5787.release_tools validate --data-root data
python -m pcs5787.release_tools metrics --data-root data --output metrics.csv
```

O primeiro comando verifica todos os hashes e os 24 modelos contra probabilidades históricas (tolerância 1e−12 no ambiente registrado); não treina. Carregue apenas modelos de origem confiável após verificar SHA-256: joblib/pickle pode executar código. `FILES.csv` e `dictionaries/table_schemas.json` descrevem os arquivos tabulares. Fonte de cada arquivo: `provenance/source_manifest.csv`. Condições de uso: [NOTICE.md](../NOTICE.md).
