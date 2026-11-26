# Níveis de reprodução

1. **Notebook autossuficiente:** ZIP aprovado byte a byte; conferir manifesto, executar 102 células em kernel novo e 44 testes. Recalcula demonstrações, métricas e figuras usando insumos distribuídos. Não requer bases externas nem modelos LightGBM. HTML incorpora figuras, mas a tipografia MathJax pode precisar de rede/cache no navegador.
2. **Resultados e inferência:** Dataset 1 + Dataset 2 + interfaces `release_tools`. Recalcula métricas H1–H4, valida relações/rótulos, SHAP persistido e os 24 modelos H2. Não refaz seleção ou bootstrap por padrão. Protocolos e estatísticas bootstrap são distribuídos.
3. **Reconstrução de atributos:** teste `test_features_reconstructed_from_observations` recompõe os 51 atributos a partir das observações e pesos publicados. Máscaras de nulos exatas; tolerância 1e−10 para pequenas diferenças de ponto flutuante em operações agregadas.
4. **Reajuste H2 explícito:** receita abaixo usa as linhas de treino publicadas, ordem histórica e configuração congelada. Foi auditada e submetida a dry-run; nenhuma nova campanha foi executada na preparação da publicação. Reutiliza limiares registrados para comparar probabilidades; não promete reconstruir automaticamente todo o histórico de otimização dos limiares.

```bash
python -m pcs5787.retrain_h2 --data-root data --dry-run
# Somente quando o pesquisador decidir reajustar os 24 modelos:
python -m pcs5787.retrain_h2 --data-root data --output run_outputs/retrained_h2
```

`--output` deve ser novo. O treino pode variar numericamente em outro ambiente; compare probabilidades/chaves/métricas antes de interpretar explicações. Categorias desconhecidas e NaN históricos não devem ser corrigidos.

Os módulos históricos `v2`, `v3` e scripts `experiments/` conservam o código científico e os caminhos relativos do acervo original. Não são todos executáveis diretamente na árvore pública: alguns leem caches, V2 integral de 177 colunas ou raízes `reports/`. Consulte seus arquivos de proveniência. A reprodução central portátil usa as interfaces acima. Aquisição nacional e todas as campanhas históricas não são um fluxo integral certificado por esta distribuição. Não baixar portais atuais como substituição silenciosa do snapshot.

A instalação de dependências pode usar rede. Após obtidos os arquivos e instaladas as dependências, inferência, métricas e notebook executam offline. Não carregar modelos joblib de origem não confiável; confira manifestos primeiro.
