# Preparação da versão pública

Acervo científico de referência: commit `ee6bcf36dbf5f27a3f304a1267a323feaecf5878`. O histórico desse repositório não foi importado para a publicação. O manifesto `manifests/source_code.json` identifica cada arquivo de código/configuração/resultado copiado sem alteração e seu SHA-256. As interfaces `release_tools.py`, `rebuild_h2_features.py` e `retrain_h2.py` são adaptações portáteis desta distribuição; não substituem os métodos históricos.

O Dataset 1 projeta 31 campos observacionais da V2 normativa de 177 colunas, conserva todas as linhas/nulos/tipos e relaciona energia às dimensões históricas. O Dataset 2 projeta as categorias originais e 51 valores meteorológicos; não faz codificação global, imputação ou reestimativa da seleção. As duas projeções foram comparadas campo a campo com a fonte. Os 164 arquivos de origem utilizados tiveram seus hashes conferidos.

Os binários H2, configurações congeladas e contribuições SHAP foram copiados byte a byte. Memberships que antes dependiam de `row_id` receberam chaves explícitas por junção validada, sem perda de linhas. As previsões H1/H3/H4 foram filtradas pelos protocolos e populações do notebook; as razões H3 compartilham a interseção original de chaves. Os resultados de intensidade permanecem apenas nos auxiliares do notebook aprovado.

Os arquivos comprimidos usam caminhos relativos, recusam travessia de diretórios e links simbólicos. A validação foi executada depois da extração independente. O notebook foi extraído diretamente do ZIP aprovado, sem modificação das fontes; sua cópia executada é apenas evidência de verificação e não substitui o original distribuído.

## Evidências verificáveis

* `manifests/verification.json`: ambiente, 124 contratos científicos, 24 modelos, erro máximo, notebook e hashes dos arquivos comprimidos.
* `requirements/verified-lock.txt`: dependências efetivamente resolvidas em ambiente virtual novo.
* `tests/test_release.py`: testes positivos e negativos de integridade, reconstrução dos 51 atributos, alinhamento SHAP e verificação dos sorteios bootstrap históricos.
* `MANIFEST.sha256` dentro de cada dataset: todos os arquivos daquele produto, exceto o próprio manifesto.

Nenhum novo modelo foi treinado ou selecionado. O dry-run conferiu as linhas e limites dos 24 ajustes possíveis. A instalação isolada, inferência, recálculo das métricas e execução do notebook foram verificadas; a reconstrução completa das campanhas originais desde os brutos nacionais não é apresentada como um fluxo certificado desta distribuição.

Arquivos de acervo não selecionados, caches, ambientes, versões antigas do artigo e milhares de modelos permaneceram no repositório científico privado. Avisos de terceiros e direitos de banco estão em `NOTICE.md`; a licença MIT do código não se aplica indistintamente aos dados.
