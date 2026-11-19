## Tópicos Técnicos

#### Deteção facial moderna e alinhamento

Devemos usar um detetor pré-treinado, podendo fazer fine-tuning dele.

O RetinaFace é um bom modelo de referência; o SCRFD tem uma família de detetores com diferentes custos computacionais.

Devemos entender a regressão da bounding-box, scores de confiança, landmarks faciais, non-maximum suppression e o papel do alinhamento no reconhecimento.

- Comparar pelo menos um detetor facial em accuracy, tempo de inferência e robustez a caras pequenas, pose, oclusão parcial e múltiplas faces.
- Usar as landmarks faciais para alinhar crops quando requerido pelo modelo de reconhecimento ou de atributos.
- Verificar se a licença do modelo/pesos permite uso académico e redistribuição.
- Usar o WIDER Face dataset na avaliação de desempenho.

---

#### Estimar Age e Gender-label

A idade é apenas uma estimativa da idade aparente, não a idade cronológica verificada.

O módulo Gender deve prever apenas as categorias presentes no dataset de treino (normalmente binário M/F). O output deve ser descrito como uma classificação visual definida pelo dataset (apresentação de género aparente), e não como a identidade de género da pessoa.

Temos de discutir as limitações das anotações, a incerteza, as diferenças de desempenho entre grupos demográficos e os riscos de inferir atributos sensíveis a partir da aparência.

O módulo pode ser multi-task, usando um modelo pré-treinado ou um classificador leve com fine-tune no UTKFace, IMDB-WIKI ou outro dataset com licença adequada.

- Estudar regressão, classificação por grupos etários e, como extensão, estimação ordinal ou baseada em distribuições.
- Reportar MAE (opcionalmente RMSE) e desempenho por grupo etário.
- Mostrar uma estimativa arredondada ou um intervalo de idade.
- Para o género, reportar precision, recall e F1 por classe e a matriz de confusão (não só accuracy).
- Outputs com baixa confiança devem ser omitidos ou marcados como incertos.

---

#### Reconhecimento de expressões faciais

Classificar padrões de expressão visíveis (neutral, happy, sad, angry, fear, surprise, disgust), conforme o modelo/dataset.

Distinguir a classificação da expressão da inferência do estado emocional: uma label prevista não prova que a pessoa sente essa emoção.

- Usar um modelo de expressão pré-treinado ou fazer fine-tune de uma CNN/Transformer compacta com o BTFER ou o AffectNet.
- Mostrar a expressão mais provável e, opcionalmente, as probabilidades top-k.
- Avaliação:
  1. Intra-dataset: treinar no AffectNet (ou BTFER) e avaliar na partição oficial de validação/teste, sem imagens partilhadas entre treino e avaliação.
  2. Cross-dataset: avaliar num dataset independente sem retreino nem fine-tune, considerando só as classes comuns aos dois.
  3. Métricas: accuracy, balanced accuracy, precision macro, recall macro e F1 macro, matriz de confusão (erros mais frequentes) e discussão das limitações e da generalização.

As labels devem ser apresentadas como estimativas da configuração facial visível. O relatório deve discutir as limitações de mapear aparência facial para emoções.

---

#### Identificação facial closed-set (< 20 identidades)

Extensão viável e valorizada. É um problema de identificação 1:N com uma galeria pequena: dado um crop, determinar se corresponde a alguém registado ou a ninguém.

Não treinar uma rede de reconhecimento de raiz; usar um modelo pré-treinado para extrair embeddings (ou o embedding do módulo multi-task age-gender).

- Criar um procedimento de registo para voluntários que consintam (< 20), com várias imagens por pessoa (expressões, poses e iluminação diferentes).
- Extrair embeddings normalizados com um modelo pré-treinado (ex.: encoder baseado em ArcFace).
- Representar cada identidade com um ou mais templates (ex.: média de vários embeddings).
- Comparar com os templates da galeria por similaridade de cosseno.
- Aplicar um threshold definido em validação: abaixo dele → "Unknown".
- Avaliar o efeito do número/diversidade de imagens de registo, tamanho da cara, pose, iluminação e oclusão.

Com uma galeria pequena, a pesquisa é barata. O difícil é escolher um threshold que limite os falsos matches sem perder reconhecimento. Esse threshold tem de ser calibrado num conjunto de validação separado, nunca depois de ver os resultados de teste.

A galeria tem de ser baseada em consentimento explícito: guardar só os dados necessários, protegê-los localmente, ter mecanismo de eliminação e não identificar pessoas fora da galeria. Não usar para controlo de acessos, presenças, monitorização ou outras decisões com consequências.
