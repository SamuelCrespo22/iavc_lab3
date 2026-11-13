## Tópicos Técnicos

#### Deteção facial moderna e alinhamento

Devemos usar um detetor pré-treinado.

O RetinaFace é um bom modelo de referência; o SCRFD tem uma família de detetores com diferentes custos computacionais.

Devemos entender a regressão da bounding-box, scores de confiança, landmarks faciais, supressão non-maximum e o papel do alinhamento no reconhecimento.

- Comparar pelo menos um detetor facial em accuracy, tempo de inferência e robustês a caras pequenas, pose, oclusão parcial e múltiplas faces.
- Usar as landmarks faciais para alinhar crops quando requeridos pelo modelo de reconhecimento ou atributo.
- Usar o WIDER face dataset.

---

#### Estimar Age e Gender-label

Estimar Age é apenas uma estimativa da idade aparente.

O módulo Gender deve prever apenas as categorias apresentadas no dataset de treino. O output deve ser descrito como uma classificação visual ou 

