
# Global Solution – Modelagem Matemática e Computacional  
## Recrutamento guiado por Álgebra Linear
  
**Disciplina:** Modelagem Matemática e Computacional  
**Professor:** Daniel Couto Gatti
**Grupo:**   
  - Luiz Miguel Martin Crocco - RM: 562796
  - Rafael Louzã Lopes - RM: 564963

---

## 1. Introdução e contextualização

Processos de recrutamento muitas vezes se baseiam em impressões subjetivas,
o que pode gerar decisões inconsistentes, enviesadas e difíceis de justificar.
Em um cenário de alta competitividade por talentos, empresas buscam decisões
mais transparentes e orientadas por dados.

Este trabalho utiliza conceitos de Álgebra Linear para modelar o processo de
decisão de contratação a partir de um conjunto de dados de candidatos
(`recruitment_data.csv`). A ideia central é representar as características dos
candidatos em uma matriz \(X\) e a decisão de contratação em um vetor \(y\),
estimando um vetor de pesos \(w\) que indique a importância relativa de cada
variável no processo seletivo.

O projeto está alinhado à proposta da disciplina de usar ferramentas
matemáticas para apoiar decisões em contextos reais, conectando teoria e
prática de forma aplicada.

---

## 2. Metodologia

### 2.1 Base de dados utilizada

Foi utilizado o arquivo **`recruitment_data.csv`**, que contém registros de
candidatos a uma vaga. Cada linha representa um candidato e cada coluna
corresponde a uma variável relevante para o processo de seleção, incluindo:

- `Age` – idade do candidato;  
- `Gender` – gênero codificado numericamente;  
- `EducationLevel` – nível de formação;  
- `ExperienceYears` – anos de experiência profissional;  
- `PreviousCompanies` – número de empresas anteriores;  
- `DistanceFromCompany` – distância da residência até a empresa;  
- `InterviewScore` – desempenho na entrevista;  
- `SkillScore` – avaliação técnica;  
- `PersonalityScore` – avaliação comportamental;  
- `RecruitmentStrategy` – estratégia de recrutamento utilizada;  
- `HiringDecision` – variável-alvo (0 = não contratado, 1 = contratado).  

A matriz \(X\) foi formada por todas as colunas **exceto**
`HiringDecision`, que foi usada como vetor-alvo \(y\).

### 2.2 Formulação do modelo

O problema foi modelado como:

\[
    X w \approx y
\]

em que:

- \(X\) é a matriz de dimensão \(n \times m\) (n candidatos, m variáveis),  
- \(w\) é o vetor de pesos (\(m \times 1\)),  
- \(y\) é o vetor de decisões de contratação (\(n \times 1\)).  

O objetivo é encontrar \(w\) que minimize o erro quadrático entre
\(Xw\) e \(y\).

### 2.3 Métodos numéricos empregados

Foram usados dois métodos para estimar \(w\):

1. **Mínimos quadrados via `np.linalg.lstsq`**  
   Resolve o problema de mínimos quadrados direto, devolvendo \(w\),
   resíduos, posto da matriz e valores singulares.

2. **Pseudoinversa de Moore–Penrose via `np.linalg.pinv`**  
   Calcula a pseudoinversa \(X^{+}\) e obtém:  
   \[
       w = X^{+} y
   \]

Na prática, para este conjunto de dados, os dois métodos geram vetores
\(w\) muito semelhantes, o que reforça a consistência numérica da solução.

### 2.4 Construção do ranking e avaliação de novo candidato

Com o vetor \(w\) estimado, foi calculado para cada candidato um **score**:

\[
    \text{Score}_i = X_i w
\]

Esses scores foram ordenados em ordem decrescente para produzir um
**ranking de candidatos**. Em seguida:

- foram destacados os **5 candidatos com maiores scores**;  
- foi criado um **novo candidato fictício** com valores plausíveis para as
  variáveis;  
- calculou-se o score desse novo candidato e sua posição no ranking geral.  

Por fim, o ajuste global do modelo foi avaliado via **coeficiente de
determinação** (\(R^2\)), comparando \(y\) com \(\hat{y} = Xw\).

---

## 3. Resultados e análises

### 3.1 Vetor de pesos \(w\) e importância das variáveis

O script gera uma tabela ordenando as variáveis de acordo com a magnitude
de seus pesos \(|w|\). Em termos qualitativos, esperam-se pesos mais altos
associados a variáveis como:

- `InterviewScore` (desempenho em entrevista),  
- `SkillScore` (competências técnicas),  
- `PersonalityScore` (perfil comportamental).  

Isso indica que o modelo “valoriza” principalmente o desempenho técnico e
comportamental, o que é coerente com boas práticas de recrutamento.

Já variáveis como `DistanceFromCompany` ou `PreviousCompanies`
tendem a apresentar pesos menores, sugerindo impacto reduzido na decisão de
contratação dentro desta base específica.

### 3.2 Ranking dos candidatos

A matriz de scores permite gerar um ranking claro dos candidatos. O script:

- adiciona a coluna **`Score`** ao DataFrame;  
- ordena em ordem decrescente;  
- atribui a cada candidato uma **`PosicaoRanking`**.  

Ao analisar os **Top 5 candidatos**, podem ser observados padrões como:

- altos valores de `InterviewScore` e `SkillScore`;  
- boa avaliação em `PersonalityScore`;  
- experiência compatível com o perfil desejado.  

Também é possível verificar se esses candidatos foram de fato contratados
(valores de `HiringDecision = 1`). Caso algum candidato com score muito alto
não tenha sido contratado, isso pode indicar decisões humanas divergentes
da lógica do modelo.

### 3.3 Avaliação do novo candidato

O novo candidato fictício foi definido, por exemplo, com:

- idade intermediária;  
- boa formação acadêmica;  
- alguns anos de experiência;  
- boa pontuação técnica e comportamental.  

A partir disso, o script calcula o score \(X_{novo} w\) e insere esse valor
no vetor de scores, identificando qual seria a **posição do candidato no
ranking geral**.

Se o candidato aparece entre as primeiras posições, isso indica forte
aderência ao perfil que o modelo enxerga como ideal. Caso contrário, mostra
que alguma dimensão importante (por exemplo, experiência ou entrevista)
ainda está abaixo da média dos melhores candidatos.

### 3.4 Qualidade do ajuste (\(R^2\))

O coeficiente de determinação \(R^2\) é calculado a partir de \(y\) e
\(\hat{y} = Xw\). Um valor de \(R^2\) próximo de 1 indica que o modelo
explica bem a variabilidade da decisão de contratação, enquanto valores
baixos sugerem que há muitos fatores não modelados (por exemplo, aspectos
qualitativos não presentes no dataset).

Em termos pedagógicos, o foco aqui é mostrar como o \(R^2\) permite avaliar
se a combinação linear de variáveis é capaz ou não de reproduzir, ao menos
parcialmente, as decisões tomadas no processo seletivo.

### 3.5 Gráficos gerados

O script produz três gráficos principais, que podem ser inseridos no
relatório final:

- **Barra de pesos de \(w\)**: visualiza quais variáveis têm maior peso.  
- **Histograma dos scores**: mostra a distribuição dos scores de todos os
  candidatos.  
- **Dispersão de \(y\) real vs. \(y\) previsto**: avalia visualmente o
  ajuste do modelo (quanto mais próximos de uma linha crescente, melhor).  

---

## 4. Discussão ética e limitações

Embora o modelo matemático traga transparência e objetividade, seu uso em
recrutamento exige cuidados importantes:

1. **Viés nos dados**  
   Se o histórico de contratações já é enviesado (por exemplo, favorecendo
   um gênero, faixa etária ou grupo social), o modelo apenas reproduzirá
   esse viés. A matemática não é neutra em relação aos dados que recebe.

2. **Redução de pessoas a números**  
   Candidatos têm trajetórias, contextos e potencial que não cabem
   completamente em colunas numéricas. O modelo não pode ser o único
   critério de decisão.

3. **Falta de variáveis relevantes**  
   Características como adaptabilidade, valores pessoais e cultura
   organizacional muitas vezes não estão presentes no dataset, mas são
   fundamentais para o sucesso no trabalho.

4. **Uso responsável**  
   A recomendação é que o modelo seja usado como **apoio à decisão**, e não
   como uma “máquina de contratar/demitir”. Cabe à equipe de RH interpretar
   os resultados, questionar os pesos e ajustar o processo conforme
   princípios de equidade e inclusão.

---

## 5. Conclusão

Este projeto mostrou como a Álgebra Linear pode ser aplicada para analisar
processos de recrutamento, a partir da construção de um modelo
\(Xw \approx y\) e da interpretação do vetor de pesos \(w\).

Entre os principais aprendizados, destacam-se:

- a transformação de um problema de negócios em um modelo matemático
  formal;  
- o uso de métodos numéricos (mínimos quadrados e pseudoinversa) para
  estimativa de parâmetros;  
- a possibilidade de gerar rankings, simular perfis de candidatos e avaliar
  a qualidade do ajuste por meio de \(R^2\);  
- a importância de discutir limitações e implicações éticas de qualquer
  modelo que envolva pessoas.

Como trabalhos futuros, seria interessante:

- incorporar novas variáveis (soft skills, feedback de gestores, desempenho
  pós-contratação);  
- comparar o modelo linear com outros modelos (por exemplo, classificadores
  não lineares);  
- aplicar técnicas de detecção e mitigação de vieses algorítmicos.

