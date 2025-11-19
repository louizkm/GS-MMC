
"""
Global Solution - Modelagem Matemática e Computacional
-------------------------------------------------------
Recrutamento guiado por Álgebra Linear

Este script:
1. Carrega o dataset de recrutamento (recruitment_data.csv).
2. Separa a matriz de features X e a variável-alvo y (HiringDecision).
3. Calcula o vetor de pesos w usando:
   - Mínimos quadrados (np.linalg.lstsq)
   - Pseudoinversa (np.linalg.pinv)
4. Compara os vetores encontrados.
5. Interpreta os pesos das variáveis.
6. Calcula o score Xw e gera um ranking dos candidatos.
7. Localiza os 5 candidatos com maior score.
8. Avalia um novo candidato definido manualmente.
9. Gera alguns gráficos simples para análise.

Autores: (preencher com os nomes do grupo)
Data: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------
# 1. Carregando os dados e separando X e y
# -------------------------------------------------------

def carregar_dados(caminho_csv: str = "recruitment_data.csv"):
    """
    Lê o arquivo CSV e separa as variáveis de entrada (X)
    e a variável-alvo (y = HiringDecision).
    """
    df = pd.read_csv(caminho_csv)

    # Define a coluna alvo
    target_col = "HiringDecision"

    # Features: todas as colunas menos a coluna alvo
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    return df, X, y, feature_cols, target_col


# -------------------------------------------------------
# 2. Resolvendo o sistema Xw ≈ y com dois métodos
# -------------------------------------------------------

def calcular_pesos(X: np.ndarray, y: np.ndarray):
    """
    Calcula o vetor de pesos w de duas formas:
    - Mínimos quadrados (np.linalg.lstsq)
    - Pseudoinversa (np.linalg.pinv)
    """
    # Método 1: least squares
    w_lstsq, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Método 2: pseudoinversa
    X_pinv = np.linalg.pinv(X)
    w_pinv = X_pinv @ y

    return w_lstsq, w_pinv, residuals, rank, s


def comparar_pesos(w_lstsq: np.ndarray, w_pinv: np.ndarray, feature_cols):
    """
    Imprime uma tabela comparando os pesos obtidos
    por mínimos quadrados e pseudoinversa.
    """
    print("\n=== Comparação dos pesos (w) por método ===\n")
    print(f"{'Feature':<25} {'w_lstsq':>12} {'w_pinv':>12} {'diferença':>12}")
    print("-" * 65)
    for feat, wl, wp in zip(feature_cols, w_lstsq, w_pinv):
        diff = wl - wp
        print(f"{feat:<25} {wl:>12.6f} {wp:>12.6f} {diff:>12.6e}")

    max_diff = np.max(np.abs(w_lstsq - w_pinv))
    print("\nMáxima diferença absoluta entre w_lstsq e w_pinv:", max_diff)


# -------------------------------------------------------
# 3. Interpretação básica dos pesos
# -------------------------------------------------------

def imprimir_pesos_ordenados(w: np.ndarray, feature_cols):
    """
    Exibe os pesos ordenados por magnitude absoluta
    para facilitar a interpretação.
    """
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Peso_w": w
    })
    coef_df["|w|"] = coef_df["Peso_w"].abs()
    coef_df = coef_df.sort_values("|w|", ascending=False)

    print("\n=== Pesos ordenados por importância (|w|) ===\n")
    print(coef_df[["Feature", "Peso_w"]].to_string(index=False))


# -------------------------------------------------------
# 4. Ranking de candidatos via score = Xw
# -------------------------------------------------------

def calcular_scores_e_ranking(df: pd.DataFrame,
                              X: np.ndarray,
                              y: np.ndarray,
                              w: np.ndarray,
                              target_col: str,
                              top_n: int = 5):
    """
    Calcula o score Xw para todos os candidatos,
    ordena em ordem decrescente e mostra os top_n.
    """
    scores = X @ w
    df_scores = df.copy()
    df_scores["Score"] = scores

    # Ranking decrescente
    df_scores = df_scores.sort_values("Score", ascending=False)
    df_scores["PosicaoRanking"] = np.arange(1, len(df_scores) + 1)

    print(f"\n=== Top {top_n} candidatos por score Xw ===\n")
    cols_to_show = list(df.columns) + ["Score", "PosicaoRanking"]
    print(df_scores[cols_to_show].head(top_n).to_string(index=False))

    # Verifica se os top_n foram contratados (y = 1)
    contratados_top = df_scores[target_col].head(top_n).values
    print("\nDecisão de contratação (y) dos Top", top_n, "candidatos:", contratados_top)

    return df_scores, scores


# -------------------------------------------------------
# 5. Avaliando um novo candidato
# -------------------------------------------------------

def avaliar_novo_candidato(w: np.ndarray, feature_cols):
    """
    Define um novo candidato manualmente, calcula seu score
    e determina sua posição no ranking.
    Os valores podem ser ajustados conforme o grupo desejar.
    """
    # Exemplo de novo candidato coerente com a estrutura do dataset
    novo_candidato = {
        "Age": 32,
        "Gender": 1,
        "EducationLevel": 3,
        "ExperienceYears": 5,
        "PreviousCompanies": 2,
        "DistanceFromCompany": 10.0,
        "InterviewScore": 75,
        "SkillScore": 80,
        "PersonalityScore": 85,
        "RecruitmentStrategy": 1
    }

    # Cria vetor na mesma ordem das features
    x_new = np.array([novo_candidato[col] for col in feature_cols], dtype=float)
    score_new = float(x_new @ w)

    print("\n=== Avaliação de novo candidato ===\n")
    print("Dados do candidato:")
    for k, v in novo_candidato.items():
        print(f"  {k}: {v}")
    print(f"\nScore gerado (x_new @ w): {score_new:.6f}")

    return novo_candidato, score_new


def inserir_novo_no_ranking(scores: np.ndarray, score_new: float):
    """
    Insere o novo score no vetor de scores original
    e calcula sua posição no ranking.
    """
    all_scores = np.append(scores, score_new)
    # Índices ordenados em ordem decrescente
    ranking_idx = np.argsort(all_scores)[::-1]
    # Índice do novo candidato (última posição no array all_scores)
    idx_novo = len(all_scores) - 1
    posicao = int(np.where(ranking_idx == idx_novo)[0][0]) + 1

    print(f"\nPosição do novo candidato no ranking: {posicao} de {len(all_scores)}")
    return posicao


# -------------------------------------------------------
# 6. Avaliação global do ajuste (R²) e gráficos simples
# -------------------------------------------------------

def avaliar_ajuste(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """
    Calcula o R² do modelo linear y_hat = Xw e imprime o valor.
    """
    y_pred = X @ w
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\nR² do modelo linear (usando todos os dados): {r2:.3f}")
    return y_pred, r2


def gerar_graficos_simples(feature_cols, w, scores, y, y_pred):
    """
    Gera alguns gráficos simples para o relatório:
    - Barra com os pesos w
    - Histograma dos scores
    - Dispersão y real vs. y previsto
    """
    # Gráfico 1: Pesos w
    plt.figure()
    plt.bar(range(len(w)), w)
    plt.xticks(range(len(w)), feature_cols, rotation=45, ha="right")
    plt.title("Pesos do vetor w por feature")
    plt.tight_layout()
    plt.savefig("grafico_pesos_w.png", dpi=150)

    # Gráfico 2: Histograma dos scores
    plt.figure()
    plt.hist(scores, bins=30, edgecolor="black")
    plt.title("Distribuição dos scores Xw")
    plt.xlabel("Score")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig("hist_scores.png", dpi=150)

    # Gráfico 3: y real vs y previsto
    plt.figure()
    plt.scatter(y_pred, y, alpha=0.5)
    plt.xlabel("y previsto (Xw)")
    plt.ylabel("y real (HiringDecision)")
    plt.title("Comparação entre y previsto e y real")
    plt.tight_layout()
    plt.savefig("scatter_y_real_vs_previsto.png", dpi=150)

    print("\nGráficos salvos como:")
    print(" - grafico_pesos_w.png")
    print(" - hist_scores.png")
    print(" - hist_scores.png")
    print(" - scatter_y_real_vs_previsto.png")


# -------------------------------------------------------
# Função principal
# -------------------------------------------------------

def main():
    print("=" * 70)
    print("RECRUTAMENTO GUIADO POR ÁLGEBRA LINEAR")
    print("Global Solution - Modelagem Matemática e Computacional")
    print("=" * 70)

    # 1) Carrega dados
    df, X, y, feature_cols, target_col = carregar_dados()

    print("\nDados carregados com sucesso!")
    print(f"Linhas (candidatos): {df.shape[0]}")
    print(f"Colunas (features + alvo): {df.shape[1]}")
    print("\nColunas disponíveis:")
    print(df.columns.tolist())

    # 2) Calcula pesos por dois métodos
    w_lstsq, w_pinv, residuals, rank, s = calcular_pesos(X, y)

    comparar_pesos(w_lstsq, w_pinv, feature_cols)
    imprimir_pesos_ordenados(w_lstsq, feature_cols)

    # 3) Ranking
    df_scores, scores = calcular_scores_e_ranking(df, X, y, w_lstsq, target_col, top_n=5)

    # 4) Avaliação de novo candidato
    novo_candidato, score_new = avaliar_novo_candidato(w_lstsq, feature_cols)
    posicao = inserir_novo_no_ranking(scores, score_new)

    # 5) Avaliação global do ajuste
    y_pred, r2 = avaliar_ajuste(X, y, w_lstsq)

    # 6) Geração de gráficos
    gerar_graficos_simples(feature_cols, w_lstsq, scores, y, y_pred)

    print("\nExecução concluída com sucesso.")


if __name__ == "__main__":
    main()
