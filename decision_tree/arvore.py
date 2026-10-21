
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score
)

# Garantir que a pasta de resultados exista
os.makedirs("data/processed", exist_ok=True)

# 1. Carregar os dados preparados
df = pd.read_csv("data/processed/dados_arvore.csv")

# 2. Definir atributos de entrada e alvo
colunas_excluir = [
    "SEQ OCORRENCIA INCENDIO",
    "NIVEL ACIONAMENTO"
]

X = df.drop(columns=colunas_excluir)
y = df["NIVEL ACIONAMENTO"]

# Separar treino e teste
# O teste fica reservado para a avaliação final.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Definir a configuração escolhida
def criar_modelo():
    return Pipeline([
        ("preencher", SimpleImputer(strategy="median")),
        ("arvore", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ))
    ])

# Validacao cruzada somente nos dados de treino
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

modelo_cv = criar_modelo()

resultados_cv = cross_validate(
    modelo_cv,
    X_treino,
    y_treino,
    cv=cv,
    scoring={
        "acuracia": "accuracy",
        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0
        )
    }
)

print("\n=== VALIDACAO CRUZADA ===")
print(
    "Acuracia media:",
    round(resultados_cv["test_acuracia"].mean(), 4)
)
print(
    "F1 macro medio:",
    round(resultados_cv["test_f1_macro"].mean(), 4)
)

# Treinar o modelo final com todos os dados de treino
modelo = criar_modelo()
modelo.fit(X_treino, y_treino)

# Avaliar uma unica vez nos dados de teste
previsoes = modelo.predict(X_teste)

comparacao = pd.DataFrame({
    "nivel_real": y_teste,
    "nivel_previsto": previsoes
})

comparacao["acertou"] = (
    comparacao["nivel_real"] ==
    comparacao["nivel_previsto"]
)

comparacao.to_csv(
    "data/processed/previsoes_arvore.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== AVALIACAO FINAL ===")
print("Total para treinamento:", len(X_treino))
print("Total para teste:", len(X_teste))
print("Total de acertos:", comparacao["acertou"].sum())
print("Total de erros:", (~comparacao["acertou"]).sum())

print(
    "\nAcuracia no teste:",
    round(accuracy_score(y_teste, previsoes), 4)
)

print("\nRelatorio por nivel:")
print(classification_report(
    y_teste,
    previsoes,
    labels=["Nível 1", "Nível 2", "Nível 3"],
    zero_division=0
))

# Matriz de confusao organizada
ordem_classes = ["Nível 1", "Nível 2", "Nível 3"]

matriz = confusion_matrix(
    y_teste,
    previsoes,
    labels=ordem_classes
)

matriz_df = pd.DataFrame(
    matriz,
    index=[f"Real: {c}" for c in ordem_classes],
    columns=[f"Previsto: {c}" for c in ordem_classes]
)

print("\nMatriz de confusao:")
print(matriz_df)

plt.figure(figsize=(8, 6))
plt.imshow(matriz, interpolation="nearest")
plt.title("Matriz de Confusao - Arvore de Decisao")
plt.xlabel("Nivel previsto")
plt.ylabel("Nivel real")
plt.xticks(range(3), ordem_classes, rotation=15)
plt.yticks(range(3), ordem_classes)
plt.colorbar(label="Quantidade")

for i in range(3):
    for j in range(3):
        plt.text(
            j, i, str(matriz[i, j]),
            ha="center", va="center"
        )

plt.tight_layout()
plt.savefig(
    "data/processed/matriz_confusao.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Salvar a visualizacao da arvore
arvore = modelo.named_steps["arvore"]

plt.figure(figsize=(24, 12))
plot_tree(
    arvore,
    feature_names=X.columns,
    class_names=arvore.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.tight_layout()
plt.savefig(
    "data/processed/arvore_decisao.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Importancia dos atributos
importancias = pd.Series(
    arvore.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nImportancia dos atributos:")
print(importancias[importancias > 0].round(4))

# 11. Analisar os erros
print("\nErros por combinacao:")
erros = comparacao[~comparacao["acertou"]]

if erros.empty:
    print("Nenhum erro encontrado.")
else:
    print(
        erros.groupby(
            ["nivel_real", "nivel_previsto"]
        ).size().sort_values(ascending=False)
    )

print("\nArquivos salvos:")
print("- data/processed/previsoes_arvore.csv")
print("- data/processed/matriz_confusao.png")
print("- data/processed/arvore_decisao.png")