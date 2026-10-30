import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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

print("\n=== TIPOS DOS ATRIBUTOS ===")
print(X.dtypes)

# Separar treino e teste
# O teste fica reservado para a avaliação final.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Modelo base: os parâmetros serão escolhidos pelos testes
pipeline = Pipeline([
    ("preencher", SimpleImputer(strategy="median")),
    ("arvore", DecisionTreeClassifier(random_state=42))
])

# Combinações de parâmetros que vamos comparar
parametros = {
    "arvore__max_depth": [4, 6, 8, 10, None],
    "arvore__min_samples_leaf": [1, 5, 10, 20],
    "arvore__class_weight": [None, "balanced"]
}

# Validação cruzada estratificada
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Escolher a combinação com melhor F1 macro médio
busca = GridSearchCV(
    estimator=pipeline,
    param_grid=parametros,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

# Executar a busca pelos melhores parâmetros
print("Testando combinações de parâmetros...")
busca.fit(X_treino, y_treino)

# Recuperar o melhor modelo, treinado com todos os dados de treino
modelo = busca.best_estimator_

print("\n=== MELHOR CONFIGURAÇÃO ===")
print(busca.best_params_)
print(
    "Melhor F1 macro médio na validação cruzada:",
    round(busca.best_score_, 4)
)

# Verificar se há diferença entre treino e validação
indice = busca.best_index_
print(
    "F1 macro no treino (média das divisões):",
    round(busca.cv_results_["mean_train_score"][indice], 4)
)

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


# Salvar uma visualização simplificada da árvore
arvore = modelo.named_steps["arvore"]

plt.figure(figsize=(22, 10))

plot_tree(
    arvore,
    feature_names=X.columns,
    class_names=arvore.classes_,
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=3
)

plt.title("Árvore de Decisão - Primeiros Níveis")
plt.tight_layout()

plt.savefig(
    "data/processed/arvore_decisao_resumida.png",
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