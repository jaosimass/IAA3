from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from rna_dados import carregar_dados


RAIZ = Path(__file__).resolve().parent.parent

MODELO = RAIZ / "models" / "rna_fire_nofire.keras"
RESULTADOS_DIR = RAIZ / "results" / "rna"


def avaliar_rna():

    # Carrega os mesmos dados e a mesma divisão
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        scaler
    ) = carregar_dados()

    # Carrega a RNA já treinada
    model = tf.keras.models.load_model(MODELO)

    # A saída da sigmoid é uma probabilidade
    probabilidades = model.predict(
        x_test,
        verbose=0
    ).ravel()

    # Converte probabilidade em classe
    # >= 0.5 = fire
    # < 0.5 = nofire
    previsoes = (
        probabilidades >= 0.5
    ).astype(int)

    # Métricas
    accuracy = accuracy_score(
        y_test,
        previsoes
    )

    precision = precision_score(
        y_test,
        previsoes
    )

    recall = recall_score(
        y_test,
        previsoes
    )

    f1 = f1_score(
        y_test,
        previsoes
    )

    matriz = confusion_matrix(
        y_test,
        previsoes
    )

    print()
    print("=== RNA - RESULTADOS ===")
    print()

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    print()
    print("Matriz de confusão:")
    print(matriz)

    print()
    print("Relatório:")
    print(
        classification_report(
            y_test,
            previsoes,
            target_names=[
                "nofire",
                "fire"
            ]
        )
    )

    # Salva as métricas
    metricas = pd.DataFrame([
        {
            "modelo": "rna",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tn": matriz[0][0],
            "fp": matriz[0][1],
            "fn": matriz[1][0],
            "tp": matriz[1][1]
        }
    ])

    metricas.to_csv(
        RESULTADOS_DIR / "metricas_rna.csv",
        index=False
    )

    # Salva cada previsão individual
    resultados = pd.DataFrame({
        "real": y_test,
        "probabilidade_fire": probabilidades,
        "previsto": previsoes
    })

    resultados.to_csv(
        RESULTADOS_DIR / "predicoes_rna.csv",
        index=False
    )

    print()
    print(
        "Métricas salvas em:",
        RESULTADOS_DIR / "metricas_rna.csv"
    )


if __name__ == "__main__":
    avaliar_rna()