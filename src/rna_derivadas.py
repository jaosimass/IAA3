from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from rna_dados import carregar_dados


RAIZ = Path(__file__).resolve().parent.parent

MODELO = RAIZ / "models" / "rna_fire_nofire.keras"
RESULTADOS_DIR = RAIZ / "results" / "rna"


# =========================================================
# PARTE 1 - DERIVADA COMO LIMITE
# =========================================================

def derivada_numerica(funcao, x, h):
    return (
        funcao(x + h) - funcao(x)
    ) / h


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivada_sigmoid_analitica(x):
    s = sigmoid(x)

    return s * (1 - s)


def demonstrar_limite():
    x = 1.0

    valores_h = [
        1,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001
    ]

    derivada_exata = derivada_sigmoid_analitica(x)

    resultados = []

    print()
    print("=== DERIVADA DA SIGMOID POR LIMITE ===")
    print()

    print(
        "Derivada analítica:",
        derivada_exata
    )

    print()

    for h in valores_h:

        aproximacao = derivada_numerica(
            sigmoid,
            x,
            h
        )

        erro = abs(
            aproximacao - derivada_exata
        )

        print(
            f"h = {h:<8} "
            f"derivada ≈ {aproximacao:.8f} "
            f"erro = {erro:.8f}"
        )

        resultados.append({
            "h": h,
            "derivada_numerica": aproximacao,
            "derivada_analitica": derivada_exata,
            "erro": erro
        })

    return resultados


# =========================================================
# PARTE 2 - DERIVADA EM UM PESO REAL DA RNA
# =========================================================

def calcular_loss(modelo, x, y):

    previsao = modelo(
        x,
        training=False
    )

    loss = tf.keras.losses.binary_crossentropy(
        y,
        previsao
    )

    return tf.reduce_mean(loss)


def derivada_peso_real():

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        scaler
    ) = carregar_dados()

    modelo = tf.keras.models.load_model(
        MODELO
    )

    # Usa somente uma amostra para deixar
    # a demonstração mais simples
    x = tf.convert_to_tensor(
        x_test[0:1],
        dtype=tf.float32
    )

    y = tf.convert_to_tensor(
        y_test[0:1].reshape(-1, 1),
        dtype=tf.float32
    )

    # ---------------------------------
    # DERIVADA CALCULADA PELO TENSORFLOW
    # ---------------------------------

    with tf.GradientTape() as tape:

        prediction = modelo(
            x,
            training=False
        )

        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y,
                prediction
            )
        )

    gradientes = tape.gradient(
        loss,
        modelo.trainable_variables
    )

    primeira_matriz_pesos = (
        modelo.trainable_variables[0]
    )

    gradiente_tensorflow = (
        gradientes[0][0, 0].numpy()
    )

    peso_original = (
        primeira_matriz_pesos[0, 0].numpy()
    )

    # ---------------------------------
    # DERIVADA NUMÉRICA
    # ---------------------------------

    h = 0.0001

    pesos_copia = (
        primeira_matriz_pesos.numpy().copy()
    )

    # Peso + h
    pesos_mais = pesos_copia.copy()
    pesos_mais[0, 0] += h

    primeira_matriz_pesos.assign(
        pesos_mais
    )

    loss_mais = calcular_loss(
        modelo,
        x,
        y
    ).numpy()

    # Peso - h
    pesos_menos = pesos_copia.copy()
    pesos_menos[0, 0] -= h

    primeira_matriz_pesos.assign(
        pesos_menos
    )

    loss_menos = calcular_loss(
        modelo,
        x,
        y
    ).numpy()

    # Restaura peso verdadeiro
    primeira_matriz_pesos.assign(
        pesos_copia
    )

    # Diferença central
    derivada_numerica_peso = (
        loss_mais - loss_menos
    ) / (2 * h)

    print()
    print(
        "=== DERIVADA EM UM PESO REAL DA RNA ==="
    )
    print()

    print(
        "Target:",
        y.numpy()[0][0]
    )

    print(
        "Prediction:",
        prediction.numpy()[0][0]
    )

    print(
        "Loss:",
        loss.numpy()
    )

    print()

    print(
        "Peso analisado:",
        peso_original
    )

    print(
        "Derivada numérica:",
        derivada_numerica_peso
    )

    print(
        "Gradiente TensorFlow:",
        gradiente_tensorflow
    )

    print(
        "Diferença:",
        abs(
            derivada_numerica_peso
            - gradiente_tensorflow
        )
    )

    return {
        "peso": peso_original,
        "derivada_numerica":
            derivada_numerica_peso,
        "gradiente_tensorflow":
            gradiente_tensorflow,
        "diferenca": abs(
            derivada_numerica_peso
            - gradiente_tensorflow
        )
    }


if __name__ == "__main__":

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    resultados_limite = (
        demonstrar_limite()
    )

    resultado_peso = (
        derivada_peso_real()
    )

    pd.DataFrame(
        resultados_limite
    ).to_csv(
        RESULTADOS_DIR
        / "limite_derivada_sigmoid.csv",
        index=False
    )

    pd.DataFrame([
        resultado_peso
    ]).to_csv(
        RESULTADOS_DIR
        / "derivada_peso_rna.csv",
        index=False
    )

    print()
    print(
        "Resultados salvos em:",
        RESULTADOS_DIR
    )