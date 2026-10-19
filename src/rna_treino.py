from pathlib import Path
import pickle

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from rna_dados import carregar_dados


RAIZ = Path(__file__).resolve().parent.parent

MODELO_SAIDA = RAIZ / "models" / "rna_fire_nofire.keras"
SCALER_SAIDA = RAIZ / "models" / "rna_scaler.pkl"
RESULTADOS_DIR = RAIZ / "results" / "rna"


def construir_rna():
    model = keras.Sequential([
        keras.layers.Input(shape=(6,)),

        keras.layers.Dense(
            16,
            activation="relu"
        ),

        keras.layers.Dense(
            8,
            activation="relu"
        ),

        keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def treinar():
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        scaler
    ) = carregar_dados()

    MODELO_SAIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = construir_rna()

    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,

        validation_data=(
            x_val,
            y_val
        ),

        epochs=100,
        batch_size=32,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print()
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    model.save(MODELO_SAIDA)

    with open(SCALER_SAIDA, "wb") as arquivo:
        pickle.dump(scaler, arquivo)

    historico = pd.DataFrame(
        history.history
    )

    historico.to_csv(
        RESULTADOS_DIR / "historico_treinamento.csv",
        index=False
    )

    print()
    print("Modelo salvo em:", MODELO_SAIDA)
    print("Scaler salvo em:", SCALER_SAIDA)
    print(
        "Histórico salvo em:",
        RESULTADOS_DIR / "historico_treinamento.csv"
    )


if __name__ == "__main__":
    treinar()