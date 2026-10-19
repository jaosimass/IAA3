from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RAIZ = Path(__file__).resolve().parent.parent

HISTORICO = (
    RAIZ
    / "results"
    / "rna"
    / "historico_treinamento.csv"
)

RESULTADOS_DIR = (
    RAIZ
    / "results"
    / "rna"
)


def carregar_historico():
    historico = pd.read_csv(HISTORICO)

    # As épocas começam visualmente em 1,
    # e não em 0.
    historico["epoch"] = range(
        1,
        len(historico) + 1
    )

    return historico


def gerar_grafico_loss(historico):

    plt.figure(figsize=(10, 6))

    plt.plot(
        historico["epoch"],
        historico["loss"],
        label="Training loss"
    )

    plt.plot(
        historico["epoch"],
        historico["val_loss"],
        label="Validation loss"
    )

    plt.title("RNA - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    caminho = (
        RESULTADOS_DIR
        / "historico_loss.png"
    )

    plt.savefig(
        caminho,
        dpi=200
    )

    plt.close()

    print(
        "Gráfico de loss salvo em:",
        caminho
    )


def gerar_grafico_accuracy(historico):

    plt.figure(figsize=(10, 6))

    plt.plot(
        historico["epoch"],
        historico["accuracy"],
        label="Training accuracy"
    )

    plt.plot(
        historico["epoch"],
        historico["val_accuracy"],
        label="Validation accuracy"
    )

    plt.title(
        "RNA - Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    caminho = (
        RESULTADOS_DIR
        / "historico_accuracy.png"
    )

    plt.savefig(
        caminho,
        dpi=200
    )

    plt.close()

    print(
        "Gráfico de accuracy salvo em:",
        caminho
    )


def main():

    historico = carregar_historico()

    print()
    print(
        "Épocas registradas:",
        len(historico)
    )

    print()

    gerar_grafico_loss(
        historico
    )

    gerar_grafico_accuracy(
        historico
    )


if __name__ == "__main__":
    main()