from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def save_cluster_scatter(dataframe: pd.DataFrame, output_path: str | Path):
    """Gráfico simples de clusters usando médias RGB como eixos."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_col, y_col = "red_mean", "green_mean"

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster in sorted(dataframe["cluster"].unique()):
        group = dataframe[dataframe["cluster"] == cluster]
        ax.scatter(group[x_col], group[y_col], alpha=0.7, s=24,
                   label=f"Cluster {cluster}")
    ax.set_title("Agrupamento K-Means: médias dos canais RGB")
    ax.set_xlabel("Média do canal vermelho")
    ax.set_ylabel("Média do canal verde")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def save_cluster_counts(dataframe: pd.DataFrame, output_path: str | Path):
    output_path = Path(output_path)
    counts = dataframe["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Quantidade de imagens por cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Número de imagens")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def save_confusion_matrix(matrix, labels, output_path: str | Path):
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Classe estimada pela interpretação dos clusters",
        ylabel="Classe real",
        title="Matriz de confusão no conjunto de teste"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def save_cluster_examples(dataframe: pd.DataFrame, output_dir: str | Path, max_per_cluster=6):
    """Monta uma imagem de exemplos de cada cluster para inspeção visual."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster in sorted(dataframe["cluster"].unique()):
        group = dataframe[dataframe["cluster"] == cluster].head(max_per_cluster)
        if group.empty:
            continue
        fig, axes = plt.subplots(1, len(group), figsize=(3 * len(group), 3))
        if len(group) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, group.iterrows()):
            try:
                with Image.open(row["image_path"]) as image:
                    ax.imshow(image.convert("RGB"))
                ax.set_title(Path(row["image_path"]).name[:18], fontsize=8)
                ax.axis("off")
            except OSError:
                ax.text(0.5, 0.5, "Imagem indisponível", ha="center")
                ax.axis("off")
        fig.suptitle(f"Exemplos do cluster {cluster}")
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{cluster}_examples.png", dpi=140)
        plt.close(fig)
