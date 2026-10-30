from pathlib import Path
import json
import sys

# Permite executar tanto `python src/main.py` quanto `python -m src.main`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import extract_dataset
from src.clustering import fit_kmeans, predict_clusters
from src.evaluation import (
    cluster_metrics, build_cluster_label_mapping, evaluate_test_set
)
from src.visualization import (
    save_cluster_scatter, save_cluster_counts,
    save_confusion_matrix, save_cluster_examples
)
from sklearn.preprocessing import StandardScaler

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
N_CLUSTERS = 2

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("1/6 - Extraindo características das imagens...")
    train = extract_dataset(DATA_DIR / "train")
    test = extract_dataset(DATA_DIR / "test")

    print(f"Imagens de treino: {len(train)}")
    print(f"Imagens de teste: {len(test)}")
    print("Classes de treino:", train["label"].value_counts().to_dict())
    print("Classes de teste:", test["label"].value_counts().to_dict())

    print("2/6 - Ajustando o K-Means com as características de treino...")
    scaler, model, feature_columns, train_clusters = fit_kmeans(
        train, n_clusters=N_CLUSTERS
    )
    train_results = train.copy()
    train_results["cluster"] = train_clusters
    train_results.to_csv(RESULTS_DIR / "train_clusters.csv", index=False)

    print("3/6 - Atribuindo as imagens de teste aos clusters...")
    test_clusters = predict_clusters(test, scaler, model, feature_columns)
    test_results = test.copy()
    test_results["cluster"] = test_clusters
    test_results.to_csv(RESULTS_DIR / "test_clusters.csv", index=False)

    print("4/6 - Avaliando os agrupamentos...")
    train_scaled = scaler.transform(train[feature_columns].astype(float))
    test_scaled = scaler.transform(test[feature_columns].astype(float))
    train_metrics = cluster_metrics(train_scaled, train_clusters, train["label"])
    test_metrics = cluster_metrics(test_scaled, test_clusters, test["label"])

    # Mapeamento de clusters para nomes de classe usa somente o treino.
    mapping = build_cluster_label_mapping(train_clusters, train["label"])
    predicted_labels, report, matrix, labels = evaluate_test_set(
        test["label"].tolist(), test_clusters, mapping
    )
    test_results["interpreted_label"] = predicted_labels
    test_results.to_csv(RESULTS_DIR / "test_clusters.csv", index=False)

    metrics = {
        "n_clusters_requested": N_CLUSTERS,
        "feature_columns": feature_columns,
        "cluster_to_label_mapping_from_train": {
            str(key): value for key, value in mapping.items()
        },
        "train_clustering_metrics": train_metrics,
        "test_clustering_metrics": test_metrics,
        "test_report_after_cluster_interpretation": report,
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(RESULTS_DIR / "confusion_matrix.csv", "w", encoding="utf-8") as file:
        file.write("real/predicted," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            file.write(label + "," + ",".join(map(str, row)) + "\n")

    print("5/6 - Criando gráficos e exemplos...")
    save_cluster_scatter(train_results, RESULTS_DIR / "clusters_rgb.png")
    save_cluster_counts(train_results, RESULTS_DIR / "cluster_counts.png")
    save_confusion_matrix(matrix, labels, RESULTS_DIR / "confusion_matrix.png")
    save_cluster_examples(train_results, RESULTS_DIR / "cluster_examples")

    print("6/6 - Concluído!")
    print(f"Resultados salvos em: {RESULTS_DIR}")
    print("Métricas de treino:", json.dumps(train_metrics, ensure_ascii=False, indent=2))
    print("Métricas de teste:", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print("Atenção: métricas e grupos precisam ser interpretados; K-Means não é, por si só, um detector de incêndios.")

if __name__ == "__main__":
    main()
