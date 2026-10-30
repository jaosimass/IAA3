from collections import Counter
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

def cluster_metrics(features_scaled, cluster_ids, true_labels=None):
    """Calcula métricas de agrupamento; rótulos reais são opcionais."""
    metrics = {
        "n_images": int(len(cluster_ids)),
        "n_clusters_found": int(len(set(cluster_ids))),
    }
    if len(set(cluster_ids)) > 1 and len(set(cluster_ids)) < len(cluster_ids):
        metrics["silhouette_score"] = float(
            silhouette_score(features_scaled, cluster_ids)
        )
    else:
        metrics["silhouette_score"] = None

    if true_labels is not None:
        metrics["adjusted_rand_index"] = float(
            adjusted_rand_score(true_labels, cluster_ids)
        )
        metrics["normalized_mutual_info"] = float(
            normalized_mutual_info_score(true_labels, cluster_ids)
        )
    return metrics

def build_cluster_label_mapping(cluster_ids, true_labels):
    """
    Dá um nome interpretável a cada cluster usando a classe mais frequente
    no conjunto de treino. Isso interpreta os clusters; não treina o K-Means.
    """
    mapping = {}
    table = pd.DataFrame({"cluster": cluster_ids, "label": true_labels})
    for cluster, group in table.groupby("cluster"):
        mapping[int(cluster)] = Counter(group["label"]).most_common(1)[0][0]
    return mapping

def evaluate_test_set(test_labels, test_clusters, train_mapping):
    predicted_labels = [train_mapping.get(int(c), "unknown") for c in test_clusters]
    labels = sorted(set(test_labels) | set(predicted_labels))
    report = classification_report(
        test_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    matrix = confusion_matrix(test_labels, predicted_labels, labels=labels)
    return predicted_labels, report, matrix, labels
