# src/Visualization.py
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_class_names(classification_report_path):
    """
    Load class names from the classification report.
    """

    report_df = pd.read_csv(classification_report_path, index_col=0)

    class_names = [
        name for name in report_df.index
        if name not in ["accuracy", "macro avg", "weighted avg"]
    ]

    return class_names


def normalize_confusion_matrix(cm):
    """
    Normalize the confusion matrix row-wise.

    Each row then describes how one true class is predicted.
    """

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm.astype("float") / np.maximum(row_sums, 1)

    return cm_normalized


def cluster_from_confusion_matrix(cm, classes, output_dir, n_clusters=10):
    """
    Cluster classes based on the confusion matrix.

    If two classes are confused in similar ways, they will be placed
    in the same cluster.
    """

    os.makedirs(output_dir, exist_ok=True)

    cm_normalized = normalize_confusion_matrix(cm)

    cosine_sim = cosine_similarity(cm_normalized)

    distance_matrix = 1.0 - cosine_sim
    distance_matrix = np.clip(distance_matrix, 0, 2)
    np.fill_diagonal(distance_matrix, 0)

    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average"
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average"
        )

    cluster_labels = clustering.fit_predict(distance_matrix)

    cluster_df = pd.DataFrame({
        "class": classes,
        "cluster": cluster_labels
    })

    cluster_df = cluster_df.sort_values(["cluster", "class"])

    output_path = f"{output_dir}/ResNet34_class_clusters.csv"
    cluster_df.to_csv(output_path, index=False)

    print(f"Saved clusters to: {output_path}")

    print("\nClusters:")
    for cluster_id in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == cluster_id]["class"].tolist()
        print(f"Cluster {cluster_id}: {members}")

    return cm_normalized, cosine_sim, cluster_labels


def plot_tsne_umap(cm_normalized, cosine_sim, classes, cluster_labels, base_output_path):
    """
    Plot t-SNE and UMAP visualizations of class confusions.
    If UMAP fails, the code continues and still saves the other plots.
    """

    n_classes = len(classes)
    perplexity = min(30, max(2, n_classes // 3))

    # ------------------------------------------------------------
    # t-SNE on normalized confusion matrix
    # ------------------------------------------------------------

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="random",
        learning_rate="auto"
    )

    tsne_embeddings = tsne.fit_transform(cm_normalized)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=cluster_labels,
        s=100
    )

    plt.title("t-SNE class confusion visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    for i, label in enumerate(classes):
        plt.annotate(
            label,
            (tsne_embeddings[i, 0], tsne_embeddings[i, 1]),
            fontsize=8
        )

    tsne_output_path = f"{base_output_path}_TSNE_clusters.png"
    plt.savefig(tsne_output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE plot to: {tsne_output_path}")

    # ------------------------------------------------------------
    # UMAP on normalized confusion matrix
    # ------------------------------------------------------------

    if HAS_UMAP:
        try:
            umap_model = umap.UMAP(
                n_neighbors=min(15, n_classes - 1),
                random_state=42
            )

            umap_embeddings = umap_model.fit_transform(cm_normalized)

            plt.figure(figsize=(10, 8))
            plt.scatter(
                umap_embeddings[:, 0],
                umap_embeddings[:, 1],
                c=cluster_labels,
                s=100
            )

            plt.title("UMAP class confusion visualization")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")

            for i, label in enumerate(classes):
                plt.annotate(
                    label,
                    (umap_embeddings[i, 0], umap_embeddings[i, 1]),
                    fontsize=8
                )

            umap_output_path = f"{base_output_path}_UMAP_clusters.png"
            plt.savefig(umap_output_path, bbox_inches="tight")
            plt.close()

            print(f"Saved UMAP plot to: {umap_output_path}")

        except Exception as e:
            print("UMAP failed, but clustering and t-SNE were still saved.")
            print("UMAP error:", e)
    else:
        print("UMAP is not installed, so UMAP plot was skipped.")

    # ------------------------------------------------------------
    # t-SNE on cosine similarity
    # ------------------------------------------------------------

    tsne_cosine = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="random",
        learning_rate="auto"
    )

    cosine_embeddings = tsne_cosine.fit_transform(cosine_sim)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        cosine_embeddings[:, 0],
        cosine_embeddings[:, 1],
        c=cluster_labels,
        s=100
    )

    plt.title("t-SNE cosine similarity visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    for i, label in enumerate(classes):
        plt.annotate(
            label,
            (cosine_embeddings[i, 0], cosine_embeddings[i, 1]),
            fontsize=8
        )

    cosine_output_path = f"{base_output_path}_Cosine_TSNE_clusters.png"
    plt.savefig(cosine_output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved cosine t-SNE plot to: {cosine_output_path}")


if __name__ == "__main__":

    confusion_matrix_path = "country_cnn/outputs/models/ResNet34_confusion_matrix.npy"
    classification_report_path = "country_cnn/outputs/models/ResNet34_classification_report.csv"

    output_dir = "country_cnn/outputs/clustering_results"
    os.makedirs(output_dir, exist_ok=True)

    n_clusters = 10

    cm = np.load(confusion_matrix_path)
    classes = load_class_names(classification_report_path)

    print("Confusion matrix shape:", cm.shape)
    print("Number of classes:", len(classes))

    if cm.shape[0] != len(classes):
        raise ValueError(
            f"Mismatch: confusion matrix has {cm.shape[0]} classes, "
            f"but classification report has {len(classes)} class names."
        )

    cm_normalized, cosine_sim, cluster_labels = cluster_from_confusion_matrix(
        cm=cm,
        classes=classes,
        output_dir=output_dir,
        n_clusters=n_clusters
    )

    plot_tsne_umap(
        cm_normalized=cm_normalized,
        cosine_sim=cosine_sim,
        classes=classes,
        cluster_labels=cluster_labels,
        base_output_path=f"{output_dir}/ResNet34_class_confusions"
    )