from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model import make_model
from cluster_dataset import CountryToClusterDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_val_loader(
    data_dir="/Users/emmakirchhoff/Desktop/AppliedMLdata2026/train_data",
    cluster_csv_path="../data/country_clusters.csv",
    batch_size=16,
    image_size=224,
    val_split=0.2,
    seed=42,
):
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    base = datasets.ImageFolder(data_dir, transform=val_transform)
    cluster_dataset = CountryToClusterDataset(base, cluster_csv_path)

    n = len(base)
    val_size = int(n * val_split)
    train_size = n - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_indices = indices[train_size:]

    val_dataset = Subset(cluster_dataset, val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return val_loader


def predict(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    return labels, preds


def main():
    output_dir = Path("../outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_csv_path = "../data/country_clusters.csv"
    model_path = "../outputs/models/ResNet34_climate_clusters.pt"

    cluster_df = pd.read_csv(cluster_csv_path)
    n_clusters = cluster_df["cluster"].nunique()

    device = get_device()
    print("Using device:", device)

    val_loader = make_val_loader(cluster_csv_path=cluster_csv_path)

    model = make_model(
        model_name="resnet34",
        num_classes=n_clusters,
        pretrained=True,
    )

    if isinstance(model, tuple):
        model = model[0]

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    y_true, y_pred = predict(model, val_loader, device)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_clusters)))

    # Save classification report as CSV
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("../outputs/models/ResNet34_climate_clusters_report.csv")

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix: climate/geography clusters")
    plt.xlabel("Predicted cluster")
    plt.ylabel("True cluster")
    plt.colorbar(label="Number of images")
    plt.xticks(range(n_clusters), range(n_clusters), rotation=90)
    plt.yticks(range(n_clusters), range(n_clusters))
    plt.tight_layout()

    cm_path = output_dir / "ResNet34_climate_clusters_confusion_matrix.png"
    plt.savefig(cm_path, dpi=250)
    plt.close()

    print(f"Saved confusion matrix to: {cm_path}")

    # Plot per-cluster F1-score
    cluster_rows = report_df.loc[
        [str(i) for i in range(n_clusters)],
        ["precision", "recall", "f1-score", "support"]
    ]

    plt.figure(figsize=(14, 6))
    plt.bar(cluster_rows.index.astype(int), cluster_rows["f1-score"])
    plt.xlabel("Cluster")
    plt.ylabel("F1-score")
    plt.title("Per-cluster F1-score")
    plt.xticks(range(n_clusters))
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    f1_path = output_dir / "ResNet34_climate_clusters_f1_scores.png"
    plt.savefig(f1_path, dpi=250)
    plt.close()

    print(f"Saved F1-score plot to: {f1_path}")


if __name__ == "__main__":
    main()
