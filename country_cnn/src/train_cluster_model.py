import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model import make_model
from train import train
from evaluate import evaluate_model
from cluster_dataset import CountryToClusterDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_cluster_dataloaders(
    data_dir,
    cluster_csv_path,
    batch_size=16,
    image_size=224,
    val_split=0.2,
    seed=42,
    num_workers=0,
):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_base = datasets.ImageFolder(data_dir, transform=train_transform)
    val_base = datasets.ImageFolder(data_dir, transform=val_transform)

    train_cluster_dataset = CountryToClusterDataset(
        train_base,
        cluster_csv_path=cluster_csv_path,
    )

    val_cluster_dataset = CountryToClusterDataset(
        val_base,
        cluster_csv_path=cluster_csv_path,
    )

    n = len(train_base)
    val_size = int(n * val_split)
    train_size = n - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_cluster_dataset, train_indices)
    val_dataset = Subset(val_cluster_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    cluster_df = pd.read_csv(cluster_csv_path)
    n_clusters = cluster_df["cluster"].nunique()

    print("Number of images:", n)
    print("Train images:", train_size)
    print("Validation images:", val_size)
    print("Number of clusters:", n_clusters)

    return train_loader, val_loader, n_clusters


def main():
    data_dir = "/Users/emmakirchhoff/Desktop/AppliedMLdata2026/train_data"

    # Climate/geography clusters
    cluster_csv_path = "../data/country_clusters.csv"

    save_name = "ResNet34_climate_clusters"

    device = get_device()
    print("Using device:", device)

    train_loader, val_loader, n_clusters = make_cluster_dataloaders(
        data_dir=data_dir,
        cluster_csv_path=cluster_csv_path,
        batch_size=16,
        image_size=224,
        val_split=0.2,
        num_workers=0,
    )

    model = make_model(
        model_name="resnet34",
        num_classes=n_clusters,
        pretrained=True,
    )

    if isinstance(model, tuple):
        model = model[0]

    model = model.to(device)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_name=save_name,
        num_epochs=1,
    )

    print("\nEvaluating cluster model:")
    results = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=None,
    )

    print("\nFinal validation results:")
    print("Top-1:", results["top1"])
    print("Top-5:", results["top5"])


if __name__ == "__main__":
    main()
