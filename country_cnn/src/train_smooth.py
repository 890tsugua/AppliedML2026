import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import pandas as pd


def get_cluster_csv_path(cluster_k):
    """
    Returns path to the cluster CSV file based on K.

    Expected file names:
        country_climate_geography_clusters_K15.csv
        country_clusters_K30.csv
    """

    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent

    if cluster_k == 15:
        return project_dir / "country_climate_geography_clusters_K15.csv"

    if cluster_k == 30:
        return project_dir / "country_clusters_K30.csv"

    raise ValueError(
        f"Unsupported cluster_k={cluster_k}. Use cluster_k=15 or cluster_k=30."
    )


def make_cluster_smoothing_matrix(
    class_names,
    device,
    cluster_k=15,
    correct_prob=0.9,
    csv_path=None,
):
    """
    Creates a soft-label smoothing matrix based on country clusters.

    For each true country:
        correct_prob goes to the true country
        1 - correct_prob is distributed across countries in the same cluster

    Args:
        class_names: list of class/country names in the same order as the model output.
        device: torch device.
        cluster_k: 15 or 30.
        correct_prob: probability assigned to the correct country.
        csv_path: optional manual path to cluster CSV.

    Returns:
        smoothing matrix of shape [num_classes, num_classes].
    """

    if not (0.0 < correct_prob <= 1.0):
        raise ValueError("correct_prob must be in the interval (0, 1].")

    if csv_path is None:
        csv_path = get_cluster_csv_path(cluster_k)

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Cluster file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"country", "cluster"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Cluster CSV must contain columns {required_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    country_to_cluster = dict(zip(df["country"], df["cluster"]))

    missing_countries = [
        country for country in class_names
        if country not in country_to_cluster
    ]

    if missing_countries:
        raise ValueError(
            "These class_names are missing from the cluster CSV: "
            f"{missing_countries}"
        )

    num_classes = len(class_names)
    smoothing = torch.zeros(num_classes, num_classes, device=device)

    for i, country in enumerate(class_names):
        cluster = country_to_cluster[country]

        same_cluster_indices = [
            j for j, other_country in enumerate(class_names)
            if country_to_cluster[other_country] == cluster and j != i
        ]

        smoothing[i, i] = correct_prob

        if len(same_cluster_indices) > 0:
            rest_prob = 1.0 - correct_prob
            prob_per_country = rest_prob / len(same_cluster_indices)

            for j in same_cluster_indices:
                smoothing[i, j] = prob_per_country
        else:
            # If a country is alone in its cluster, keep it as one-hot.
            smoothing[i, i] = 1.0

    return smoothing


def soft_cross_entropy(predictions, soft_targets):
    """
    Cross entropy for soft labels.
    """
    log_probs = F.log_softmax(predictions, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def run_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler=None,
    train=True
):
    """
    Run one training or validation epoch.

    criterion can be either:
        - normal loss function, e.g. torch.nn.CrossEntropyLoss()
        - smoothing matrix tensor of shape [num_classes, num_classes]
    """

    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    running_corrects_top5 = 0
    n_samples = 0

    for images, labels in tqdm(dataloader, desc="Running epoch :) "):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    predictions = model(images)

                    if isinstance(criterion, torch.Tensor):
                        soft_targets = criterion[labels]
                        loss = soft_cross_entropy(predictions, soft_targets)
                    else:
                        loss = criterion(predictions, labels)

                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            else:
                predictions = model(images)

                if isinstance(criterion, torch.Tensor):
                    soft_targets = criterion[labels]
                    loss = soft_cross_entropy(predictions, soft_targets)
                else:
                    loss = criterion(predictions, labels)

                if train:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        n_samples += batch_size

        running_loss += loss.item() * batch_size

        _, best_pred = torch.max(predictions, 1)

        k = min(5, predictions.shape[1])
        _, top_5_preds = torch.topk(predictions, k, dim=1)

        running_corrects += torch.sum(best_pred == labels.data).item()
        running_corrects_top5 += (
            top_5_preds == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / n_samples
    epoch_acc_top5 = running_corrects_top5 / n_samples

    return epoch_loss, epoch_acc, epoch_acc_top5


def train_with_two_checkpoints(
    model,
    train_loader,
    val_loader,
    device,
    save_name,
    save_checkpoints=True,
    optimizer=None,
    criterion=None,
    num_epochs=50,
    patience=10,
    model_dir="country_cnn/outputs/models",
    class_names=None,
    use_cluster_smoothing=False,
    cluster_k=15,
    correct_prob=0.9,
    cluster_csv_path=None,
):
    """
    Training loop with two checkpoints.

    Saves:
        - best validation-loss checkpoint
        - best validation-accuracy checkpoint

    Early stopping:
        Stops only when BOTH validation loss and validation accuracy
        have not improved for `patience` epochs.

    Loss options:
        1. criterion is provided:
            uses the provided criterion.

        2. use_cluster_smoothing=True:
            creates cluster-based soft labels using cluster_k and correct_prob.

        3. otherwise:
            uses normal unweighted CrossEntropyLoss.
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_acc_top5": [],
        "val_acc_top5": [],
        "learning_rate": [],
    }

    if optimizer is None:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.00087,
            weight_decay=1e-5
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    if criterion is None:
        if use_cluster_smoothing:
            if class_names is None:
                raise ValueError(
                    "class_names must be provided when use_cluster_smoothing=True."
                )

            criterion = make_cluster_smoothing_matrix(
                class_names=class_names,
                device=device,
                cluster_k=cluster_k,
                correct_prob=correct_prob,
                csv_path=cluster_csv_path,
            )

            print(
                f"Using cluster smoothing: K={cluster_k}, "
                f"correct_prob={correct_prob}"
            )

        else:
            criterion = torch.nn.CrossEntropyLoss()
            print("Using normal CrossEntropyLoss")

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_val_acc = 0.0

    epochs_without_loss_improvement = 0
    epochs_without_acc_improvement = 0

    best_loss_path = model_dir / f"{save_name}_best_loss.pt"
    best_acc_path = model_dir / f"{save_name}_best_acc.pt"
    history_path = model_dir / f"{save_name}_history.csv"

    for epoch in range(num_epochs):
        train_loss, train_acc, train_acc_top5 = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            train=True
        )

        val_loss, val_acc, val_acc_top5 = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            train=False
        )

        scheduler.step()

        improved_loss = val_loss < best_val_loss
        improved_acc = val_acc > best_val_acc

        if improved_loss:
            best_val_loss = val_loss
            epochs_without_loss_improvement = 0

            if save_checkpoints:
                torch.save(model.state_dict(), best_loss_path)
                print("Saved best validation-loss model")
        else:
            epochs_without_loss_improvement += 1

        if improved_acc:
            best_val_acc = val_acc
            epochs_without_acc_improvement = 0

            if save_checkpoints:
                torch.save(model.state_dict(), best_acc_path)
                print("Saved best validation-accuracy model")
        else:
            epochs_without_acc_improvement += 1

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_acc_top5"].append(train_acc_top5)
        history["val_acc_top5"].append(val_acc_top5)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, top5: {train_acc_top5:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, top5: {val_acc_top5:.4f}"
        )

        if (
            epochs_without_loss_improvement >= patience
            and epochs_without_acc_improvement >= patience
        ):
            print(
                f"Early stopping after {epoch + 1} epochs. "
                f"Best val loss: {best_val_loss:.4f}, "
                f"Best val acc: {best_val_acc:.4f}"
            )
            break

    return history, best_loss_path, best_acc_path