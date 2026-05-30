# src/evaluate.py

import torch
from sklearn.metrics import classification_report, confusion_matrix

 
def load_model(model, checkpoint_path, device):
    """
    Load saved model weights.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, dataloader, device):
    """
    Run model on a dataloader and collect predictions, labels and probabilities.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    return (
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_probs),
    )


def topk_accuracy(probs, labels, k=5):
    """
    Compute top-k accuracy from probabilities.
    """
    k = min(k, probs.shape[1])
    topk_preds = probs.topk(k, dim=1).indices
    correct = topk_preds.eq(labels.view(-1, 1)).any(dim=1)
    return correct.float().mean().item()


def evaluate_model(model, dataloader, device, class_names=None):
    """
    Full evaluation: top1, top5, classification report, confusion matrix.
    """
    preds, labels, probs = predict(model, dataloader, device)

    top1 = (preds == labels).float().mean().item()
    top5 = topk_accuracy(probs, labels, k=5)

    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-5 accuracy: {top5:.4f}")

    if class_names is not None:
        print("\nClassification report:")
        print(classification_report(labels, preds, target_names=class_names))
    else:
        print("\nClassification report:")
        print(classification_report(labels, preds))

    cm = confusion_matrix(labels, preds)

    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True
    )

    return {
        "top1": top1,
        "top5": top5,
        "preds": preds,
        "labels": labels,
        "probs": probs,
        "confusion_matrix": cm,
        "classification_report": report
    }