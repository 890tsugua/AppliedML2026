from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import pandas as pd

def run_epoch(model, dataloader, optimizer, criterion, device, scaler, train=True):
    """
    Run one epoch of training or validation. Returns training or validation loss for the epoch
    """

    if train: # If we are training, set to training mode
        model.train()
    else:
        model.eval() # If not to eval mode so gradients are not calculated

    running_loss = 0.0
    running_corrects = 0
    running_corrects_top5 = 0

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Running epoch :) ")):
        # Move to cuda or relevant device. 
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()   # Reset gradients. 
        
        with torch.set_grad_enabled(train):     # Compute gradients if training
            if device.type == "cuda":
                with torch.cuda.amp.autocast(): # For mixed precision training
                    predictions = model(images)
                    loss = criterion(predictions, labels)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                predictions = model(images)
                loss = criterion(predictions, labels)
                if train:
                    loss.backward()     # Backpropagate loss
                    optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item() * images.size(0)
        _, best_pred = torch.max(predictions, 1) # Returns index of max prediction (models best prediction)
        k = min(5, predictions.shape[1]) # In case there are less than 5 classes, adjust k accordingly
        _, top_5_preds = torch.topk(predictions, k, dim=1)
        running_corrects += torch.sum(best_pred == labels.data)
        running_corrects_top5 += (top_5_preds == labels.unsqueeze(1)).any(dim=1).sum()

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_acc_top5 = running_corrects_top5.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc, epoch_acc_top5


def train(model, train_loader, val_loader, device, save_name, save_checkpoints, class_weights=None, optimizer=None, criterion=None, num_epochs=100, patience=5):
    """
    
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_acc_top5": [],
        "val_acc_top5": [],
        "learning_rate": []
    }

    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00087017, weight_decay=1e-05) # From Optuna tuning
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    if class_weights == None:
        european_countries = {
        'Albania', 'Austria', 'Belgium', 'Bulgaria', 'Croatia',
        'Denmark', 'Estonia', 'Finland', 'France', 'Germany',
        'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy',
        'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
        'Montenegro', 'Netherlands', 'NorthMacedonia',
        'Norway', 'Poland', 'Portugal', 'Romania',
        'Russia', 'Serbia', 'Slovakia', 'Slovenia',
        'Spain', 'Sweden', 'Switzerland', 'Ukraine',
        'UnitedKingdom'
    }

    # Liste af lande
    class_names = ['Albania', 'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belgium', 'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'Colombia', 'Croatia', 'Denmark', 'DominicanRepublic', 'Ecuador', 'Estonia', 'Eswatini', 'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malaysia', 'Malta', 'Mexico', 'Mongolia', 'Montenegro', 'Netherlands', 'NewZealand', 'Nigeria', 'NorthMacedonia', 'Norway', 'Oman', 'Panama', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'SouthAfrica', 'SouthKorea', 'Spain', 'Sweden', 'Switzerland', 'Thailand', 'Tunisia', 'Turkey', 'USA', 'Uganda', 'Ukraine', 'UnitedArabEmirates', 'UnitedKingdom', 'Uruguay']
    #class_names = pd.read_csv("../country_to_idx.csv", header=None).iloc[:, 0].tolist()

    europe_weight = 1.5
    other_weight = 1.0

    class_weights = torch.tensor(
        [
            europe_weight if country in european_countries
            else other_weight
            for country in class_names
        ],
        dtype=torch.float32,
        device=device
    )

    if criterion == None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    model_dir = Path("country_cnn/outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_accuracy_top5 = run_epoch(model, train_loader, optimizer, criterion, device, scaler, train=True)
        val_loss, val_accuracy, val_accuracy_top5 = run_epoch(model, val_loader, optimizer, criterion, device, scaler, train=False)
        scheduler.step() # Step the learning rate scheduler
        

        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if save_checkpoints:
                best_model_path = model_dir / f"{save_name}.pt"
                torch.save(
                    model.state_dict(), best_model_path)
                print("Saved best model")
        else:
            epochs_without_improvement += 1
    

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_accuracy.item())
        history["val_acc"].append(val_accuracy.item())
        history["train_acc_top5"].append(train_accuracy_top5.item())
        history["val_acc_top5"].append(val_accuracy_top5.item())
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_accuracy:.4f}, top5: {train_accuracy_top5:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_accuracy:.4f}, top5: {val_accuracy_top5:.4f}"
        )

        # Print history to a file after each epoch
        history_path = model_dir / f"{save_name}_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping after {epoch+1} epochs. "
                f"Best validation loss: {best_val_loss:.4f}"
            )
            break

    return history