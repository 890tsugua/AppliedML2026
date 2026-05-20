from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def run_epoch(model, dataloader, optimizer, criterion, device, train=True):
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

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Description")):
        # Move to cuda or relevant device. 
        images = images.to(device)
        labels = labels.to(device)  

        if train:
            optimizer.zero_grad()   # Reset gradients.
        
        with torch.set_grad_enabled(train):     # Compute gradients if training
            predictions = model(images)
            loss = criterion(predictions, labels)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, best_pred = torch.max(predictions, 1) # Returns index of max prediction (models best prediction)
        _, top_5_preds = torch.topk(predictions, 5, dim=1)
        running_corrects += torch.sum(best_pred == labels.data)
        running_corrects_top5 += (top_5_preds == labels.unsqueeze(1)).any(dim=1).sum()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_acc_top5 = running_corrects_top5.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc, epoch_acc_top5


def train(model, train_loader, val_loader, device, optimizer=None, criterion=None, num_epochs=100):
    """
    
    """
    train_losses = []
    val_losses = []

    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if criterion == None:
        criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss, train_accuracy = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_accuracy = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        train_losses.append(train_loss)     # Append losses for plotting training history
        val_losses.append(val_loss)

    
    return train_losses, val_losses