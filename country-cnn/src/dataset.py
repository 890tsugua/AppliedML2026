from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def make_dataloaders_from_dir(data_dir, batch_size=32, image_size=224, val_split=0.2):
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), #changes training images to smaller sizes,
        transforms.RandomHorizontalFlip(),           #applies random horizontal flip for data augmentation during training
        transforms.ToTensor(),                       #converts images to PyTorch tensors for training
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), #changes validation images to smaller sizes
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform) #For every folder the folder name is the label, combines all the folders to one dataset

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # important: validation should use val_transform, not augmentation
    val_dataset.dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader