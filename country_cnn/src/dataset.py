from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch


def make_dataloaders_from_dir(data_dir, batch_size=32, image_size=224, val_split=0.2,
                              num_workers=8, 
                              pin_memory=True, 
                              prefetch_factor=4,
                              persistent_workers=True,
                              random_crop=True,
                              color_jitter=True,
                              rotation=True,
                              horizontal_flip=True):
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33)
        ) if random_crop else None,#transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip() if horizontal_flip else None,
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ) if color_jitter else None,
        transforms.RandomRotation(5) if rotation else None,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Remove none transforms if there are any
    train_transform.transforms = [t for t in train_transform.transforms if t is not None]

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), #changes validation images to smaller sizes
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform) #For every folder the folder name is the label, combines all the folders to one dataset

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # important: validation should use val_transform, not augmentation
    val_dataset.dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=pin_memory, 
                              prefetch_factor=prefetch_factor,
                              persistent_workers=persistent_workers)
    
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=pin_memory, 
                            prefetch_factor=prefetch_factor,
                            persistent_workers=persistent_workers)

    return train_loader, val_loader

def make_test_dataloader_from_dir(data_dir, batch_size=32, image_size=224):
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    return test_loader


def show_image(image, label=None, prediction=None, class_names=None):
    """
    Display a tensor image with optional truth/prediction labels.

    Args:
        image:
            PyTorch tensor of shape [C, H, W]

        label:
            Ground truth class index

        prediction:
            Predicted class index

        class_names:
            List mapping class indices to names
    """

    # move channel dimension last
    image = image.permute(1, 2, 0).cpu()

    # matplotlib expects values in [0,1]
    image = image.clamp(0, 1)

    plt.imshow(image)

    title = []

    if label is not None:
        if class_names is not None:
            label = class_names[label]
        title.append(f"True: {label}")

    if prediction is not None:
        if class_names is not None:
            prediction = class_names[prediction]
        title.append(f"Pred: {prediction}")

    plt.title(" | ".join(title))
    plt.axis("off")
    plt.show()