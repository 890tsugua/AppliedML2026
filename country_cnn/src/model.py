import torch.nn as nn
from torchvision import models


def make_model(
    model_name="resnet34",
    num_classes=85,
    pretrained=True,
    trainable_layers=1
):
    """
    Create a torchvision ResNet model.

    Args:
        model_name: resnet18, resnet34, resnet50, ...
        num_classes: number of output classes
        pretrained: use ImageNet pretrained weights
        trainable_layers:
            0 = only fc
            1 = fc + layer4
            2 = fc + layer4 + layer3
            3 = fc + layer4 + layer3 + layer2
            4 = fc + layer4 + layer3 + layer2 + layer1
            5 = full model trainable
    """

    # get model constructor dynamically
    model_fn = getattr(models, model_name)

    # load weights or not
    if pretrained:
        model = model_fn(weights="DEFAULT")
    else:
        model = model_fn(weights=None)

    # replace classifier
    in_features = model.fc.in_features
    #model.fc = nn.Linear(in_features, num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    # freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # layers ordered from last to first
    layers_to_unfreeze = [
        model.fc,
        model.layer4,
        model.layer3,
        model.layer2,
        model.layer1,
    ]

    # unfreeze requested number of layers
    for layer in layers_to_unfreeze[:trainable_layers + 1]:
        for param in layer.parameters():
            param.requires_grad = True

    return model