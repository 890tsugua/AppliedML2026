import torch.nn as nn
from torchvision import models


def make_model(
    model_name="resnet18",
    num_classes=85,
    pretrained=True
):
    """
    Create a torchvision ResNet model.

    Args:
        model_name: resnet18, resnet34, resnet50, ...
        num_classes: number of output classes. Default 85.
        pretrained: whether to use ImageNet pretrained weights. Default True
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
    model.fc = nn.Linear(in_features, num_classes) 
    # fc "fully connected" layer is the last layer of the model, which does classification.

    return model