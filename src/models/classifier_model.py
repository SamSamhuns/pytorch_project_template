"""Classifiers from torchvision's models, https://pytorch.org/vision/stable/models.html
e.g. resnet18, mobilenet_v2, mobilenet_v3_large, efficientnet_b0, convnext_tiny, vgg11 or vit_b_16
"""

import torchvision
from torch import nn


def torchvision_classifier(
    backbone: str = "mobilenet_v2",
    num_classes: int = 10,
    feat_extract: bool = False,
    pretrained_weights: str | None = "MobileNet_V2_Weights.IMAGENET1K_V1",
) -> nn.Module:
    """A torchvision model whose last Linear layer is replaced to predict num_classes classes.
    backbone: a torchvision model whose classifier head ends in nn.Linear (not squeezenet)
    feat_extract: replace the last Linear layer with Identity instead, so the model returns that layer's input
        features (e.g. 1280 for mobilenet_v2, 4096 for vgg11); exclusive to num_classes
    pretrained_weights: torchvision weights name, e.g. "MobileNet_V2_Weights.IMAGENET1K_V1", or None
    """
    if feat_extract and num_classes:
        raise RuntimeError("feat_extract mode is not compatible when num_classes greater than 0")
    model = torchvision.models.get_model(backbone, weights=pretrained_weights)

    # replace only the last Linear layer (the ImageNet head), keeping the layers before it
    name, last = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    model.set_submodule(name, nn.Identity() if feat_extract else nn.Linear(last.in_features, num_classes))
    return model  # it returns logits, or features with feat_extract
