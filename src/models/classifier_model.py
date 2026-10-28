"""Model that  use existing models from PyTorch https://pytorch.org/vision/stable/models.html

List of existing models in torchvision.models
    resnet18, mobilenet_v2, mobilenet_v3_large, efficientnet_b0, convnext_tiny, vgg11, vit_b_16, etc
INT8 quantized models are also available
    models.quantization.mobilenet_v3_large()

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are at least 224.
The input images are in the range [0, 1]

    from torchvision import transforms
    # assumption is object of interest is at the center of our input image
    transform = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )])
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .base_model import BaseModel


class ClassifierModel(BaseModel):
    """backbone: torchvision model whose classifier head ends in nn.Linear (not squeezenet)
    num_classes: num classes to predict/ num of outputs of network, exclusive to feat_extract
    feat_extract: replace the last Linear layer with Identity, so the model returns that layer's input
        features (e.g. 1280 for mobilenet_v2, 4096 for vgg11), exclusive to num_classes
    pretrained_weights: torchvision weights name, e.g. "MobileNet_V2_Weights.IMAGENET1K_V1", or None
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v2",
        num_classes: int = 10,
        feat_extract: bool = False,
        pretrained_weights: str = "MobileNet_V2_Weights.IMAGENET1K_V1",
        **kwargs,
    ):
        super().__init__()
        if feat_extract and num_classes:
            raise RuntimeError("feat_extract mode is not compatible when num_classes greater than 0")
        self.feat_extract = feat_extract
        self.model = models.get_model(backbone, weights=pretrained_weights)

        # replace only the last Linear layer (the ImageNet head), keeping the layers before it
        name, last = [(n, m) for n, m in self.model.named_modules() if isinstance(m, nn.Linear)][-1]
        new_head = nn.Identity() if feat_extract else nn.Linear(last.in_features, num_classes)
        self.model.set_submodule(name, new_head)

    def forward(self, x):
        x = self.model(x)
        return x if self.feat_extract else F.log_softmax(x, dim=1)
