"""
Model that  use existing models from PyTorch https://pytorch.org/vision/stable/models.html

List of existing models in torchvision.models
    squeezenet1_0, densenet161, inception_v3, googlenet, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, etc
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
from collections import OrderedDict
import torchvision.models as models
import torch.nn as nn
import torch
from .base_model import BaseModel


class ClassifierModel(BaseModel):
    """
    backbone: network to extract features
    num_classes: num classes to predict/ num of outputs of network, exclusive to feat_extract
    feat_extract: only use network for feat extract, exclusive to num_classes
    pretrained: use weights pretrained from imagenet
    """
    def __init__(self,
                 backbone: str = "mobilenet_v2",
                 num_classes: int = 10,
                 feat_extract: bool = False,
                 pretrained_weights: str = "MobileNet_V2_Weights.IMAGENET1K_V1",
                 **kwargs):
        super().__init__()
        self.backbone = getattr(models, backbone)(weights=pretrained_weights)

        if feat_extract and (num_classes > 0 and num_classes is not None):
            raise RuntimeError(
                "feat_extract mode is not compatible when num_classes greater than 0")

        final_layer_name = None
        if hasattr(self.backbone, 'classifier'):
            final_layer_name = 'classifier'
        elif hasattr(self.backbone, 'fc'):
            final_layer_name = 'fc'
        else:
            print("custom backbone")
            self.model = self.backbone
            return

        # use net as a feat extractor by setting final layer as Identity
        if feat_extract:
            setattr(self.backbone, final_layer_name, nn.Identity())
        # use net as classifier
        else:
            # self.backbone.fc.in_features
            final_layer = getattr(self.backbone, final_layer_name)
            final_layer = final_layer[-1] if isinstance(final_layer, nn.Sequential) else final_layer
            n_inputs = getattr(final_layer, "in_features")
            classifier = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=0.2, inplace=False)),
                ('fc1', nn.Linear(n_inputs, num_classes)),
                ('softmax', nn.LogSoftmax(dim=1))
            ]))
            setattr(self.backbone, final_layer_name, classifier)
        self.model = self.backbone

    def forward(self, x):
        x = self.model(x)
        return x


def main():
    model = Classifier(train_mode=False).eval()
    model_input = torch.rand([1, 3, 244, 244])
    model_output = model(model_input)
    print(model_output.shape)


if __name__ == "__main__":
    main()
