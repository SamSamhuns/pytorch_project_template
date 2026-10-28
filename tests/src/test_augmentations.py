import torch
from PIL import Image

from src.augmentations import imagenet_transforms, mnist_transforms

####################################################################
# tests for functions from augmentations.classifier_transforms
####################################################################


def test_output_shapes(sample_pilimage):
    train_transform, eval_transform, _ = imagenet_transforms()
    image = sample_pilimage(244, 244, 3)
    assert train_transform(image).shape == (3, 224, 224)
    assert eval_transform(image).shape == (3, 224, 224)


def test_train_transform_augments(sample_pilimage):
    train_transform, _, _ = imagenet_transforms()
    image = sample_pilimage()
    assert not torch.equal(train_transform(image), train_transform(image))


def test_eval_transform_follows_the_weights():
    """V2 weights resize to 232 before the 224 crop; the standard ImageNet transform resizes to 256"""
    _, eval_transform, eval_settings = imagenet_transforms("ResNet50_Weights.IMAGENET1K_V2")
    assert eval_transform.resize_size == [232]
    assert eval_settings["resize_size"] == 232
    _, eval_transform, _ = imagenet_transforms(None)
    assert eval_transform.resize_size == [256]


####################################################################
# tests for functions from augmentations.mnist_transforms
####################################################################


def test_mnist_output_shape():
    train_transform, _, _ = mnist_transforms()
    assert train_transform(Image.new("L", (28, 28))).shape == (1, 28, 28)


def test_mnist_eval_takes_any_rgb_image():
    # inference loads images as RGB, of any size; the model needs one 28x28 channel
    _, eval_transform, _ = mnist_transforms()
    assert eval_transform(Image.new("RGB", (64, 48))).shape == (1, 28, 28)
