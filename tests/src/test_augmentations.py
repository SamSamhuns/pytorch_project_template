import torch
from PIL import Image

from src.augmentations import ImagenetClassifierPreprocess, MnistPreprocess

####################################################################
# tests for functions from augmentations.classifier_transforms
####################################################################


def test_output_shapes(sample_pilimage):
    preprocess = ImagenetClassifierPreprocess()
    image = sample_pilimage(244, 244, 3)
    assert preprocess.train_transform(image).shape == (3, 224, 224)
    assert preprocess.eval_transform(image).shape == (3, 224, 224)


def test_train_transform_augments(sample_pilimage):
    train_transform = ImagenetClassifierPreprocess().train_transform
    image = sample_pilimage()
    assert not torch.equal(train_transform(image), train_transform(image))


def test_eval_transform_follows_the_weights():
    """V2 weights resize to 232 before the 224 crop; the standard ImageNet transform resizes to 256"""
    assert ImagenetClassifierPreprocess("ResNet50_Weights.IMAGENET1K_V2").eval_transform.resize_size == [232]
    assert ImagenetClassifierPreprocess(None).eval_transform.resize_size == [256]


####################################################################
# tests for functions from augmentations.mnist_transforms
####################################################################


def test_mnist_output_shape():
    assert MnistPreprocess().train_transform(Image.new("L", (28, 28))).shape == (1, 28, 28)


def test_mnist_eval_takes_any_rgb_image():
    # inference loads images as RGB, of any size; the model needs one 28x28 channel
    assert MnistPreprocess().eval_transform(Image.new("RGB", (64, 48))).shape == (1, 28, 28)
