import pytest
import torch
from PIL import Image

from src.augmentations import ImagenetClassifierPreprocess, MnistPreprocess

####################################################################
# tests for functions from augmentations.classifier_transforms
####################################################################


def test_output_types(sample_pilimage):
    transformed_image = ImagenetClassifierPreprocess().train(sample_pilimage())
    assert isinstance(transformed_image, torch.Tensor)


def test_output_shapes(sample_pilimage):
    transformed_image = ImagenetClassifierPreprocess().train(sample_pilimage(244, 244, 3))
    assert transformed_image.shape == (3, 224, 224)


class TestMnistPreprocess:
    @pytest.fixture
    def preprocess(self):
        return MnistPreprocess()

    @pytest.fixture
    def example_image(self):
        # Create a dummy image (28x28, single channel)
        return Image.new('L', (28, 28))

    def test_initialization(self, preprocess):
        assert isinstance(preprocess, MnistPreprocess)

    def test_transform_non_equality(self, preprocess):
        assert preprocess.train is not preprocess.val
        assert preprocess.train is not preprocess.test

    def test_inference_matches_test_transform(self, preprocess):
        assert repr(preprocess.inference) == repr(preprocess.test)

    def test_output_type(self, preprocess, example_image):
        tensor = preprocess.train(example_image)
        assert isinstance(tensor, torch.Tensor)

    def test_output_shape(self, preprocess, example_image):
        tensor = preprocess.train(example_image)
        assert tensor.shape == (1, 28, 28)  # Channel, Height, Width

    def test_inference_output_shape(self, preprocess):
        # inference loads images as RGB, of any size; the model needs one 28x28 channel
        tensor = preprocess.inference(Image.new('RGB', (64, 48)))
        assert tensor.shape == (1, 28, 28)  # Channel, Height, Width
