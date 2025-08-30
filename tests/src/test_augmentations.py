import numpy as np
import pytest
import torch
from PIL import Image

from src.augmentations import ImagenetClassifierPreprocess, MnistPreprocess, NumerizeLabels

####################################################################
# tests for functions from augmentations.numeric
####################################################################


def test_numerize_labels_without_stable_sort():
    """Testing with a basic list of labels"""
    labels = ['cat', 'dog', 'bird']
    numerizer = NumerizeLabels(labels, use_stable_sort=False)

    # Check if all labels are correctly numerized
    assert numerizer('cat') == 0, "Numerization of 'cat' failed"
    assert numerizer('dog') == 1, "Numerization of 'dog' failed"
    assert numerizer('bird') == 2, "Numerization of 'bird' failed"


def test_numerize_labels_with_stable_sort():
    """Test the order maintained by stable_sort
    stable sort behavior:
    ['1', '10', '2'] would be sorted as ['1', '2', '10'],
    """
    # Assuming stable_sort sorts numbers in ascending order and strings alphabetically
    labels = ['cat', 'dog', 'apple', 10, 2, 'banana', 1]
    numerizer = NumerizeLabels(labels)
    sorted_label_dict = {1: 0, 2: 1, 10: 2, 'apple': 3, 'banana': 4, 'cat': 5, 'dog': 6}

    for label, index in sorted_label_dict.items():
        assert numerizer(label) == index, f"Numerization or order of {label} failed"


def test_numerize_labels_with_numpy_array():
    """Test initialization with a NumPy array"""
    labels = np.array(['cat', 'dog', 'bird'])
    numerizer = NumerizeLabels(labels)

    assert numerizer('bird') == 0, "Numerization of 'bird' failed"
    assert numerizer('cat') == 1, "Numerization of 'cat' failed"
    assert numerizer('dog') == 2, "Numerization of 'dog' failed"


def test_numerize_labels_error_on_unknown_label():
    """Test error raised when an unknown label is passed to the numerizer"""
    labels = ['cat', 'dog', 'bird']
    numerizer = NumerizeLabels(labels)

    with pytest.raises(KeyError):
        # Trying to numerize an unlisted label should raise KeyError
        numerizer('fish')


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
    @pytest.fixture(scope="class")
    def preprocess(self):
        return MnistPreprocess()

    @pytest.fixture(scope="class")
    def example_image(self):
        # Create a dummy image (28x28, single channel)
        return Image.new('L', (28, 28))

    def test_initialization(self, preprocess):
        assert isinstance(preprocess, MnistPreprocess)

    def test_transform_non_equality(self, preprocess):
        assert preprocess.train is not preprocess.val
        assert preprocess.train is not preprocess.test

    def test_inference_transform_difference(self, preprocess):
        assert preprocess.inference != preprocess.train

    def test_output_type(self, preprocess, example_image):
        tensor = preprocess.train(example_image)
        assert isinstance(tensor, torch.Tensor)

    def test_output_shape(self, preprocess, example_image):
        tensor = preprocess.train(example_image)
        assert tensor.shape == (1, 28, 28)  # Channel, Height, Width

    def test_inference_output_shape(self, preprocess, example_image):
        # Resize should ensure the image is 28x28 even if the input is not
        tensor = preprocess.inference(example_image)
        assert tensor.shape == (1, 28, 28)  # Channel, Height, Width
