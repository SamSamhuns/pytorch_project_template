import os.path as osp

import pytest

from src.datasets import ClassifierDataset


def test_initialization_imgs(mock_classifierdataset):
    """Test initialization with image folders."""
    assert mock_classifierdataset.train_set is not None, "Training set should be initialized."
    assert mock_classifierdataset.val_set is not None, "Val set should be initialized."
    assert mock_classifierdataset.test_set is not None, "Test set should be initialized."


def test_initialization_webdataset(basic_params, mock_webdataset_path):
    """Test initialization with webdataset."""
    wdset_path = mock_webdataset_path
    basic_params["train_path"] = osp.basename(wdset_path)
    basic_params["val_path"] = None
    basic_params["test_path"] = None
    dataset = ClassifierDataset(data_mode="webdataset", **basic_params)
    assert dataset.train_set is not None, "Training set should be initialized."


def test_split_missing_a_class_dir_raises(tmp_path, create_and_save_dummy_imgs):
    """A test split without class_2 would map its classes to different indices than train"""
    create_and_save_dummy_imgs(str(tmp_path / "train"), n_cls=3, n_imgs_p_cls=2)
    create_and_save_dummy_imgs(str(tmp_path / "test"), n_cls=2, n_imgs_p_cls=2)
    with pytest.raises(ValueError, match="differ from the train classes"):
        ClassifierDataset(root=str(tmp_path), train_path="train", val_path=None, test_path="test")


def test_val_split_uses_eval_transform(tmp_path, create_and_save_dummy_imgs):
    """A val set carved out of the train dir must get the eval transform, not the train augmentation"""
    create_and_save_dummy_imgs(str(tmp_path / "train"), n_cls=2, n_imgs_p_cls=10)
    dataset = ClassifierDataset(
        train_transform=lambda img: "augmented",
        eval_transform=lambda img: "eval",
        root=str(tmp_path),
        val_path=None,
        test_path=None,
        validation_split=0.2,
    )

    assert {image for image, _ in dataset.train_set} == {"augmented"}
    assert {image for image, _ in dataset.val_set} == {"eval"}
    assert len(dataset.val_set) == 4
    assert sorted(dataset.train_set.indices + dataset.val_set.indices) == list(range(20))


def test_non_implemented_mode(basic_params):
    """Test initialization with a non-implemented mode."""
    with pytest.raises(NotImplementedError):
        _ = ClassifierDataset(data_mode="numpy", **basic_params)
