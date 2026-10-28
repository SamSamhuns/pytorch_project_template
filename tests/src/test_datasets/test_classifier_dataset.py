import glob
import os.path as osp
import tarfile
from collections import Counter
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import ClassifierDataset, build_dataloaders, log_label_counts
from tests.conftest import NUM_CLS, NUM_IMGS_P_CLS


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


def test_webdataset_shards_keep_the_image_files(mock_webdataset_path, dump_mock_img_data_dir):
    """The converter stores each image file's bytes without re-encoding, shuffled so each shard mixes the classes"""
    stored = Counter()
    for shard in glob.glob(osp.join(mock_webdataset_path, "*.tar")):
        with tarfile.open(shard) as tar:
            files = {member.name: tar.extractfile(member).read() for member in tar}
        stored.update(data for name, data in files.items() if name.endswith(".jpg"))
        assert len({data for name, data in files.items() if name.endswith(".cls")}) == NUM_CLS
    images = glob.glob(osp.join(dump_mock_img_data_dir, "*", "*.jpg"))
    assert stored == Counter(Path(image).read_bytes() for image in images)


def test_webdataset_workers_read_every_sample_once(mock_webdataset_path):
    """Each DataLoader worker reads its own shards, and a worker without a shard reads nothing"""
    dataset = ClassifierDataset(
        train_transform=transforms.ToTensor(),
        data_mode="webdataset",
        root=osp.dirname(mock_webdataset_path),
        train_path=osp.basename(mock_webdataset_path),
        val_path=None,
        test_path=None,
    )
    loader = DataLoader(dataset.train_set, batch_size=10, num_workers=4)  # 3 shards of 50 images
    batches = [target for _, target in loader]

    assert len(dataset.train_set) == NUM_CLS * NUM_IMGS_P_CLS  # from info.json, without reading the images
    assert len(loader) == len(batches) == 15
    assert Counter(torch.cat(batches).tolist()) == dict.fromkeys(range(NUM_CLS), NUM_IMGS_P_CLS)
    assert dataset.train_set.classes == [f"class_{i}" for i in range(NUM_CLS)]


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


@pytest.mark.parametrize(
    "val_path, val_split, expected_excep",
    [(None, 0.1, None), ("/fake/val/path", 0, FileNotFoundError), ("/fake/val/path", 0.1, ValueError)],
)
def test_validation_split_configuration(basic_params, val_path, val_split, expected_excep):
    """The val set comes from a val dir or from validation_split, not both"""
    basic_params.update(val_path=val_path, validation_split=val_split)
    if expected_excep:
        with pytest.raises(expected_excep):
            ClassifierDataset(**basic_params)
    else:
        assert len(ClassifierDataset(**basic_params).val_set) == 15


def test_build_dataloaders(mock_classifierdataset):
    """A loader for each split; the train loader drops its last, smaller batch"""
    cfg = OmegaConf.create({"batch_size": 16, "num_workers": 0})
    train_loader, val_loader, test_loader = build_dataloaders(mock_classifierdataset, cfg)
    assert train_loader.drop_last
    assert len(val_loader.dataset) == len(test_loader.dataset) == NUM_CLS * NUM_IMGS_P_CLS

    mock_classifierdataset.test_set = None
    assert build_dataloaders(mock_classifierdataset, cfg)[2] is None


def test_build_dataloaders_lets_a_webdataset_shuffle_itself(basic_params, mock_webdataset_path):
    """A WebDataset is an IterableDataset, which the DataLoader must not be asked to shuffle"""
    basic_params.update(train_path=osp.basename(mock_webdataset_path), val_path=None, test_path=None)
    dataset = ClassifierDataset(data_mode="webdataset", **basic_params)
    train_loader, _, _ = build_dataloaders(dataset, OmegaConf.create({"batch_size": 16, "num_workers": 0}))
    data, target = next(iter(train_loader))
    assert len(data) == len(target) == 16


def test_log_label_counts(basic_params, caplog):
    """--verbose logs the size and label counts of each split, including a val split carved from the train dir"""
    basic_params.update(val_path=None, validation_split=0.2)
    with caplog.at_level("INFO"):
        log_label_counts(ClassifierDataset(**basic_params))
    assert "Train: 120 data points" in caplog.text
    assert "Val: 30 data points" in caplog.text
    assert "Val labels: [0 1 2 3 4]" in caplog.text
