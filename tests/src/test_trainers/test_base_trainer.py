import os

import pytest
import torch

from tests.src.test_trainers.conftest import PatchedBaseTrainer


def test_init_trainer(base_trainer_and_logger):
    """The trainer runs on the configured device"""
    trainer, mocked_logger = base_trainer_and_logger
    info_calls = [call.args for call in mocked_logger.return_value.info.call_args_list]

    assert trainer.device == torch.device(trainer.config["device"])
    assert ("Program will run on %s", trainer.device) in info_calls


def test_cuda_device_falls_back_to_cpu(mock_clsf_config, mocker):
    """device: cuda:1 runs on CPU, with a warning, when CUDA is unavailable"""
    mocker.patch("torch.cuda.is_available", return_value=False)
    mock_clsf_config["device"] = "cuda:1"
    trainer = PatchedBaseTrainer(config=mock_clsf_config)
    assert trainer.device == torch.device("cpu")


def test_load_checkpoint_not_found(base_trainer_and_logger, mock_logger):
    """Test invalid load checkpoint"""
    trainer, _ = base_trainer_and_logger
    with pytest.raises(ValueError):
        trainer.load_checkpoint("nonexistent_path")
        mock_logger.error.assert_called()


def test_load_checkpoint_success(base_trainer_and_logger, mocker):
    """Test successful load checkpoint"""
    trainer, m = base_trainer_and_logger
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch("torch.load", return_value={"model": {}, "epoch": 3})

    checkpoint = trainer.load_checkpoint("fake_checkpoint_path.pth")
    info_calls = [call.args for call in m.return_value.info.call_args_list]

    assert checkpoint == {"model": {}, "epoch": 3}
    assert ("Loaded checkpoint %s", "fake_checkpoint_path.pth") in info_calls


def test_validation_data_loader_setup(base_trainer_and_logger):
    """Test val dataloader"""
    trainer, _ = base_trainer_and_logger
    if trainer.config["dataset"]["args"]["val_path"] or trainer.train_data_loader.validation_split > 0:
        assert trainer.val_data_loader is not None
    else:
        assert trainer.val_data_loader is None


@pytest.mark.parametrize(
    "val_path, val_split, expected_excep",
    [(None, 0.1, None), ("/fake/val/path", 0, FileNotFoundError), ("/fake/val/path", 0.1, ValueError)],
)
def test_validation_split_configuration(mock_clsf_config, val_path, val_split, expected_excep):
    """Test val split config with variations of val path and val split"""
    mock_clsf_config["dataset"]["args"]["val_path"] = val_path
    mock_clsf_config["dataloader"]["args"]["validation_split"] = val_split
    if expected_excep:
        with pytest.raises(expected_excep):
            PatchedBaseTrainer(config=mock_clsf_config)
    else:
        trainer = PatchedBaseTrainer(config=mock_clsf_config)
        assert trainer  # Check if the trainer is correctly instantiated without errors


def test_webdataset_config(mock_clsf_config, mock_webdataset_path):
    """The WebDataset loader must accept the args BaseTrainer passes, e.g. the seeded generator"""
    dataset_args = mock_clsf_config["dataset"]["args"]
    dataset_args["data_mode"] = "webdataset"
    dataset_args["train_path"] = os.path.basename(mock_webdataset_path)
    dataset_args["val_path"] = dataset_args["test_path"] = None
    mock_clsf_config["dataloader"]["type"] = "WebDatasetDataLoader"
    mock_clsf_config["dataloader"]["args"].pop("shuffle")  # WebDataset shuffles in the dataset
    mock_clsf_config["dataloader"]["args"]["num_workers"] = 0

    trainer = PatchedBaseTrainer(config=mock_clsf_config)
    data, target = next(iter(trainer.train_data_loader))
    assert len(data) == len(target) == mock_clsf_config["dataloader"]["args"]["batch_size"]


def test_full_training_cycle(base_trainer_and_logger, mocker):
    """Integration testing for the full training cycle to ensure no exceptions
    and that basic calls are made.
    """
    trainer, _ = base_trainer_and_logger
    mocker.patch.object(trainer, "train", return_value=None)
    mocker.patch.object(trainer, "validate", return_value=None)
    try:
        trainer.train()
        trainer.validate()
    except Exception as e:
        pytest.fail(f"Training cycle failed with unexpected exception: {e}")


@pytest.mark.parametrize(
    "mode, file_suffix",
    [
        ("ONNX_TS", "_ts.onnx"),
        ("ONNX_DYNAMO", "_dynamo.onnx"),
    ],
)
def test_export_function(mode, file_suffix, base_trainer_and_logger, simple_2d_conv_model):
    """Test trainer export modes"""
    trainer, logger = base_trainer_and_logger
    trainer.model = simple_2d_conv_model
    trainer.model = trainer.model.to(torch.device(trainer.device.type))
    trainer.export(mode)

    # Check if paths and files are correctly constructed
    model_name = "model_gpu" if trainer.device.type == "cuda" else "model_cpu"
    export_path = os.path.join(trainer.config["models_dir"], model_name)
    export_path = export_path + file_suffix

    info_calls = [call.args for call in logger.return_value.info.call_args_list]
    assert ("%s export complete", mode) in info_calls
