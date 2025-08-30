import os
import warnings

import pytest
import torch

from tests.src.test_trainers.conftest import PatchedBaseTrainer


def test_init_trainer(base_trainer_and_logger):
    """Test if trainer is initialized properly with both cpu & gpu mode."""
    trainer, mocked_logger = base_trainer_and_logger
    logger_instance = mocked_logger.return_value
    info_calls = [call.args for call in logger_instance.info.call_args_list]

    gpu_dev = trainer.config["gpu_device"]
    if gpu_dev and not torch.cuda.is_available():
        warnings.warn("CUDA device not found, GPU testing has been skipped")
        assert ("Program will run on CPU",) in info_calls
    elif gpu_dev is None:
        assert ("Program will run on CPU",) in info_calls
    else:
        gpu_dev = [gpu_dev] if isinstance(gpu_dev, int) else gpu_dev
        device = torch.device("cuda", gpu_dev[0])
        assert ("Program will run on GPU device %s", device) in info_calls


def test_load_checkpoint_not_found(base_trainer_and_logger, mock_logger):
    """Test invalid load checkpoint"""
    trainer, _ = base_trainer_and_logger
    with pytest.raises(ValueError):
        trainer.load_checkpoint(torch.tensor([1]), "nonexistent_path")
        mock_logger.error.assert_called()


def test_save_checkpoint(base_trainer_and_logger, mocker):
    """Test successful save checkpoint"""
    mock_torch_save = mocker.patch('torch.save')
    model_mock = mocker.Mock()
    trainer, _ = base_trainer_and_logger
    trainer.save_checkpoint(model_mock)
    mock_torch_save.assert_called_once()


def test_load_checkpoint_success(base_trainer_and_logger, mocker):
    """Test successful load checkpoint"""
    trainer, m = base_trainer_and_logger
    mocker.patch('os.path.isfile', return_value=True)
    mocker.patch('torch.load', return_value={'state_dict': mocker.Mock()})
    model_mock = mocker.Mock()

    trainer.load_checkpoint(model_mock, "fake_checkpoint_path.pth")
    info_calls = [call.args for call in m.return_value.info.call_args_list]

    model_mock.load_state_dict.assert_called()
    assert ("Loaded checkpoint %s", "fake_checkpoint_path.pth") in info_calls


def test_validation_data_loader_setup(base_trainer_and_logger):
    """Test val dataloader"""
    trainer, _ = base_trainer_and_logger
    if (trainer.config["dataset"]["args"]["val_path"] or 
            trainer.train_data_loader.validation_split > 0):
        assert trainer.val_data_loader is not None
    else:
        assert trainer.val_data_loader is None


@pytest.mark.parametrize("val_path, val_split, expected_excep", [
    (None, 0.1, None),
    ("/fake/val/path", 0, FileNotFoundError),
    ("/fake/val/path", 0.1, ValueError)
])
def test_validation_split_configuration(mock_clsf_config, mock_logger, val_path, val_split, expected_excep, mocker):
    """Test val split config with variations of val path and val split"""
    mock_clsf_config["dataset"]["args"]["val_path"] = val_path
    mock_clsf_config["dataloader"]["args"]["validation_split"] = val_split
    if expected_excep:
        with pytest.raises(expected_excep):
            # Patch the logger to use the mocked logger for capturing outputs
            mocker.patch('src.loggers.get_logger', return_value=mock_logger)
            PatchedBaseTrainer(config=mock_clsf_config)
    else:
        mocker.patch('src.loggers.get_logger', return_value=mock_logger)
        trainer = PatchedBaseTrainer(config=mock_clsf_config)
        assert trainer  # Check if the trainer is correctly instantiated without errors


def test_full_training_cycle(base_trainer_and_logger, mocker):
    """Integration testing for the full training cycle to ensure no exceptions
    and that basic calls are made.
    """
    trainer, _ = base_trainer_and_logger
    mocker.patch.object(trainer, 'train', return_value=None)
    mocker.patch.object(trainer, 'validate', return_value=None)
    try:
        trainer.train()
        trainer.validate()
    except Exception as e:
        pytest.fail(f"Training cycle failed with unexpected exception: {e}")


@pytest.mark.parametrize("mode, file_suffix", [
    ("ONNX_TS", "_ts.onnx"),
    ("ONNX_DYNAMO", "_dynamo.onnx"),
    ("TS_TRACE", "_traced.pt"),
    ("TS_SCRIPT", "_scripted.pt")
])
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
    assert ('%s export complete', mode) in info_calls
