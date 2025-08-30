import warnings

import pytest
import torch

from src.config_parser import CustomDictConfig
from src.trainers import ClassifierTrainer


def test_init_trainer(clsf_trainer_and_logger):
    """Test if trainer is initialized properly with both cpu & gpu mode."""
    trainer, mocked_logger = clsf_trainer_and_logger
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

    assert isinstance(trainer.model, torch.nn.Module)
    assert isinstance(trainer.loss, torch.nn.Module)
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LRScheduler)
    assert isinstance(trainer.scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_train_one_epoch_method(clsf_trainer_and_logger):
    trainer, mocked_logger = clsf_trainer_and_logger
    initial_params = [p.clone() for p in trainer.model.parameters()]

    trainer.train_one_epoch()

    final_params = list(trainer.model.parameters())
    param_changes = any(not torch.equal(ip, fp)
                        for ip, fp in zip(initial_params, final_params, strict=True))
    assert param_changes, "Model parameters did not change after training epoch."

    logger_instance = mocked_logger.return_value
    info_calls = [call.args for call in logger_instance.info.call_args_list]
    assert any("Train Epoch" in str(call)
               for call in info_calls), "Training log entry missing"
    assert any("Loss:" in str(call)
               for call in info_calls), "No logging for Loss"


def test_eval_one_epoch_method(mock_clsf_config: CustomDictConfig):
    """Test eval_one_epoch"""
    mock_clsf_config["dataloader"]["args"]["validation_split"] = 0.1
    mock_clsf_config["dataset"]["args"]["val_path"] = None
    trainer = ClassifierTrainer(config=mock_clsf_config,
                                logger_name="time_series_clsf_logger")
    trainer.model.eval()
    loader = trainer.val_data_loader

    test_loss, y_true, y_score, y_pred = trainer.eval_one_epoch(loader)

    assert isinstance(test_loss, float), \
        "Loss is not calculated as a float."
    assert len(y_true) == len(y_score) == len(y_pred), \
        "Mismatch in output lengths of targets and predictions."


def test_train_method(clsf_trainer_and_logger):
    trainer, mocked_logger = clsf_trainer_and_logger
    initial_epoch = trainer.current_epoch

    trainer.train()

    assert trainer.current_epoch > initial_epoch, "No epochs were completed."
    logger_instance = mocked_logger.return_value
    assert logger_instance.info.call_count > 0, "No logging occurred during training."


def test_validate_method(mock_clsf_config: CustomDictConfig, mocker):
    mocked_logger = mocker.patch('logging.getLogger')
    mock_clsf_config["dataset"]["args"]["val_path"] = None
    mock_clsf_config["dataloader"]["args"]["validation_split"] = 0.1
    trainer = ClassifierTrainer(config=mock_clsf_config,
                                logger_name="time_series_clsf_logger")
    if trainer.val_data_loader is not None:
        trainer.validate()
        logger_instance = mocked_logger.return_value
        info_calls = [
            call.args for call in logger_instance.info.call_args_list]
        assert any("Validation" in str(call)
                   for call in info_calls), "Validation log entry missing"
        assert any("loss" in str(call)
                   for call in info_calls), "No logging for validation loss"
    else:
        pytest.skip("No validation data loader configured for this test.")


def test_test_method(clsf_trainer_and_logger):
    trainer, mocked_logger = clsf_trainer_and_logger
    trainer.test()

    logger_instance = mocked_logger.return_value
    info_calls = [call.args for call in logger_instance.info.call_args_list]
    assert any("Test set:" in str(call)
               for call in info_calls), "Test log entry missing"
