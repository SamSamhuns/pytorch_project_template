import os.path as osp
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchvision

from src.config_parser import CustomDictConfig
from src.metrics import calc_metric
from src.models import init_model
from src.trainers import ClassifierTrainer


def test_init_trainer(clsf_trainer_and_logger):
    """Test if trainer is initialized properly with both cpu & gpu mode."""
    trainer, mocked_logger = clsf_trainer_and_logger
    logger_instance = mocked_logger.return_value
    info_calls = [call.args for call in logger_instance.info.call_args_list]

    gpu_dev = trainer.config["gpu_device"]
    if gpu_dev and not torch.cuda.is_available():
        warnings.warn("CUDA device not found, GPU testing has been skipped", stacklevel=2)
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
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_train_one_epoch_method(clsf_trainer_and_logger):
    trainer, mocked_logger = clsf_trainer_and_logger
    initial_params = [p.clone() for p in trainer.model.parameters()]

    trainer.train_one_epoch()

    final_params = list(trainer.model.parameters())
    param_changes = any(not torch.equal(ip, fp) for ip, fp in zip(initial_params, final_params, strict=True))
    assert param_changes, "Model parameters did not change after training epoch."

    logger_instance = mocked_logger.return_value
    info_calls = [call.args for call in logger_instance.info.call_args_list]
    assert any("Train Epoch" in str(call) for call in info_calls), "Training log entry missing"
    assert any("Loss:" in str(call) for call in info_calls), "No logging for Loss"


def test_eval_one_epoch_method(mock_clsf_config: CustomDictConfig):
    """Test eval_one_epoch"""
    mock_clsf_config["dataloader"]["args"]["validation_split"] = 0.1
    mock_clsf_config["dataset"]["args"]["val_path"] = None
    trainer = ClassifierTrainer(config=mock_clsf_config, logger_name="time_series_clsf_logger")
    trainer.model.eval()
    loader = trainer.val_data_loader

    test_loss, y_true, y_score, y_pred = trainer.eval_one_epoch(loader)

    assert isinstance(test_loss, float), "Loss is not calculated as a float."
    assert len(y_true) == len(y_score) == len(y_pred), "Mismatch in output lengths of targets and predictions."


def test_train_method(clsf_trainer_and_logger):
    trainer, mocked_logger = clsf_trainer_and_logger
    initial_epoch = trainer.current_epoch

    trainer.train()

    assert trainer.current_epoch > initial_epoch, "No epochs were completed."
    assert osp.isfile(osp.join(trainer.config["models_dir"], "last.pth"))
    logger_instance = mocked_logger.return_value
    assert logger_instance.info.call_count > 0, "No logging occurred during training."


def test_pretrained_weights_survive_trainer_init(clsf_trainer_and_logger):
    """The trainer must not re-initialize a pretrained backbone"""
    trainer, _ = clsf_trainer_and_logger
    pretrained = torchvision.models.get_model("mobilenet_v2", weights="MobileNet_V2_Weights.IMAGENET1K_V1")

    expected = pretrained.features.state_dict()
    actual = trainer.model.model.features.state_dict()
    assert all(torch.equal(actual[k].cpu(), v) for k, v in expected.items())


def test_best_checkpoint_keeps_highest_val_accuracy(clsf_trainer_and_logger, mocker):
    """Val accuracies 90% -> 50% -> 60% must save best.pth once, for the 90%"""
    trainer, _ = clsf_trainer_and_logger
    save_checkpoint = mocker.patch.object(trainer, "save_checkpoint")
    y_true = np.zeros(10, dtype=int)
    for n_wrong in [1, 5, 4]:
        y_pred = (np.arange(10) < n_wrong).astype(int)
        mocker.patch.object(trainer, "eval_one_epoch", return_value=(0.5, y_true, y_pred.astype(float), y_pred))
        trainer.validate()

    save_checkpoint.assert_called_once_with(trainer.model, "best.pth")


def test_eval_loss_is_mean_over_samples(clsf_trainer_and_logger):
    """Batches of unequal size must be weighted by their size"""
    trainer, _ = clsf_trainer_and_logger
    trainer.model = torch.nn.Identity()  # the batches below already hold the model's log-probabilities
    log_probs = torch.randn(4, 5).log_softmax(dim=1)
    target = torch.tensor([0, 1, 2, 3])

    loss, _, _, _ = trainer.eval_one_epoch([(log_probs[:3], target[:3]), (log_probs[3:], target[3:])])
    assert loss == pytest.approx(F.nll_loss(log_probs, target).item())


def test_binary_y_score_is_positive_class_probability(clsf_trainer_and_logger):
    """A perfect binary classifier must get ROC-AUC 1.0"""
    trainer, _ = clsf_trainer_and_logger
    trainer.model = torch.nn.Identity()  # the batch below already holds the model's log-probabilities
    p1 = torch.tensor([0.1, 0.4, 0.6, 0.9])  # P(class 1): every positive scores above every negative
    log_probs = torch.stack([1 - p1, p1], dim=1).log()
    target = torch.tensor([0, 0, 1, 1])

    _, y_true, y_score, _ = trainer.eval_one_epoch([(log_probs, target)])
    assert calc_metric("roc_auc_score", y_true=y_true, y_score=y_score) == 1.0


def test_compiled_checkpoint_loads_into_plain_model(mock_clsf_config: CustomDictConfig, mocker):
    """torch_compile_model must not add the "_orig_mod." prefix to saved keys"""
    mocker.patch("logging.getLogger")
    mock_clsf_config["torch_compile_model"] = True
    mock_clsf_config["trainer"]["epochs"] = 1
    trainer = ClassifierTrainer(config=mock_clsf_config, logger_name="compile_logger")
    mocker.patch.object(trainer, "train_one_epoch")  # train() still compiles and saves last.pth
    trainer.train()

    plain_model = init_model("ClassifierModel", **mock_clsf_config["model"]["args"])
    plain_model.load_state_dict(torch.load(osp.join(mock_clsf_config["models_dir"], "last.pth"), weights_only=True))


def test_validate_method(mock_clsf_config: CustomDictConfig, mocker):
    mocked_logger = mocker.patch("logging.getLogger")
    mock_clsf_config["dataset"]["args"]["val_path"] = None
    mock_clsf_config["dataloader"]["args"]["validation_split"] = 0.1
    trainer = ClassifierTrainer(config=mock_clsf_config, logger_name="time_series_clsf_logger")
    if trainer.val_data_loader is not None:
        trainer.validate()
        logger_instance = mocked_logger.return_value
        info_calls = [call.args for call in logger_instance.info.call_args_list]
        assert any("Validation" in str(call) for call in info_calls), "Validation log entry missing"
        assert any("loss" in str(call) for call in info_calls), "No logging for validation loss"
    else:
        pytest.skip("No validation data loader configured for this test.")


def test_test_method(clsf_trainer_and_logger):
    trainer, mocked_logger = clsf_trainer_and_logger
    trainer.test()

    logger_instance = mocked_logger.return_value
    info_calls = [call.args for call in logger_instance.info.call_args_list]
    assert any("Test set:" in str(call) for call in info_calls), "Test log entry missing"
