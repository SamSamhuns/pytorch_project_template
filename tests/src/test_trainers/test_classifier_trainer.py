import os.path as osp

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
    info_calls = [call.args for call in mocked_logger.return_value.info.call_args_list]
    assert ("Program will run on %s", trainer.device) in info_calls

    assert isinstance(trainer.model, torch.nn.Module)
    assert isinstance(trainer.loss, torch.nn.Module)
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert isinstance(trainer.optimizer, torch.optim.SGD)
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
    """Val accuracies 90% -> 50% -> 60% must save best.pth once, in the first epoch"""
    trainer, _ = clsf_trainer_and_logger
    trainer.config["trainer"]["epochs"] = 3
    trainer.config["trainer"]["valid_freq"] = 1
    mocker.patch.object(trainer, "train_one_epoch")
    mocker.patch.object(trainer, "validate", side_effect=[90.0, 50.0, 60.0])
    save_checkpoint = mocker.patch.object(trainer, "save_checkpoint")

    trainer.train()

    saved = [call.args[0] for call in save_checkpoint.call_args_list]
    assert saved == ["best.pth", "last.pth", "last.pth", "last.pth"]
    assert trainer.best_val_acc == 90.0


def test_resume_continues_where_training_stopped(mock_clsf_config: CustomDictConfig, mocker):
    """Resuming from last.pth restores the epoch, the optimizer state and the LR schedule"""
    mocker.patch("logging.getLogger")
    mock_clsf_config["trainer"]["epochs"] = 2
    mock_clsf_config["lr_scheduler"] = {"type": "CosineAnnealingLR", "args": {"T_max": 4}, "warmup_epochs": 1}
    trainer = ClassifierTrainer(config=mock_clsf_config, logger_name="resume_logger")
    trainer.train()

    mock_clsf_config["trainer"]["resume_checkpoint"] = osp.join(mock_clsf_config["models_dir"], "last.pth")
    resumed = ClassifierTrainer(config=mock_clsf_config, logger_name="resume_logger")

    assert resumed.current_epoch == 3
    assert resumed.optimizer.param_groups[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()
    momentum = [state["momentum_buffer"] for state in trainer.optimizer.state.values()]
    resumed_momentum = [state["momentum_buffer"] for state in resumed.optimizer.state.values()]
    assert all(torch.equal(a, b) for a, b in zip(momentum, resumed_momentum, strict=True))


def test_eval_loss_is_mean_over_samples(clsf_trainer_and_logger):
    """Batches of unequal size must be weighted by their size"""
    trainer, _ = clsf_trainer_and_logger
    trainer.model = torch.nn.Identity()  # the batches below already hold the model's logits
    logits = torch.randn(4, 5)
    target = torch.tensor([0, 1, 2, 3])

    loss, _, _, _ = trainer.eval_one_epoch([(logits[:3], target[:3]), (logits[3:], target[3:])])
    assert loss == pytest.approx(F.cross_entropy(logits, target).item())


def test_amp_scores_convert_to_numpy(clsf_trainer_and_logger):
    """With use_amp the model outputs bfloat16, which numpy can't hold, so the scores must be cast"""
    trainer, _ = clsf_trainer_and_logger
    trainer.config["use_amp"] = True

    _, _, y_score, _ = trainer.eval_one_epoch([(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))])
    assert y_score.dtype == np.float32


def test_binary_y_score_is_positive_class_probability(clsf_trainer_and_logger):
    """A perfect binary classifier must get ROC-AUC 1.0"""
    trainer, _ = clsf_trainer_and_logger
    trainer.model = torch.nn.Identity()  # the batch below already holds the model's logits
    p1 = torch.tensor([0.1, 0.4, 0.6, 0.9])  # P(class 1): every positive scores above every negative
    logits = torch.stack([1 - p1, p1], dim=1).log()  # softmax maps these back to the probabilities
    target = torch.tensor([0, 0, 1, 1])

    _, y_true, y_score, _ = trainer.eval_one_epoch([(logits, target)])
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
    checkpoint = torch.load(osp.join(mock_clsf_config["models_dir"], "last.pth"), weights_only=True)
    plain_model.load_state_dict(checkpoint["model"])


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
