import os.path as osp

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.metrics import calc_metric
from src.models import torchvision_classifier
from src.trainers import ClassifierTrainer
from tests.src.test_trainers.conftest import CLASSES, tiny_model


def test_init_builds_the_configured_parts(trainer):
    """The loss, optimizer and LR scheduler are the torch classes the config names"""
    assert isinstance(trainer.loss, torch.nn.CrossEntropyLoss)
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_train_one_epoch_updates_the_model(trainer_cfg, run_dir, loader, device, caplog):
    trainer = ClassifierTrainer(trainer_cfg, tiny_model(), run_dir, device)
    initial_params = [p.clone() for p in trainer.model.parameters()]

    with caplog.at_level("INFO"):
        train_loss = trainer.train_one_epoch(loader)

    assert any(not torch.equal(a, b) for a, b in zip(initial_params, trainer.model.parameters(), strict=True))
    assert train_loss > 0
    assert "Train Epoch 1" in caplog.text


def test_fit_saves_the_checkpoints(trainer, loader):
    trainer.fit(loader, val_loader=loader)

    assert trainer.epoch == 2
    assert trainer.best_path == osp.join(trainer.run_dir, "models", "best.pth")
    checkpoint = torch.load(osp.join(trainer.run_dir, "models", "last.pth"), weights_only=True)
    assert checkpoint["epoch"] == 2
    assert checkpoint["classes"] == CLASSES


def test_trainer_keeps_the_model_weights(trainer_cfg, run_dir):
    """The trainer used to re-initialize every model, which wiped pretrained backbones"""
    model = torchvision_classifier(num_classes=3, pretrained_weights=None)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    trainer = ClassifierTrainer(trainer_cfg, model, run_dir, torch.device("cpu"))
    assert all(torch.equal(value, weights[name]) for name, value in trainer.model.state_dict().items())


def test_best_checkpoint_keeps_highest_val_accuracy(trainer, loader, mocker):
    """Val accuracies 90% -> 50% -> 60% must save best.pth once, in the first epoch"""
    trainer.cfg.trainer.epochs = 3
    mocker.patch.object(trainer, "train_one_epoch", return_value=1.0)
    mocker.patch.object(trainer, "validate", side_effect=[(90.0, 0.1), (50.0, 0.5), (60.0, 0.4)])
    save = mocker.patch.object(trainer, "save")

    trainer.fit(loader, val_loader=loader)

    assert [call.args[0] for call in save.call_args_list] == ["best.pth", "last.pth", "last.pth", "last.pth"]
    assert trainer.best_val_acc == 90.0


def test_resume_continues_where_training_stopped(trainer_cfg, run_dir, loader):
    """Resuming from last.pth restores the epoch, the optimizer state and the LR schedule"""
    trainer_cfg.lr_scheduler = {"type": "CosineAnnealingLR", "args": {"T_max": 4}, "warmup_epochs": 1}
    trainer = ClassifierTrainer(trainer_cfg, tiny_model(), run_dir, torch.device("cpu"))
    trainer.fit(loader)

    resumed = ClassifierTrainer(trainer_cfg, tiny_model(), run_dir, torch.device("cpu"))
    resumed.load(osp.join(run_dir, "models", "last.pth"))

    assert resumed.epoch == 3
    assert resumed.optimizer.param_groups[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()
    momentum = [state["momentum_buffer"] for state in trainer.optimizer.state.values()]
    resumed_momentum = [state["momentum_buffer"] for state in resumed.optimizer.state.values()]
    assert all(torch.equal(a, b) for a, b in zip(momentum, resumed_momentum, strict=True))


def test_eval_loss_is_mean_over_samples(trainer):
    """Batches of unequal size must be weighted by their size"""
    trainer.model = torch.nn.Identity()  # the batches below already hold the model's logits
    logits = torch.randn(4, 5)
    target = torch.tensor([0, 1, 2, 3])

    loss, _, _, _ = trainer.eval_one_epoch([(logits[:3], target[:3]), (logits[3:], target[3:])])
    assert loss == pytest.approx(F.cross_entropy(logits, target).item())


def test_amp_scores_convert_to_numpy(trainer):
    """With use_amp the model outputs bfloat16, which numpy can't hold, so the scores must be cast"""
    trainer.cfg.use_amp = True

    _, _, y_score, _ = trainer.eval_one_epoch([(torch.randn(2, 3, 16, 16), torch.tensor([0, 1]))])
    assert y_score.dtype == np.float32


def test_binary_y_score_is_positive_class_probability(trainer):
    """A perfect binary classifier must get ROC-AUC 1.0"""
    trainer.model = torch.nn.Identity()  # the batch below already holds the model's logits
    p1 = torch.tensor([0.1, 0.4, 0.6, 0.9])  # P(class 1): every positive scores above every negative
    logits = torch.stack([1 - p1, p1], dim=1).log()  # softmax maps these back to the probabilities
    target = torch.tensor([0, 0, 1, 1])

    _, y_true, y_score, _ = trainer.eval_one_epoch([(logits, target)])
    assert calc_metric("roc_auc_score", y_true=y_true, y_score=y_score) == 1.0


def test_compiled_checkpoint_loads_into_plain_model(trainer, loader, mocker):
    """torch_compile_model must not add the "_orig_mod." prefix to saved keys"""
    trainer.cfg.torch_compile_model = True
    trainer.cfg.trainer.epochs = 1
    mocker.patch.object(trainer, "train_one_epoch", return_value=1.0)  # fit() still compiles and saves last.pth
    trainer.fit(loader)

    checkpoint = torch.load(osp.join(trainer.run_dir, "models", "last.pth"), weights_only=True)
    tiny_model().load_state_dict(checkpoint["model"])


def test_validate_logs_the_metrics(trainer, loader, caplog):
    with caplog.at_level("INFO"):
        accuracy, loss = trainer.validate(loader)
    assert 0 <= accuracy <= 100
    assert loss > 0
    assert "Validation at Epoch 1" in caplog.text
    assert "accuracy_score" in caplog.text


def test_evaluate_tests_the_checkpoint(trainer, trainer_cfg, run_dir, loader, caplog):
    """evaluate() with weights tests that checkpoint, not the trainer's current weights"""
    saved = trainer.save("best.pth")
    other = ClassifierTrainer(trainer_cfg, tiny_model(), run_dir, torch.device("cpu"))

    with caplog.at_level("INFO"):
        other.evaluate(loader, weights=saved)

    assert all(torch.equal(a, b) for a, b in zip(other.model.parameters(), trainer.model.parameters(), strict=True))
    assert "test loss" in caplog.text
    assert "confusion_matrix" in caplog.text


def test_tensorboard_logs_to_the_run_dir(trainer_cfg, run_dir, loader):
    trainer_cfg.trainer.use_tensorboard = True
    trainer_cfg.trainer.epochs = 1
    trainer = ClassifierTrainer(trainer_cfg, tiny_model(), run_dir, torch.device("cpu"))
    trainer.fit(loader, val_loader=loader)
    trainer.evaluate(loader)
    trainer.writer.close()
    assert osp.isdir(osp.join(run_dir, "tf_logs"))
