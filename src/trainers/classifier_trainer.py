"""Image classification trainer"""

import logging
import os.path as osp
import textwrap
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from src.metrics import PLOTS, calc_metric, plot_metric
from src.utils.custom_statistics import get_model_params

logger = logging.getLogger(__name__)


def build_scheduler(optimizer: Optimizer, cfg: DictConfig) -> lr_scheduler.LRScheduler:
    """Build any scheduler in torch.optim.lr_scheduler from cfg.type and cfg.args.
    With cfg.warmup_epochs, the LR first climbs linearly from 10% to 100% (not with ReduceLROnPlateau).
    """
    scheduler = getattr(lr_scheduler, cfg["type"])(optimizer, **cfg["args"])
    warmup_epochs = cfg.get("warmup_epochs", 0)
    if warmup_epochs:
        warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        scheduler = lr_scheduler.SequentialLR(optimizer, [warmup, scheduler], milestones=[warmup_epochs])
    return scheduler


class ClassifierTrainer:
    """Trains and tests an image classifier with the loss, optimizer and LR scheduler the config names.
    fit() trains up to cfg.trainer.epochs, validates every cfg.trainer.valid_freq epochs, and saves last.pth every
    epoch and best.pth whenever the val accuracy improves, to run_dir/models. evaluate() logs the test metrics.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        run_dir: str,
        device: torch.device,
        classes: list[str] | None = None,
        verbose: bool = False,
    ):
        """classes: the class names, saved in the checkpoints for inference.py and export.py"""
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.classes = classes
        self.verbose = verbose
        self.model = model.to(device)
        logger.info("Model info: %s", get_model_params(self.model))
        # the loss, optimizer and LR scheduler are torch's own, named in the config, e.g. CrossEntropyLoss and SGD
        self.loss = getattr(nn, cfg.loss.type)(**cfg.loss.args).to(device)
        self.optimizer = getattr(torch.optim, cfg.optimizer.type)(self.model.parameters(), **cfg.optimizer.args)
        self.scheduler = build_scheduler(self.optimizer, cfg.lr_scheduler)
        self.epoch = 1  # the epoch to train next; after fit(), the last one trained
        self.best_val_acc = float("-inf")  # best.pth is saved whenever the val accuracy beats it
        self.best_path = None  # set once fit() saves a best.pth
        self.writer = None
        if cfg.trainer.use_tensorboard:
            self.writer = SummaryWriter(osp.join(run_dir, "tf_logs"))
            # the whole config as YAML, indented so TensorBoard's markdown shows it as a code block
            self.writer.add_text("config", textwrap.indent(OmegaConf.to_yaml(cfg), "    "))

    def save(self, file_name: str) -> str:
        """Save everything needed to resume training to run_dir/models/file_name, and return its path"""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "best_val_acc": self.best_val_acc,
            "classes": self.classes,  # for inference.py and export.py, which name the predicted classes
        }
        path = osp.join(self.run_dir, "models", file_name)
        torch.save(checkpoint, path)
        return path

    def load(self, path: str) -> None:
        """Continue from a checkpoint saved by save(): the weights, the optimizer and LR schedule, and the epoch"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.epoch = checkpoint["epoch"] + 1
        logger.info("Resuming from %s at epoch %d", path, self.epoch)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        """Train from self.epoch up to cfg.trainer.epochs"""
        if self.cfg.torch_compile_model:
            logger.info("Using torch compile mode")
            # compiles in place, so state_dict keys stay the same as the uncompiled model's
            self.model.compile()
        train_start_tm, start_epoch = time.perf_counter(), self.epoch
        logger.info("Starting training...")
        for epoch in range(start_epoch, self.cfg.trainer.epochs + 1):
            self.epoch = epoch
            train_loss = self.train_one_epoch(train_loader)
            val_accuracy = val_loss = None
            if val_loader is not None and epoch % self.cfg.trainer.valid_freq == 0:
                val_accuracy, val_loss = self.validate(val_loader)
            # ReduceLROnPlateau steps on a loss: the val loss, or the train loss without a val set
            if not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            elif val_loader is None:
                self.scheduler.step(train_loss)
            elif val_loss is not None:
                self.scheduler.step(val_loss)

            # saved after the scheduler step, so a resumed run continues with the next epoch's LR
            if val_accuracy is not None and val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.best_path = self.save("best.pth")
            self.save("last.pth")
        train_tm = time.perf_counter() - train_start_tm

        logger.info("Total train time: %.3fs", train_tm)
        logger.info("Average epoch time: %.3fs", train_tm / max(self.cfg.trainer.epochs - start_epoch + 1, 1))
        logger.info("Finished training.\n")

    def train_one_epoch(self, loader: DataLoader) -> float:
        """One epoch of training, returns the mean train loss"""
        epoch_start_tm = time.perf_counter()
        self.model.train()
        # summed on the device and read once per epoch, as .item() makes the CPU wait for the GPU
        cum_train_loss = torch.zeros((), device=self.device)
        correct = torch.zeros((), dtype=torch.long, device=self.device)
        train_size = 0

        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            train_size += data.shape[0]

            # mixed precision for the forward pass (model + loss); bfloat16 needs no GradScaler
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cfg.use_amp):
                output = self.model(data)
                loss = self.loss(output, target)
            # loss is the batch mean, so weight it by the batch size
            cum_train_loss += loss.detach() * data.size(0)
            # the predicted class has the largest logit
            correct += (output.argmax(dim=1) == target).sum()

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.cfg.trainer.batch_log_freq == 0:
                if self.verbose:
                    logger.info(
                        "\tTrain Epoch: %s [%6d/%d (%.1f%%)] Loss: %.6f",
                        self.epoch,
                        batch_idx * len(data),
                        len(loader.dataset),
                        100 * batch_idx / len(loader),
                        loss.item(),
                    )
                if self.writer is not None:
                    # counts batches since the start of training, so it continues after a resume
                    step = (self.epoch - 1) * len(loader) + batch_idx
                    self.writer.add_scalar("Loss/train/iteration", loss.item(), step)

        train_loss = cum_train_loss.item() / train_size
        train_accuracy = 100.0 * correct.item() / train_size
        epoch_tm = time.perf_counter() - epoch_start_tm
        logger.info(
            "\tTrain Epoch %d, time: %.2fs, Loss: %.4f, Accuracy: %.2f%%",
            self.epoch,
            epoch_tm,
            train_loss,
            train_accuracy,
        )
        if self.writer is not None:
            # data still holds the epoch's last batch, so the loader isn't read again just for this;
            # the images are normalized, so make_grid rescales them to the [0, 1] range TensorBoard shows
            self.writer.add_image("preprocessed image batch", make_grid(data, normalize=True), self.epoch)
            self.writer.add_scalar("Loss/train/epoch", train_loss, self.epoch)
            self.writer.add_scalar("Accuracy/train/epoch", train_accuracy, self.epoch)
            self.writer.add_scalar("Learning Rate/train/epoch", self.optimizer.param_groups[0]["lr"], self.epoch)
        return train_loss

    def eval_one_epoch(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the model on one pass over the loader.
        Returns the mean loss, y_true, y_score and y_pred.
        y_score is P(class 1) for binary models and the class probabilities otherwise.
        """
        self.model.eval()
        cumu_test_loss = torch.zeros((), device=self.device)  # read once at the end; .item() would sync every step
        y_true, y_score, y_pred = [], [], []
        with torch.inference_mode():
            for data, target in tqdm.tqdm(loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cfg.use_amp):
                    output = self.model(data)
                    loss = self.loss(output, target)
                # sum up batch loss (the loss is the batch mean, so weight it by the batch size)
                cumu_test_loss += loss * data.size(0)
                # get the class probabilities (proba) & the predicted class (pred); float() as numpy has no bfloat16
                proba = F.softmax(output.float(), dim=1)
                pred = proba.argmax(dim=1)
                if proba.shape[1] == 2:  # binary: roc/pr/calibration take P(class 1)
                    proba = proba[:, 1]

                y_true.append(target)
                y_score.append(proba)
                y_pred.append(pred)
        y_true = torch.concat(y_true).cpu().numpy()
        y_score = torch.concat(y_score).cpu().numpy()
        y_pred = torch.concat(y_pred).cpu().numpy()

        return cumu_test_loss.item() / len(y_true), y_true, y_score, y_pred

    def validate(self, loader: DataLoader) -> tuple[float, float]:
        """One pass over the val set that logs the cfg.metrics.val metrics. Returns the accuracy in % and the loss"""
        val_start_tm = time.perf_counter()
        val_loss, y_true, y_score, y_pred = self.eval_one_epoch(loader)
        logger.info("\tValidation at Epoch %d, time: %.3fs", self.epoch, time.perf_counter() - val_start_tm)
        self.log_metrics("val", val_loss, y_true, y_score, y_pred, plot_prefix=f"val_e{self.epoch}")
        # a python float, since checkpoints load with weights_only=True, which rejects numpy scalars
        return 100.0 * float((y_true == y_pred).mean()), val_loss

    def evaluate(self, loader: DataLoader, weights: str | None = None) -> None:
        """Log the cfg.metrics.test metrics on the loader's data
        weights: a checkpoint to test, e.g. models/best.pth. None tests the current weights
        """
        if weights is not None:
            self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
            logger.info("Testing %s", weights)
        test_start_tm = time.perf_counter()
        test_loss, y_true, y_score, y_pred = self.eval_one_epoch(loader)
        logger.info("Test set, time: %.3fs", time.perf_counter() - test_start_tm)
        self.log_metrics("test", test_loss, y_true, y_score, y_pred, plot_prefix="test")

    def log_metrics(self, split: str, loss: float, y_true, y_score, y_pred, plot_prefix: str) -> None:
        """Log the loss and the cfg.metrics[split] metrics, also to TensorBoard, and plot the curves to run_dir/logs"""
        scores, reports = [f"{split} loss: {loss:.4f}"], []
        if self.writer is not None:
            self.writer.add_scalar(f"Loss/{split}/epoch", loss, self.epoch)
        for name in self.cfg.metrics[split]:
            if (name == "roc_auc_score" or name in PLOTS) and y_score.ndim > 1:
                continue  # binary-only metrics; multi-class y_score holds one column per class
            if name in PLOTS:
                plot_metric(name, y_true, y_score, osp.join(self.run_dir, "logs", f"{plot_prefix}_{name}.png"))
                continue
            value = calc_metric(name, y_true, y_score, y_pred, self.cfg.metrics.average)
            if name in {"confusion_matrix", "classification_report"}:
                reports.append(f"{name}:\n{value}")
                continue
            scores.append(f"{name}: {value:.4f}")
            if self.writer is not None:
                self.writer.add_scalar(f"{name}/{split}/epoch", value, self.epoch)
        logger.info("\t%s", " | ".join(scores))
        for report in reports:
            logger.info("\t%s", report)
