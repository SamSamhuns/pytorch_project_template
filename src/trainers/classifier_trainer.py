"""Image classification trainer"""

import glob
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets.folder import is_image_file
from torchvision.utils import make_grid

from src.config_parser import CustomDictConfig
from src.losses import init_loss
from src.metrics import calc_metric, plot_metric
from src.models import init_model
from src.optimizers import init_optimizer
from src.schedulers import init_scheduler
from src.trainers import BaseTrainer
from src.utils.custom_statistics import get_model_params


class ClassifierTrainer(BaseTrainer):
    """Main ClassifierTrainer with the train, test, validate, inference, load_checkpoint and save_checkpoint funcs"""

    def __init__(self, config: CustomDictConfig, logger_name: str):
        super().__init__(config, logger_name)

        # define models
        self.model = init_model(self.config["model"]["type"], **self.config["model"]["args"])
        self.logger.info("Model info: %s", get_model_params(self.model))

        # resume from a checkpoint, if given
        resume_ckpt = self.config["trainer"]["resume_checkpoint"]
        checkpoint = self.load_checkpoint(resume_ckpt) if resume_ckpt else None
        if checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.logger.info("Training will be done from scratch")
        self.model = self.model.to(self.device)

        # do not progress beyond for INFERENCE mode
        if config["mode"] in {"INFERENCE"}:
            return

        # define loss and instantiate it
        self.loss = init_loss(self.config["loss"]["type"], **self.config["loss"]["args"]).to(self.device)
        # define optimizer
        self.optimizer = init_optimizer(
            self.config["optimizer"]["type"], **self.config["optimizer"]["args"], params=self.model.parameters()
        )
        # define lr scheduler
        self.scheduler = init_scheduler(self.optimizer, self.config["lr_scheduler"])
        # best.pth is saved whenever the val accuracy beats best_val_acc
        self.best_val_acc = float("-inf")
        self.best_checkpoint_path = None  # set once this run saves a best.pth
        # initialize counter
        self.current_epoch = 1
        if checkpoint:  # continue training where the checkpoint left off
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.best_val_acc = checkpoint["best_val_acc"]
            self.current_epoch = checkpoint["epoch"] + 1

    def save_checkpoint(self, file_name: str) -> str:
        """Save everything needed to resume training to models_dir/file_name and return the path"""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "best_val_acc": self.best_val_acc,
            "classes": self.data_set.classes,  # for inference.py and export.py, which name the predicted classes
        }
        save_path = osp.join(self.config["models_dir"], file_name)
        torch.save(checkpoint, save_path)
        return save_path

    def train_one_epoch(self) -> None:
        """One epoch of training"""
        if self.config.verbose:
            self.logger.info("Training set:")
        epoch_start_tm = time.perf_counter()
        self.model.train()
        # summed on the device and read once per epoch, as .item() makes the CPU wait for the GPU
        cum_train_loss = torch.zeros((), device=self.device)
        correct = torch.zeros((), dtype=torch.long, device=self.device)
        train_size = 0
        train_data_len = len(self.train_data_loader.dataset)

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            train_size += data.shape[0]

            # mixed precision for the forward pass (model + loss); bfloat16 needs no GradScaler
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.config["use_amp"]):
                output = self.model(data)
                loss = self.loss(output, target)
            # loss is the batch mean, so weight it by the batch size
            cum_train_loss += loss.detach() * data.size(0)
            # the predicted class has the largest logit
            correct += (output.argmax(dim=1) == target).sum()

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config["trainer"]["batch_log_freq"] == 0:
                if self.config.verbose:
                    self.logger.info(
                        "\tTrain Epoch: %s [%6d/%d (%.1f%%)] Loss: %.6f",
                        self.current_epoch,
                        batch_idx * len(data),
                        train_data_len,
                        100 * batch_idx / len(self.train_data_loader),
                        loss.item(),
                    )
                if self.config["trainer"]["use_tensorboard"]:
                    # counts batches since the start of training, so it continues after a resume
                    step = (self.current_epoch - 1) * len(self.train_data_loader) + batch_idx
                    self.tboard_writer.add_scalar("Loss/train/iteration", loss.item(), step)

        train_loss = cum_train_loss.item() / train_size
        correct = correct.item()
        train_accuracy = 100.0 * correct / train_size
        epoch_end_tm = time.perf_counter()
        epoch_tm = epoch_end_tm - epoch_start_tm
        if self.config.verbose:
            self.logger.info("\tEpoch time: %.2fs", epoch_tm)
            self.logger.info(
                "\tAverage loss: %.4f, Accuracy: %s/%s (%.0f%%)\n", train_loss, correct, train_size, train_accuracy
            )
        else:
            self.logger.info(
                "\tTrain Epoch %d, time: %.2fs, Loss: %.4f, Accuracy: %.2f%%",
                self.current_epoch,
                epoch_tm,
                train_loss,
                train_accuracy,
            )

        # if no val_data_loader, then run ReduceLROnPlateau on train_loss instead
        if self.val_data_loader is None and isinstance(self.scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau scheduler takes metrics during its step call
            self.scheduler.step(metrics=train_loss)

        if self.config["trainer"]["use_tensorboard"]:
            # data still holds the epoch's last batch, so the loader isn't read again just for this;
            # the images are normalized, so make_grid rescales them to the [0, 1] range TensorBoard shows
            self.tboard_writer.add_image(
                "preprocessed image batch", make_grid(data, normalize=True), self.current_epoch
            )
            self.tboard_writer.add_scalar("Loss/train/epoch", train_loss, self.current_epoch)
            self.tboard_writer.add_scalar("Accuracy/train/epoch", train_accuracy, self.current_epoch)
            self.tboard_writer.add_scalar(
                "Learning Rate/train/epoch", self.optimizer.param_groups[0]["lr"], self.current_epoch
            )

    def eval_one_epoch(self, data_loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model on one epoch with the given dataloader.
        Returns the mean loss, y_true, y_score and y_pred.
        y_score is P(class 1) for binary models and the class probabilities otherwise.
        """
        self.model.eval()
        cumu_test_loss = torch.zeros((), device=self.device)  # read once at the end; .item() would sync every step
        y_true, y_score, y_pred = [], [], []
        with torch.inference_mode():
            for data, target in tqdm.tqdm(data_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.config["use_amp"]):
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

    def train(self) -> None:
        """Main training function with loop"""
        if self.config["torch_compile_model"]:
            self.logger.info("Using torch compile mode")
            # compiles in place, so state_dict keys stay the same as the uncompiled model's
            self.model.compile()
        train_start_tm = time.perf_counter()
        epochs = self.config["trainer"]["epochs"]
        start_epoch = self.current_epoch  # the epoch after the checkpoint's when resuming
        self.logger.info("Starting training...")
        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            self.train_one_epoch()
            val_accuracy = None
            if epoch % self.config["trainer"]["valid_freq"] == 0 and self.val_data_loader is not None:
                val_accuracy = self.validate()
            # ReduceLROnPlateau steps on a loss instead: the val loss in validate(), or the train loss without a val set
            if not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

            # saved after the scheduler step, so a resumed run continues with the next epoch's LR
            if val_accuracy is not None and val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.best_checkpoint_path = self.save_checkpoint("best.pth")
            self.save_checkpoint("last.pth")
        train_end_tm = time.perf_counter()
        train_tm = train_end_tm - train_start_tm

        self.logger.info("Total train time: %.3fs", train_tm)
        self.logger.info("Average epoch time: %.3fs", train_tm / max(epochs - start_epoch + 1, 1))
        self.logger.info("Finished training.\n")

    def validate(self) -> float:
        """One epoch of model validation, returns the val accuracy in %"""
        val_start_tm = time.perf_counter()

        val_loss, y_true, y_score, y_pred = self.eval_one_epoch(self.val_data_loader)
        # a python float, since checkpoints load with weights_only=True, which rejects numpy scalars
        val_accuracy = 100.0 * float((y_true == y_pred).mean())

        if isinstance(self.scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau scheduler takes metrics during its step call
            self.scheduler.step(metrics=val_loss)

        # add all val metrics
        metric_name_val_dict = {}
        for metric_name in self.config["metrics"]["val"]:
            if metric_name in {"roc_auc_score", "roc_curve", "pr_curve", "calibration_curve"} and y_score.ndim > 1:
                continue  # binary-only metrics; multi-class y_score holds one column per class
            if metric_name in {"roc_curve", "pr_curve", "calibration_curve"}:
                if self.config.verbose:
                    self.logger.info("\tPlotting val %s", metric_name)
                plot_metric(
                    metric_name,
                    y_true=y_true,
                    y_score=y_score,
                    savepath=os.path.join(self.config["logs_dir"], f"val_e{self.current_epoch}_{metric_name}.png"),
                )
                continue
            metric_val = calc_metric(metric_name, y_true=y_true, y_score=y_score, y_pred=y_pred)
            if (
                metric_name not in {"confusion_matrix", "classification_report"}
                and self.config["trainer"]["use_tensorboard"]
            ):
                self.tboard_writer.add_scalar(f"{metric_name}/val/epoch", metric_val, self.current_epoch)
            metric_name_val_dict[metric_name] = metric_val

        val_end_tm = time.perf_counter()
        if self.config.verbose:
            self.logger.info("Validation set:")
            self.logger.info("\tval time: %.3fs", (val_end_tm - val_start_tm))
            self.logger.info("\tavg loss: %.4f", val_loss)
            for metric_name, metric_val in metric_name_val_dict.items():
                if metric_name in {"confusion_matrix", "classification_report"}:
                    self.logger.info("\t%s: \n%s", metric_name, metric_val)
                else:
                    self.logger.info("\t%s: %.4f", metric_name, metric_val)
        else:
            log_message = (
                f"\tValidation at Epoch {self.current_epoch} | val time: {val_end_tm - val_start_tm:.3f}s"
                f" | avg loss: {val_loss:.4f}"
            )
            for metric_name, metric_val in metric_name_val_dict.items():
                # confusion_matrix and classification_report are not reported in non-verbose mode
                if metric_name not in {"confusion_matrix", "classification_report"}:
                    log_message += f" | {metric_name}: {metric_val:.4f}"
            self.logger.info(log_message)
        if self.config["trainer"]["use_tensorboard"]:
            self.tboard_writer.add_scalar("Loss/validation/epoch", val_loss, self.current_epoch)
        return val_accuracy

    def test(self, weight_path: str | None = None) -> None:
        """Main test function
        args:
            weight_path: checkpoint to test, e.g. models/best.pth. None tests the current weights
        """
        self.logger.info("Testing model...\n")
        test_start_tm = time.perf_counter()

        if weight_path is not None:
            self.model.load_state_dict(self.load_checkpoint(weight_path)["model"])
        if self.test_data_loader is None:
            raise NotImplementedError("test_data_loader is None or missing.test_path might not have been set")

        test_loss, y_true, y_score, y_pred = self.eval_one_epoch(self.test_data_loader)

        self.logger.info("Test set:")
        self.logger.info("\tAverage loss: %.4f", test_loss)
        if self.config["trainer"]["use_tensorboard"]:
            self.tboard_writer.add_scalar("Loss/test/epoch", test_loss, self.current_epoch)
        # add all test metrics
        for metric_name in self.config["metrics"]["test"]:
            if metric_name in {"roc_auc_score", "roc_curve", "pr_curve", "calibration_curve"} and y_score.ndim > 1:
                continue  # binary-only metrics; multi-class y_score holds one column per class
            if metric_name in {"roc_curve", "pr_curve", "calibration_curve"}:
                self.logger.info("\tPlotting %s", metric_name)
                plot_metric(
                    metric_name,
                    y_true=y_true,
                    y_score=y_score,
                    savepath=os.path.join(self.config["logs_dir"], f"{metric_name}.png"),
                )
                continue
            metric_val = calc_metric(metric_name, y_true=y_true, y_score=y_score, y_pred=y_pred)
            if metric_name in {"confusion_matrix", "classification_report"}:
                self.logger.info("\t%s: \n%s", metric_name, metric_val)
            else:
                self.logger.info("\t%s: %.4f", metric_name, metric_val)
                if self.config["trainer"]["use_tensorboard"]:
                    self.tboard_writer.add_scalar(f"{metric_name}/test/epoch", metric_val, self.current_epoch)

        test_end_tm = time.perf_counter()
        self.logger.info("\nTest time: %.3fs", test_end_tm - test_start_tm)
        self.logger.info("Finished testing model.")

    def inference(
        self, source_path: str, weight_path: str | None = None, log_txt_preds: bool = True
    ) -> list[tuple[str, float]]:
        """Run inference on an image file or directory with existing model or model loaded with new weight_path
        Args:
            source_path: str
        """
        if weight_path is not None:
            self.model.load_state_dict(self.load_checkpoint(weight_path)["model"])
        self.model.eval()

        image_list = []
        if osp.isdir(source_path):
            fpaths = glob.glob(osp.join(source_path, "*"))
            image_list = [path for path in fpaths if is_image_file(path)]
        elif osp.isfile(source_path) and is_image_file(source_path):
            image_list = [source_path]
        else:
            raise ValueError(f"Inference source {source_path} is not an image file or dir with images")

        pred_file_labels = []
        with torch.inference_mode():
            for image_path in image_list:
                image = Image.open(image_path).convert("RGB")
                data = self.preprocess.eval_transform(image)
                data = data.to(self.device).unsqueeze(0)
                output = self.model(data)
                # the predicted class has the largest logit
                pred = output.max(1, keepdim=True)[1].item()

                self.logger.info("Image path=%s: predicted label=%s", image_path, pred)
                pred_file_labels.append([image_path, pred])
        if log_txt_preds:
            with open(osp.join(self.config["logs_dir"], "pred.txt"), "w", encoding="utf-8") as pred_ptr:
                for image_path, pred in pred_file_labels:
                    pred_ptr.write(f"{image_path}, {pred}\n")

        self.logger.info("Inference complete for %s", source_path)
        return pred_file_labels
