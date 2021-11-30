# FIXME: @Samyak, could you please help add docstring to this file? Thanks!
import os
import time
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.models as models
from gradattack.utils import StandardizeLayer
from sklearn import metrics
from torch.nn import init
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, ReduceLROnPlateau, StepLR

from .covidmodel import *
from .densenet import *
from .googlenet import *
from .mobilenet import *
from .nasnet import *
from .resnet import *
from .resnext import *
from .simple import *
from .vgg import *
from .multihead_resnet import *
from .LeNet import *


class StepTracker:
    def __init__(self):
        self.in_progress = False

    def start(self):
        self.cur_loss = 0
        self.in_progress = True

    def end(self, deduction: float = 0):
        self.in_progress = False


class LightningWrapper(pl.LightningModule):
    """Wraps a torch module in a pytorch-lightning module. Any ."""
    def __init__(
        self,
        model: torch.nn.Module,
        training_loss_metric: Callable = F.mse_loss,
        optimizer: str = "SGD",
        lr_scheduler: str = "ReduceLROnPlateau",
        tune_on_val: float = 0.02,
        lr_factor: float = 0.5,
        lr_step: int = 10,
        batch_size: int = 64,
        lr: float = 0.05,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        nesterov: bool = False,
        log_auc: bool = False,
        multi_class: bool = False,
        multi_head: bool = False,
    ):
        super().__init__()
        # if we didn't copy here, then we would modify the default dict by accident
        self.save_hyperparameters(
            "optimizer",
            "lr_scheduler",
            "tune_on_val",
            "lr_step",
            "lr_factor",
            "lr",
            "momentum",
            "nesterov",
            "weight_decay",
            "batch_size",
        )

        self._model = model
        self._training_loss_metric = training_loss_metric
        self._val_loss_metric = training_loss_metric

        self._batch_transformations = []
        self._grad_transformations = []
        self._opt_transformations = []

        self._epoch_end_callbacks = []
        self._step_end_callbacks = []
        self._log_gradients = False

        self.current_val_loss = 100
        self._on_train_epoch_start_callbacks = []
        self.automatic_optimization = False

        self.step_tracker = StepTracker()
        self.log_train_acc = True
        self.log_auc = log_auc
        self.multi_class = multi_class
        self.multi_head = multi_head

    def forward(self, x):
        if self.multi_head:
            output = self._model(x)
            output = torch.stack(output)
            output = torch.transpose(output, 0, 1)
            return output
        else:
            return self._model(x)

    def should_accumulate(self):
        return self.trainer.train_loop.should_accumulate()

    def on_train_epoch_start(self) -> None:
        for callback in self._on_train_epoch_start_callbacks:
            callback(self)

    def _transform_batch(self, batch, batch_idx, *args):
        for transform in self._batch_transformations:
            batch = transform(batch, batch_idx, *args)
        return batch

    def _transform_gradients(self):
        for transform in self._grad_transformations:
            self._model = transform(self._model)

    def _compute_training_step(self,
                               batch: torch.tensor,
                               batch_idx: int,
                               apply_batch_transforms=True) -> dict:
        """Computes the loss for a single training step (without updating any trackers or logging).
        Args:
            batch : The batch inputs. Should be a torch tensor with outermost dimension 2, where dimension 0 corresponds to inputs and dimension 1
            corresponds to labels.

        Returns:
            dict: The results from the training step. Is a dictionary with keys "loss", "transformed_batch", and "model_outputs".
        """
        if apply_batch_transforms:
            batch = self._transform_batch(batch, batch_idx)
        x, y = batch
        y_hat = self(x)

        if self.multi_head:
            loss = []
            for j in range(y_hat.size(1)):
                loss.append(self._training_loss_metric(y_hat[:, j], y[:, j]))
            loss = sum(loss)
        else:
            loss = self._training_loss_metric(y_hat, y)

        return {
            "loss": loss,
            "transformed_batch": (x, y),
            "model_outputs": y_hat,
        }

    def training_step(self, batch, batch_idx, *_) -> dict:
        if self.step_tracker.in_progress == False:
            self.step_tracker.start()

        training_step_results = self._compute_training_step(batch, batch_idx)
        self.step_tracker.cur_loss += training_step_results["loss"].item()

        self.manual_backward(training_step_results["loss"])

        if self.should_accumulate():
            # Special case opacus optimizers to reduce memory footprint
            # see: (https://github.com/pytorch/opacus/blob/244265582bffbda956511871a907e5de2c523d86/opacus/privacy_engine.py#L393)
            if hasattr(self.optimizer, "virtual_step"):
                with torch.no_grad():
                    self.optimizer.virtual_step()
        else:
            self.on_non_accumulate_step()

        if self.log_train_acc:
            top1_acc = accuracy(
                training_step_results["model_outputs"],
                training_step_results["transformed_batch"][1],
                multi_head=self.multi_head,
            )[0]
            self.log(
                "step/train_acc",
                top1_acc,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )

        return training_step_results

    def get_batch_gradients(
        self,
        batch: torch.tensor,
        batch_idx: int = 0,
        create_graph: bool = False,
        clone_gradients: bool = True,
        apply_transforms=True,
        eval_mode: bool = False,
        stop_track_bn_stats: bool = True,
        BN_exact: bool = False,
        attacker: bool = False,
        *args,
    ):
        batch = tuple(k.to(self.device) for k in batch)
        if eval_mode is True:
            self.eval()
        else:
            self.train()

        if BN_exact:
            for module in self._model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    if not attacker:
                        module.reset_running_stats()  # reset BN statistics
                        module.momentum = (
                            1  # save current BN statistics as running statistics
                        )
                    if attacker:
                        self.training = False  # set BN module to eval mode
                        module.momentum = 0  # stop tracking BN statistics
                        if hasattr(module, "weight"):
                            module.weight.requires_grad_(True)
                        if hasattr(module, "bias"):
                            module.bias.requires_grad_(True)

        if stop_track_bn_stats:
            for module in self._model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.momentum = 0  # Stop tracking running mean and std any more

        self.zero_grad()
        training_step_results = self._compute_training_step(
            batch, batch_idx, apply_batch_transforms=apply_transforms, *args)

        # Make sure to apply transformations to gradients
        if apply_transforms:
            training_step_results["loss"].backward()
            self._transform_gradients()
            # Clone to prevent the gradients from being changed by training
            batch_gradients = tuple(
                p.grad.clone() if clone_gradients is True else p.grad
                for p in self.parameters())
        else:
            batch_gradients = torch.autograd.grad(
                training_step_results["loss"],
                self._model.parameters(),
                create_graph=create_graph,
            )

        return batch_gradients, training_step_results

    def on_non_accumulate_step(self) -> None:
        # This hook runs only after accumulation
        self._transform_gradients()

        if self._log_gradients:
            grad_norm_dict = self.grad_norm(1)
            for k, v in grad_norm_dict.items():
                self.log(
                    f"gradients/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        self.optimizer.step()
        self.optimizer.zero_grad()

        for callback in self._step_end_callbacks:
            TERMINATE = callback(self, self.step_tracker)
            if TERMINATE:
                self.trainer.should_stop = True

        self.step_tracker.end()
        self.log(
            "step/train_loss",
            self.step_tracker.cur_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["lr"],
                momentum=self.hparams["momentum"],
                weight_decay=self.hparams["weight_decay"],
                nesterov=self.hparams["nesterov"],
            )
        self.configure_lr_scheduler()

        for transform in self._opt_transformations:
            transform(self)

        return self.optimizer

    def configure_lr_scheduler(self):
        self.lr_scheduler = None
        if self.hparams["lr_scheduler"] == "StepLR":
            self.lr_scheduler = StepLR(
                self.optimizer,
                step_size=self.hparams["lr_step"],
                gamma=self.hparams["lr_factor"],
                verbose=True,
            )
        elif self.hparams["lr_scheduler"] == "MultiStepLR":
            self.lr_scheduler = MultiStepLR(
                self.optimizer,
                milestones=self.hparams["lr_step"],
                gamma=self.hparams["lr_factor"],
                verbose=True,
            )
        elif self.hparams["lr_scheduler"] == "LambdaLR":
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=[lambda epoch: self.hparams["lr_lambda"]**epoch],
                verbose=True,
            )
        elif self.hparams["lr_scheduler"] == "ReduceLROnPlateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.hparams["lr_factor"],
                patience=2,
                min_lr=1e-4,
                verbose=True,
            )
        elif self.hparams["lr_scheduler"] == "CosineAnnealingLR":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=200)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        if self.multi_head:
            loss = []
            for j in range(y_hat.size(1)):
                loss.append(self._val_loss_metric(y_hat[:, j], y[:, j]))
            loss = sum(loss) / len(loss)
        else:
            loss = self._val_loss_metric(y_hat, y)
        top1_acc = accuracy(y_hat, y, multi_head=self.multi_head)[0]
        if self.log_auc:
            pred_list, true_list = auc_list(y_hat, y)
        else:
            pred_list, true_list = None, None
        return {
            "batch/val_loss": loss,
            "batch/val_accuracy": top1_acc,
            "batch/val_pred_list": pred_list,
            "batch/val_true_list": true_list,
        }

    def validation_epoch_end(self, outputs):
        # outputs is whatever returned in `validation_step`
        avg_loss = torch.stack([x["batch/val_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["batch/val_accuracy"]
                                    for x in outputs]).mean()
        if self.log_auc:
            self.log_aucs(outputs, stage="val")

        self.current_val_loss = avg_loss
        if self.current_epoch > 0:
            if self.hparams["lr_scheduler"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(self.current_val_loss)
            else:
                self.lr_scheduler.step()

        self.cur_lr = self.optimizer.param_groups[0]["lr"]

        self.log(
            "epoch/val_accuracy",
            avg_accuracy,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("epoch/val_loss",
                 avg_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("epoch/lr",
                 self.cur_lr,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        for callback in self._epoch_end_callbacks:
            callback(self)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if self.multi_head:
            loss = []
            for j in range(y_hat.size(1)):
                loss.append(self._val_loss_metric(y_hat[:, j], y[:, j]))
            loss = sum(loss) / len(loss)
        else:
            loss = self._val_loss_metric(y_hat, y)
        top1_acc = accuracy(y_hat, y, multi_head=self.multi_head)[0]
        if self.log_auc:
            pred_list, true_list = auc_list(y_hat, y)
        else:
            pred_list, true_list = None, None
        return {
            "batch/test_loss": loss,
            "batch/test_accuracy": top1_acc,
            "batch/test_pred_list": pred_list,
            "batch/test_true_list": true_list,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["batch/test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["batch/test_accuracy"]
                                    for x in outputs]).mean()
        if self.log_auc:
            self.log_aucs(outputs, stage="test")

        self.log("run/test_accuracy",
                 avg_accuracy,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("run/test_loss",
                 avg_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def log_aucs(self, outputs, stage="test"):
        pred_list = np.concatenate(
            [x[f"batch/{stage}_pred_list"] for x in outputs])
        true_list = np.concatenate(
            [x[f"batch/{stage}_true_list"] for x in outputs])

        aucs = []
        for c in range(len(pred_list[0])):
            fpr, tpr, thresholds = metrics.roc_curve(true_list[:, c],
                                                     pred_list[:, c],
                                                     pos_label=1)
            auc_val = metrics.auc(fpr, tpr)
            aucs.append(auc_val)

            self.log(
                f"epoch/{stage}_auc/class_{c}",
                auc_val,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        self.log(
            f"epoch/{stage}_auc/avg",
            np.mean(aucs),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


def create_lightning_module(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    ckpt: str = None,
    freeze_extractor: bool = False,
    *args,
    **kwargs,
) -> LightningWrapper:
    if "models" in model_name:  # Official models by PyTorch
        model_name = model_name.replace("models.", "")
        if pretrained is False:
            _model = models.__dict__[model_name](pretrained=False,
                                                 num_classes=num_classes)
        else:
            _model = models.__dict__[model_name](pretrained=True)
            if num_classes != 1000:
                name, layer = list(_model.named_modules())[-1]
                if isinstance(layer, nn.Linear):
                    _model._modules[name] = nn.Linear(
                        in_features=layer.in_features,
                        out_features=num_classes)
    else:
        _model = globals()[model_name](num_classes)

    if ckpt is not None:
        assert os.path.exists(ckpt), f"Failed to load checkpoint {ckpt}"
        checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
        pretrained_dict = checkpoint["state_dict"]
        pretrained_dict = {
            k.replace("_model.", ""): v
            for k, v in pretrained_dict.items()
            # if "fc" not in k and "classifier" not in k
        }
        pretrained_dict = {
            k.replace("module.", ""): v
            for k, v in pretrained_dict.items()
            # if "fc" not in k and "classifier" not in k
        }  # Unwrap data parallel model
        _model.load_state_dict(pretrained_dict, strict=False)

    if freeze_extractor:
        do_freeze_extractor(_model)
        _model.classifier = nn.Sequential(StandardizeLayer(n_features=1024),
                                          nn.Linear(1024, num_classes))
    return LightningWrapper(_model, *args, **kwargs)


def do_freeze_extractor(model):
    model.eval()
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def multihead_accuracy(output, target):
    prec1 = []
    for j in range(output.size(1)):
        acc = accuracy(output[:, j], target[:, j], topk=(1, ))
        prec1.append(acc[0])
    return torch.mean(torch.Tensor(prec1))


def accuracy(output, target, topk=(1, ), multi_head=False):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if len(target.size()) == 1:  # single-class classification
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        else:  # multi-class classification
            if multi_head:
                res = [multihead_accuracy(output, target)]
            else:
                assert len(topk) == 1
                pred = torch.sigmoid(output).ge(0.5).float()
                correct = torch.count_nonzero(pred == target).float()
                correct *= 100.0 / (batch_size * target.size(1))
                res = [correct]
    return res


def auc_list(output, target):
    assert len(target.size()) == 2
    pred_list = torch.sigmoid(output).cpu().detach().numpy()
    true_list = target.cpu().detach().numpy()

    return pred_list, true_list
