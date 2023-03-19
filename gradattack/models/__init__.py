import os
from typing import Callable

import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, ReduceLROnPlateau, StepLR

from gradattack.utils import StandardizeLayer
from .LeNet import *
from .covidmodel import *
from .densenet import *
from .googlenet import *
from .mobilenet import *
from .multihead_resnet import *
from .nasnet import *
from .resnet import *
from .resnext import *
from .simple import *
from .vgg import *


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
            log_dir: str = None
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

        self._optimizer = optimizer

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
        self.log_dir = log_dir

    def forward(self, x):
        if self.multi_head:
            output = self._model(x)
            output = torch.stack(output)
            output = torch.transpose(output, 0, 1)
            return output
        else:
            return self._model(x)

    def _transform_batch(self, batch, batch_idx, *args):
        for transform in self._batch_transformations:
            batch = transform(batch, batch_idx, *args)
        return batch

    def on_train_epoch_start(self) -> None:
        for callback in self._on_train_epoch_start_callbacks:
            callback(self)

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

        # if self.should_accumulate():
        #     # Special case opacus optimizers to reduce memory footprint
        #     # see: (https://github.com/pytorch/opacus/blob/244265582bffbda956511871a907e5de2c523d86/opacus/privacy_engine.py#L393)
        #     if hasattr(self.optimizer, "virtual_step"):
        #         with torch.no_grad():
        #             self.optimizer.virtual_step()
        # else:
        #     self.on_non_accumulate_step()
        self.on_non_accumulate_step()

        top1_acc = accuracy(
            training_step_results["model_outputs"],
            training_step_results["transformed_batch"][1],
            multi_head=self.multi_head,
        )[0]
        self.log("train/acc", top1_acc, on_epoch=True, logger=True)

        return training_step_results

    def on_non_accumulate_step(self) -> None:
        # This hook runs only after accumulation
        self._transform_gradients()

        if self._log_gradients:
            grad_norm_dict = self.grad_norm(1)
            for k, v in grad_norm_dict.items():
                self.log(f"gradients/{k}", v,
                         on_epoch=True,
                         logger=True)

        self.optimizer.step()
        self.optimizer.zero_grad()

        for callback in self._step_end_callbacks:
            TERMINATE = callback(self, self.step_tracker)
            if TERMINATE:
                self.trainer.should_stop = True

        self.step_tracker.end()
        self.log("train/loss",
                 self.step_tracker.cur_loss,
                 on_epoch=True,
                 logger=True)

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
                lr_lambda=[lambda epoch: self.hparams["lr_lambda"] ** epoch],
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

        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', top1_acc, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        path = self.log_dir + '/last.ckpt'
        if os.path.exists(self.log_dir + '/last.ckpt'):
            os.remove(self.log_dir + '/last.ckpt')
        self.trainer.save_checkpoint(self.log_dir + '/last.ckpt')

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


def create_lightning_module(model_name: str,
                            num_classes: int,
                            pretrained: bool = False,
                            ckpt: str = None,
                            freeze_extractor: bool = False,
                            *args,
                            **kwargs) -> LightningWrapper:
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
        acc = accuracy(output[:, j], target[:, j], topk=(1,))
        prec1.append(acc[0])
    return torch.mean(torch.Tensor(prec1))


def accuracy(output, target, topk=(1,), multi_head=False):
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
