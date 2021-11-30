import copy
from typing import Any, Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from gradattack.metrics.gradients import CosineSimilarity, L2Diff
from gradattack.metrics.pixelwise import MeanPixelwiseError
from gradattack.models import LightningWrapper
from gradattack.trainingpipeline import TrainingPipeline
from gradattack.utils import patch_image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

# DEFAULT_HPARAMS = {
#     "optimizer": "Adam",
#     "lr_scheduler": True,
#     "lr": 0.1,
#     "total_variation": 1e-1,
#     "l2": 0,
#     "bn_reg": 0,
#     "first_bn_multiplier": 10,
#     # If true, will apply image priors on the absolute value of the recovered images
#     "signed_image": False,
#     "signed_gradients": True,
#     "boxed": True,
#     "attacker_eval_mode": True,
#     "recipe": 'Geiping'
# }


class BNFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (input[0].permute(1, 0, 2,
                                3).contiguous().view([nch,
                                                      -1]).var(1,
                                                               unbiased=False))

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.mean = mean
        self.var = var
        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def l2_norm(x, signed_image=False):
    if signed_image:
        x = torch.abs(x)
    batch_size = len(x)
    loss_l2 = torch.norm(x.view(batch_size, -1), dim=1).mean()
    return loss_l2


def total_variation(x, signed_image=False):
    """Anisotropic TV."""
    if signed_image:
        x = torch.abs(x)
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


class DummyGradientDataset(Dataset):
    def __init__(self, num_values: int):
        self.num_values = num_values

    def __len__(self):
        return self.num_values

    def __getitem__(self, idx: int):
        return (1, 2)


class GradientReconstructor(pl.LightningModule):
    def __init__(
        self,
        pipeline: TrainingPipeline,
        ground_truth_inputs: tuple,
        ground_truth_gradients: tuple,
        ground_truth_labels: tuple,
        intial_reconstruction: torch.tensor = None,
        reconstruct_labels=False,
        attack_loss_metric: Callable = CosineSimilarity(),
        mean_std: tuple = (0.0, 1.0),
        num_iterations=10000,
        optimizer: str = "Adam",
        lr_scheduler: bool = True,
        lr: float = 0.1,
        total_variation: float = 1e-1,
        l2: float = 0,
        bn_reg: float = 0,
        first_bn_multiplier: float = 1,
        signed_image: bool = False,
        signed_gradients: bool = True,
        boxed: bool = True,
        attacker_eval_mode: bool = True,
        recipe: str = 'Geiping',
        BN_exact: bool = False,
        grayscale: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters("optimizer", "lr_scheduler", "lr",
                                  "total_variation", "l2", "bn_reg",
                                  "first_bn_multiplier", "signed_image",
                                  "signed_gradients", "boxed",
                                  "attacker_eval_mode", "recipe")

        # Need to clone model - otherwise, it could be modified by training
        self._model = copy.deepcopy(pipeline.model)
        self.BN_exact = BN_exact
        self.recipe = recipe

        self.automatic_optimization = False
        self._attack_loss_metric = attack_loss_metric

        self.num_iterations = num_iterations
        self.num_images = len(ground_truth_inputs)
        self.image_shape = ground_truth_inputs[0].shape
        self.mean_std = mean_std
        self.grayscale = grayscale
        self.ground_truth_inputs = ground_truth_inputs
        self.ground_truth_labels = ground_truth_labels
        self.ground_truth_gradients = ground_truth_gradients
        self.save_hyperparameters({
            "num_images": self.num_images,
            "image_shape": self.image_shape
        })

        if self.grayscale:
            self.best_guess_grayscale = (
                torch.nn.Parameter(intial_reconstruction)
                if intial_reconstruction is not None else torch.nn.Parameter(
                    torch.randn(
                        (self.num_images, 1, self.image_shape[1],
                         self.image_shape[2]),
                        requires_grad=True,
                        device=self.device,
                    )))
            self.best_guess = self.best_guess_grayscale.repeat(1, 3, 1, 1)
        else:
            self.best_guess = (torch.nn.Parameter(intial_reconstruction)
                               if intial_reconstruction is not None else
                               torch.nn.Parameter(
                                   torch.randn(
                                       (self.num_images, *self.image_shape),
                                       requires_grad=True,
                                       device=self.device,
                                   )))
        self.labels = ground_truth_labels
        if reconstruct_labels:
            self.labels = torch.nn.Parameter(
                torch.randn(
                    (self.num_images, len(ground_truth_gradients[-1])),
                    requires_grad=True,
                    device=self.device,
                ))
            self._reconstruct_labels = True

            class loss_fn(torch.nn.Module):
                def __call__(self, pred, labels):
                    if len(labels.shape) >= 2:
                        labels = torch.nn.functional.softmax(labels, dim=-1)
                        return torch.mean(
                            torch.sum(
                                -labels *
                                torch.nn.functional.log_softmax(pred, dim=-1),
                                1,
                            ))
                    else:
                        return torch.nn.functional.cross_entropy(pred, labels)

            self._model._training_loss_metric = None
            self._model._training_loss_metric = loss_fn()
        else:
            self._reconstruct_labels = False

        self._batch_transferred = False

        self.loss_r_feature_layers = []

        for module in self._model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.loss_r_feature_layers.append(BNFeatureHook(module))

    def forward(self, x):
        return self._model(x)

    @staticmethod
    def guess_labels(gradients):
        """Uses the method from iDLG to retrieve the labels given gradients"""
        labels = (torch.min(gradients[-2], dim=1)[0] < 0).nonzero(
            as_tuple=False)
        return labels.flatten()

    def on_train_start(self) -> None:
        self.logger.experiment.add_image(
            "Ground truth",
            patch_image(
                self.normalize_image(self.ground_truth_inputs),
                (self.ground_truth_inputs.shape[2],
                 self.ground_truth_inputs.shape[3]),
            ),
            global_step=self.global_step,
        )
        if self.hparams["signed_image"]:
            self.logger.experiment.add_image(
                "Abs(Ground truth)",
                patch_image(
                    self.normalize_image(torch.abs(self.ground_truth_inputs)),
                    (
                        self.ground_truth_inputs.shape[2],
                        self.ground_truth_inputs.shape[3],
                    ),
                ),
                global_step=self.global_step,
            )

        self.ground_truth_inputs = self.ground_truth_inputs.to(self.device)
        self.mean_std = (
            self.mean_std[0].to(self.device),
            self.mean_std[1].to(self.device),
        )
        self._model = self._model.to(self.device)

        self.optimizer.zero_grad()
        self.zero_grad()
        self._model.zero_grad()

        # collet BN stats from the original batch
        recovered_gradients, step_results = self._model.get_batch_gradients(
            (self.ground_truth_inputs, self.ground_truth_labels),
            create_graph=True,
            apply_transforms=False,
            clone_gradients=False,
            eval_mode=self.hparams["attacker_eval_mode"],
        )

        self.mean_gt = [
            mod.mean for (idx, mod) in enumerate(self.loss_r_feature_layers)
        ]
        self.var_gt = [
            mod.var for (idx, mod) in enumerate(self.loss_r_feature_layers)
        ]

        for module in self._model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # print(module.running_mean)
                break

    def normalize_image(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def train_dataloader(self) -> Any:
        return DataLoader(
            DummyGradientDataset(num_values=self.num_iterations, ), )

    def transfer_batch_to_device(self, batch: Any,
                                 device: Optional[torch.device]) -> Any:
        if not self._batch_transferred:
            self.ground_truth_labels = self.ground_truth_labels.detach().to(
                self.device)
            self.ground_truth_gradients = tuple(
                x.detach().to(self.device)
                for x in self.ground_truth_gradients)
            self._batch_transferred = True
        return (self.ground_truth_gradients, self.ground_truth_labels)

    def training_step(self, batch, *args):
        input_gradients, labels = batch

        def _closure():
            self.optimizer.zero_grad()
            self.zero_grad()
            self._model.zero_grad()
            if self.grayscale:
                self.best_guess = self.best_guess_grayscale.repeat(1, 3, 1, 1)
            recovered_gradients, step_results = self._model.get_batch_gradients(
                (self.best_guess, self.labels),
                create_graph=True,
                apply_transforms=False,
                clone_gradients=False,
                eval_mode=self.hparams["attacker_eval_mode"],
                BN_exact=self.BN_exact,
                attacker=True,
            )
            if self.recipe == 'Geiping':
                reconstruction_loss = self._attack_loss_metric(
                    recovered_gradients, input_gradients)
                reconstruction_loss += self.hparams[
                    "total_variation"] * total_variation(
                        self.best_guess, self.hparams["signed_image"])
                reconstruction_loss += self.hparams["l2"] * l2_norm(
                    self.best_guess, self.hparams["signed_image"])
            elif self.recipe == 'Zhu':  ## TODO: test
                self._attack_loss_metric = L2Diff()
                reconstruction_loss = self._attack_loss_metric(
                    recovered_gradients, input_gradients)

            recon_mean = [
                mod.mean
                for (idx, mod) in enumerate(self.loss_r_feature_layers)
            ]
            recon_var = [
                mod.var for (idx, mod) in enumerate(self.loss_r_feature_layers)
            ]

            if self.hparams["bn_reg"] > 0:
                rescale = [self.hparams["first_bn_multiplier"]] + [
                    1.0 for _ in range(len(self.loss_r_feature_layers) - 1)
                ]
                loss_r_feature = sum([
                    mod.r_feature * rescale[idx]
                    for (idx, mod) in enumerate(self.loss_r_feature_layers)
                ])

                reconstruction_loss += self.hparams["bn_reg"] * loss_r_feature

            self.logger.experiment.add_scalar("Loss",
                                              step_results["loss"],
                                              global_step=self.global_step)
            self.logger.experiment.add_scalar(
                "Reconstruction Metric Loss",
                reconstruction_loss,
                global_step=self.global_step,
            )

            if self.global_step % 100 == 0:
                for i in range(len(self.loss_r_feature_layers)):
                    self.logger.experiment.add_scalar(
                        f"BN_loss/layer_{i}_mean_loss",
                        torch.sqrt(
                            sum((recon_mean[i] - self.mean_gt[i])**2) /
                            len(recon_mean[i])),
                        global_step=self.global_step,
                    )
                    self.logger.experiment.add_scalar(
                        f"BN_loss/layer_{i}_var_loss",
                        torch.sqrt(
                            sum((recon_var[i] - self.var_gt[i])**2) /
                            len(recon_mean[i])),
                        global_step=self.global_step,
                    )
            self.manual_backward(reconstruction_loss, self.optimizer)
            if self.hparams["signed_gradients"]:
                if self.grayscale:
                    self.best_guess_grayscale.grad.sign_()
                else:
                    self.best_guess.grad.sign_()
            return reconstruction_loss

        reconstruction_loss = self.optimizer.step(closure=_closure)
        if self.hparams["lr_scheduler"]:
            self.lr_scheduler.step()
        if self.hparams["boxed"]:
            dm, ds = self.mean_std
            self.best_guess.data = torch.max(
                torch.min(self.best_guess, (1 - dm) / ds), -dm / ds)
        if self.global_step % 50 == 0:
            with torch.no_grad():
                self.logger.experiment.add_image(
                    "Reconstruction",
                    patch_image(
                        self.normalize_image(self.best_guess),
                        (
                            self.ground_truth_inputs.shape[2],
                            self.ground_truth_inputs.shape[3],
                        ),
                    ),
                    global_step=self.global_step,
                )
                if self.hparams["signed_image"]:
                    self.logger.experiment.add_image(
                        "Abs(Reconstruction)",
                        patch_image(
                            self.normalize_image(torch.abs(self.best_guess)),
                            (
                                self.ground_truth_inputs.shape[2],
                                self.ground_truth_inputs.shape[3],
                            ),
                        ),
                        global_step=self.global_step,
                    )
                psnrs = [
                    torchmetrics.functional.psnr(a, b)
                    for (a,
                         b) in zip(self.best_guess, self.ground_truth_inputs)
                ]
                avg_psnr = sum(psnrs) / self.num_images
                self.logger.experiment.add_scalar("Avg. PSNR",
                                                  avg_psnr,
                                                  global_step=self.global_step)

                rmses = [
                    MeanPixelwiseError(a, b)
                    for (a,
                         b) in zip(self.best_guess, self.ground_truth_inputs)
                ]
                avg_rmse = sum(rmses) / self.num_images
                self.logger.experiment.add_scalar("Avg. Pixelwise Error",
                                                  avg_rmse,
                                                  global_step=self.global_step)

                if self.hparams["signed_image"]:
                    abs_rmses = [
                        MeanPixelwiseError(torch.abs(a), torch.abs(b))
                        for (a, b) in zip(self.best_guess,
                                          self.ground_truth_inputs)
                    ]
                    abs_avg_rmse = sum(abs_rmses) / self.num_images
                    self.logger.experiment.add_scalar(
                        "Avg. Absolute Pixelwise Error",
                        abs_avg_rmse,
                        global_step=self.global_step,
                    )

        return reconstruction_loss

    def configure_optimizers(self):
        self.best_guess = self.best_guess.to(self.device)
        self.labels = self.labels.to(self.device)
        if self.grayscale:
            parameters = ([
                self.best_guess_grayscale, self.labels
            ] if self._reconstruct_labels else [self.best_guess_grayscale])
        else:
            parameters = ([self.best_guess, self.labels]
                          if self._reconstruct_labels else [self.best_guess])
        if self.hparams["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(parameters,
                                              lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=5e-4,
            )
        self.configure_lr_scheduler()

        return [self.optimizer]

    def configure_lr_scheduler(self):
        if self.hparams["lr_scheduler"]:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    self.num_iterations // 2.667,
                    self.num_iterations // 1.6,
                    self.num_iterations // 1.142,
                ],
                gamma=0.1,
            )  # 3/8 5/8 7/8
