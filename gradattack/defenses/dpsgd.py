# DPSGD defense. The implementation of DPSGD is based on Opacus: https://opacus.ai/

import time
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from pytorch_lightning.core.datamodule import LightningDataModule
from gradattack.defenses.defense import GradientDefense
from gradattack.models import StepTracker
from gradattack.trainingpipeline import TrainingPipeline

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


class DPSGDDefense(GradientDefense):
    r"""

    Args:
        mini_batch_size (int): Training minibatch size. Used in the privacy accountant.
        sample_size (int): The size of the sample (dataset). Used in the privacy accountant.
        n_accumulation_steps (int): Accumulates gradients every k minibatches. i.e., n_accumulation_steps * mini_batch_size is the real batch size (also referred to as microbatch size).
        max_grad_norm (float): The maximum norm of the per-sample gradients. Any gradient with norm higher than this will be clipped to this value.
        noise_multiplier (float): The ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function to which the noise is added.
        delta_list (List[float], optional): The target list of delta. Used in the privacy accountant. Defaults to [1e-3, 1e-4, 1e-5].
        max_epsilon (float, optional): The privacy budget. Training will terminate if the privacy spent so far exceeds this budget. Defaults to 2.
        secure_rng (bool, optional): If on, it will use ``torchcsprng`` for secure random number generation. Comes with a significant performance cost. Defaults to False.
        freeze_extractor (bool, optional): If on, only finetune the classifier (the final fully-connected layers). Defaults to False.
    """
    def __init__(
        self,
        mini_batch_size: int,
        sample_size: int,
        n_accumulation_steps: int,
        max_grad_norm: float,
        noise_multiplier: float,
        delta_list: List[float] = [1e-3, 1e-4, 1e-5],
        max_epsilon: float = 2,
        secure_rng: bool = False,
        freeze_extractor: bool = False,
    ):

        super().__init__()
        self.delta_list = delta_list
        self.max_epsilon = max_epsilon

        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.sample_size = sample_size
        self.noise_multiplier = noise_multiplier
        self.n_accumulation_steps = n_accumulation_steps
        self.secure_rng = secure_rng

        if self.secure_rng:
            import torchcsprng as prng

            self.generator = prng.create_random_device_generator(
                "/dev/urandom")
        else:
            self.generator = None

    def apply(self, pipeline: TrainingPipeline) -> None:
        """Apply differentially private training to the given training pipeline. The implementation of DPSGD is based on Opacus: https://opacus.ai/

        Args:
            pipeline (TrainingPipeline): The training pipeline to which we apply the DPSGD defense.
        """
        super().apply(pipeline)

        def step_callback_dpsgd(model: pl.LightningModule,
                                timer: StepTracker = None) -> bool:
            r"""Per-step callback for DPSGD to compute and log the (epsilon, delta) privacy budget spent so far.

            Args:
                model (pl.LightningModule, optional): The Pytorch module to which we are attaching the privacy engine.
                timer (StepTracker, optional): The StepTracker which records per-step running time. Defaults to None.

            Returns:
                bool: A termination flag: if Ture, terminate training. We set the flag to be true if the spent privacy exceeds the privacy budget.
            """
            for delta in self.delta_list:
                epsilon, best_alpha = model.optimizer.privacy_engine.get_privacy_spent(
                    delta)
                # timer.pause()
                model.log(
                    f"step/ε/δ={delta}",
                    epsilon,
                    on_step=True,
                    prog_bar=True,
                    logger=True,
                )
                model.log(
                    f"step/α/δ={delta}",
                    best_alpha,
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                )
                # timer.cont()
                if self.max_epsilon is not None and epsilon >= self.max_epsilon:
                    return True
                else:
                    return False

        def do_dpsgd(model: pl.LightningModule):
            r"""Generate a DPSGD engine based on given configurations, and attach it to the model. Check out here for the official documentation of Opacus's privacy engine: https://opacus.ai/api/privacy_engine.html.

            Args:
                model (pl.LightningModule): The Pytorch module to which we are attaching the privacy engine.
            """

            assert self.n_accumulation_steps == model.trainer.accumulate_grad_batches
            privacy_engine = PrivacyEngine(
                model._model,
                sample_size=self.sample_size,
                batch_size=self.n_accumulation_steps * self.mini_batch_size,
                alphas=ORDERS,
                noise_multiplier=self.noise_multiplier,
                secure_rng=self.secure_rng,
                max_grad_norm=self.max_grad_norm,
            )
            privacy_engine.attach(model.optimizer)

        def do_DPSGD_sampler(dataset: LightningDataModule):
            r"""Samples batch elements according to the Gaussian Mechanism.

            Args:
                dataset (LightningDataModule): The LightningDataModule to be sampled.
            """
            dataset.batch_sampler = UniformWithReplacementSampler(
                num_samples=self.sample_size,
                sample_rate=self.mini_batch_size / len(dataset.train_set),
                generator=self.generator,
            )

        # Converts all BatchNorm modules to another module (defaults to GroupNorm) that is privacy compliant  # FIXME: should update the privacy accountant
        # pipeline.model._model = convert_batchnorm_modules(
        #     pipeline.model._model)

        pipeline._data_transformations.append(do_DPSGD_sampler)
        pipeline.model._opt_transformations.append(do_dpsgd)
        pipeline.model._step_end_callbacks.append(step_callback_dpsgd)
        pipeline.model._log_gradients = True
