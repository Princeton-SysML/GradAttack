import numpy as np
import torch
from torch.distributions.dirichlet import Dirichlet
from torch.nn.functional import one_hot
from torch.utils.data.dataset import Dataset

import torchcsprng as csprng
from gradattack.defenses.defense import GradientDefense
from gradattack.trainingpipeline import TrainingPipeline


class InstahideDefense(GradientDefense):
    def __init__(self,
                 mix_dataset: Dataset,
                 klam: int,
                 upper_bound: float,
                 lower_bound: float,
                 device: torch.device = None,
                 use_csprng: bool = True,
                 cs_prng: torch.Generator = None):
        """
        Args:
            mix_dataset (Dataset): the original training dataset
            klam (int): the numebr of data points to mix for each encoding
            upper_bound (float): the upper bound for mixing coefficients
            lower_bound (float): the lower bound for mixing coefficients
            device (torch.device, optional): the device to run training on. Defaults to None.
            use_csprng (bool, optional): whether to use cryptographically secure pseudorandom number generator. Defaults to True.
            cs_prng (torch.Generator, optional): the cryptographically secure pseudorandom number generator. Defaults to None.
        """

        super().__init__()
        self.klam = klam
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.device = device
        self.alpha = [1.0] * klam
        self.alpha[0] = 3.0

        self.mix_dataset = mix_dataset
        self.x_values, self.y_values = None, None
        if isinstance(self.mix_dataset, torch.utils.data.Subset):
            all_classes = list(self.mix_dataset.dataset.classes)
        elif isinstance(self.mix_dataset, torch.utils.data.dataset.Dataset):
            all_classes = list(self.mix_dataset.classes)
        self.num_classes = len(all_classes)
        self.dataset_size = len(self.mix_dataset)

        self.lambda_sampler_single = Dirichlet(torch.tensor(self.alpha))
        self.lambda_sampler_whole = Dirichlet(
            torch.tensor(self.alpha).repeat(self.dataset_size, 1))
        self.use_csprng = use_csprng

        if self.use_csprng:
            if cs_prng is None:
                self.cs_prng = csprng.create_random_device_generator()
            else:
                self.cs_prng = cs_prng

    # @profile
    def generate_mapping(self, return_tensor=True):
        """Generate the mapping and coefficients for InstaHide

        Args:
            return_tensor (bool, optional): whether to return the results in the format of PyTorch tensor. Defaults to True.

        Returns:
            (numpy.array): the mapping and coefficients array
        """
        if not self.use_csprng:
            lams = np.random.dirichlet(alpha=self.alpha,
                                       size=self.dataset_size)

            selects = np.asarray([
                np.random.permutation(self.dataset_size)
                for _ in range(self.klam)
            ])
            selects = np.transpose(selects)

            for i in range(self.dataset_size):
                # enforce that k images are non-repetitive
                while len(set(selects[i])) != self.klam:
                    selects[i] = np.random.randint(0, self.dataset_size,
                                                   self.klam)
                if self.klam > 1:
                    while (lams[i].max() > self.upper_bound) or (
                            lams[i].min() <
                            self.lower_bound):  # upper bounds a single lambda
                        lams[i] = np.random.dirichlet(alpha=self.alpha)
            if return_tensor:
                return (
                    torch.from_numpy(lams).float().to(self.device),
                    torch.from_numpy(selects).long().to(self.device),
                )
            else:
                return np.asarray(lams), np.asarray(selects)

        else:
            lams = self.lambda_sampler_whole.sample().to(self.device)
            selects = torch.stack([
                torch.randperm(self.dataset_size,
                               device=self.device,
                               generator=self.cs_prng)
                for _ in range(self.klam)
            ])
            selects = torch.transpose(selects, 0, 1)

            for i in range(self.dataset_size):
                # enforce that k images are non-repetitive
                while len(set(selects[i])) != self.klam:
                    selects[i] = torch.randint(0,
                                               self.dataset_size,
                                               self.klam,
                                               generator=self.cs_prng)
                if self.klam > 1:
                    while (lams[i].max() > self.upper_bound) or (
                            lams[i].min() <
                            self.lower_bound):  # upper bounds a single lambda
                        lams[i] = self.lambda_sampler_single.sample().to(
                            self.device)
            if return_tensor:
                return lams, selects
            else:
                return np.asarray(lams), np.asarray(selects)

    def instahide_batch(
        self,
        inputs: torch.tensor,
        lams_b: float,
        selects_b: np.array,
    ):
        """Generate an InstaHide batch.

        Args:
            inputs (torch.tensor): the original batch (only its size is used)
            lams_b (float): the coefficients for InstaHide
            selects_b (np.array): the mappings for InstaHide

        Returns:
            (torch.tensor): the InstaHide images and labels
        """
        mixed_x = torch.zeros_like(inputs)
        mixed_y = torch.zeros((len(inputs), self.num_classes),
                              device=self.device)

        for i in range(self.klam):
            x = torch.index_select(self.x_values, 0, selects_b[:, i]).clone()
            ys_onehot = torch.index_select(self.y_values, 0,
                                           selects_b[:, i]).clone()
            # need to broadcast here to make row-wise multiplication work
            # see: https://stackoverflow.com/questions/53987906/how-to-multiply-a-tensor-row-wise-by-a-vector-in-pytorch
            mixed_x += lams_b[:, i][:, None, None, None] * x
            mixed_y += lams_b[:, i][:, None] * ys_onehot

        # Apply InstaHide random sign flip
        sign = torch.randint(2, size=list(mixed_x.shape),
                             device=self.device) * 2.0 - 1
        mixed_x *= sign.float()

        return mixed_x, mixed_y

    def apply(self, pipeline: TrainingPipeline):
        super().apply(pipeline)

        self.cur_selects, self.cur_selects = None, None

        def regenerate_mappings(module):
            """Regenerate InstaHide mapping and coefficients at the begining of each epoch

            Args:
                module (pl.LightningModule): the pl.LightningModule for training
            """
            self.cur_lams, self.cur_selects = self.generate_mapping(
                return_tensor=True)

            # IMPORTANT! Use new augmentations for every epoch
            self.x_values = torch.stack([data[0] for data in self.mix_dataset
                                         ]).to(self.device)

            if self.y_values is None:
                self.y_values = torch.from_numpy(
                    np.asarray([data[1]
                                for data in self.mix_dataset])).to(self.device)
                if len(self.y_values.shape) == 1:
                    self.y_values = one_hot(
                        self.y_values, num_classes=self.num_classes).float()

        pipeline.model._on_train_epoch_start_callbacks.append(
            regenerate_mappings)

        regenerate_mappings(pipeline.model)

        # @profile
        def do_instahide(batch, batch_idx, use_tensor=True, *args, **kwargs):
            """Run InstaHide for a given batch

            Args:
                batch ((torch.tensor)): the original batch (only the size information is used)
                batch_idx (int): index of the batch; used to slice the whole mapping array
                use_tensor (bool, optional): whether the mapping and coefficients are in the format of PyTorch tensor. Defaults to True.
            """
            inputs, targets = batch
            batch_size = len(inputs)
            start_idx = batch_idx * batch_size
            batch_device = batch[0].device
            if use_tensor:
                batch_indices = torch.arange(start_idx,
                                             start_idx + batch_size,
                                             device=self.device).long()
                lams_b = torch.index_select(self.cur_lams, 0, batch_indices)
                selects_b = torch.index_select(self.cur_selects, 0,
                                               batch_indices)
            else:
                batch_indices = range(start_idx, start_idx + batch_size)
                lams_b = (torch.from_numpy(
                    np.asarray([self.cur_lams[i] for i in batch_indices
                                ])).float().to(batch_device))
                selects_b = (torch.from_numpy(
                    np.asarray([self.cur_selects[i] for i in batch_indices
                                ])).long().to(batch_device))

            mixed_inputs, mixed_targets = self.instahide_batch(
                inputs, lams_b, selects_b)
            return (mixed_inputs, mixed_targets)

        pipeline.model._batch_transformations.append(do_instahide)
        pipeline.model.log_train_acc = False

        if pipeline.model.multi_class:
            pipeline.model._val_loss_metric = torch.nn.BCEWithLogitsLoss(
                reduction="mean")
        else:
            pipeline.model._val_loss_metric = torch.nn.CrossEntropyLoss(
                reduction="mean")
