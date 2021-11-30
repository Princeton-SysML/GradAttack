"""Some helper functions for PyTorch, including:
    - parse_args, parse_augmentation: parse for command-line options and arguments
    - get_mean_and_std: calculate the mean and std value of dataset.
    - save_fig: save an image array to file.
    - patch_image: patch a batch of images for better visualization.
    - cross_entropy_for_onehot: cross-entropy loss for soft labels.
"""

import argparse
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import log_softmax


def parse_args():
    parser = argparse.ArgumentParser(description="gradattack training")
    parser.add_argument("--gpuid", default="0", type=int, help="gpu id to use")
    parser.add_argument("--model",
                        default="ResNet18",
                        type=str,
                        help="name of model")
    parser.add_argument("--data",
                        default="CIFAR10",
                        type=str,
                        help="name of dataset")
    parser.add_argument(
        "--results_dir",
        default="./results",
        type=str,
        help="directory to save attack results",
    )
    parser.add_argument("--n_epoch",
                        default=200,
                        type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="batch size")
    parser.add_argument("--optimizer",
                        default="SGD",
                        type=str,
                        help="which optimizer")
    parser.add_argument("--lr", default=0.05, type=float, help="initial lr")
    parser.add_argument("--decay",
                        default=5e-4,
                        type=float,
                        help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--scheduler",
                        default="ReduceLROnPlateau",
                        type=str,
                        help="which scheduler")
    parser.add_argument("--lr_step",
                        default=30,
                        type=int,
                        help="reduce LR per ? epochs")
    parser.add_argument("--lr_lambda",
                        default=0.95,
                        type=float,
                        help="lambda of LambdaLR scheduler")
    parser.add_argument("--lr_factor",
                        default=0.5,
                        type=float,
                        help="factor of lr reduction")
    parser.add_argument("--disable_early_stopping",
                        dest="early_stopping",
                        action="store_false")
    parser.add_argument("--patience",
                        default=10,
                        type=int,
                        help="patience for early stopping")
    parser.add_argument(
        "--tune_on_val",
        default=0.02,
        type=float,
        help=
        "fraction of validation data. If set to 0, use test data as the val data",
    )
    parser.add_argument("--log_auc", dest="log_auc", action="store_true")
    parser.add_argument("--logname",
                        default="vanilla",
                        type=str,
                        help="log name")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--ckpt",
                        default=None,
                        type=str,
                        help="directory for ckpt")
    parser.add_argument(
        "--freeze_extractor",
        dest="freeze_extractor",
        action="store_true",
        help="Whether only training the fc layer",
    )

    # Augmentation
    parser.add_argument(
        "--dis_aug_crop",
        dest="aug_crop",
        action="store_false",
        help="Whether to apply random cropping",
    )
    parser.add_argument(
        "--dis_aug_hflip",
        dest="aug_hflip",
        action="store_false",
        help="Whether to apply horizontally flipping",
    )
    parser.add_argument(
        "--aug_affine",
        dest="aug_affine",
        action="store_true",
        help="Enable random affine",
    )
    parser.add_argument(
        "--aug_rotation",
        type=float,
        default=0,
        help="Maximum degree of the random rotation augmentatiom",
    )
    parser.add_argument("--aug_colorjitter",
                        nargs="*",
                        help="brightness, contrast, saturation, hue")

    # Mixup or InstaHide
    parser.add_argument("--defense_mixup",
                        dest="defense_mixup",
                        action="store_true")
    parser.add_argument("--defense_instahide",
                        dest="defense_instahide",
                        action="store_true")
    parser.add_argument("--klam",
                        default=4,
                        type=int,
                        help="How many images to mix with")
    parser.add_argument("--c_1",
                        default=0,
                        type=float,
                        help="Lower bound of mixing coefs")
    parser.add_argument("--c_2",
                        default=1,
                        type=float,
                        help="Upper bound of mixing coefs")
    parser.add_argument("--use_csprng", dest="use_csprng", action="store_true")
    # GradPrune
    parser.add_argument("--defense_gradprune",
                        dest="defense_gradprune",
                        action="store_true")
    parser.add_argument("--p", default=0.9, type=float, help="prune ratio")
    # DPSGD
    parser.add_argument("--defense_DPSGD",
                        dest="defense_DPSGD",
                        action="store_true")
    parser.add_argument(
        "--delta_list",
        nargs="*",
        default=[1e-3, 1e-4, 1e-5],
        type=float,
        help="Failure prob of DP",
    )
    parser.add_argument("--max_epsilon",
                        default=2,
                        type=float,
                        help="Privacy budget")
    parser.add_argument(
        "--max_grad_norm",
        default=1,
        type=float,
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument("--noise_multiplier",
                        default=1,
                        type=float,
                        help="Noise multiplier")

    parser.add_argument(
        "--n_accumulation_steps",
        default=1,
        type=int,
        help="Run optimization per ? step",
    )
    parser.add_argument(
        "--secure_rng",
        dest="secure_rng",
        action="store_true",
        help=
        "Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    # For attack
    parser.add_argument("--reconstruct_labels", action="store_true")
    parser.add_argument("--attack_lr",
                        default=0.1,
                        type=float,
                        help="learning rate for attack")
    parser.add_argument("--tv", default=0.1, type=float, help="coef. for tv")
    parser.add_argument("--mini",
                        action="store_true",
                        help="use the mini set for attack")
    parser.add_argument("--large",
                        action="store_true",
                        help="use the large set for attack")
    parser.add_argument("--data_seed",
                        default=None,
                        type=int,
                        help="seed to select attack subset")
    parser.add_argument("--attack_epoch",
                        default=0,
                        type=int,
                        help="iterations for the attack")
    parser.add_argument(
        "--bn_reg",
        default=0,
        type=float,
        help="coef. for batchnorm regularization term",
    )
    parser.add_argument(
        "--attacker_eval_mode",
        action="store_true",
        help="use eval model for gradients calculation for attack",
    )
    parser.add_argument(
        "--defender_eval_mode",
        action="store_true",
        help="use eval model for gradients calculation for training",
    )
    parser.add_argument(
        "--BN_exact",
        action="store_true",
        help="use training batch's mean and var",
    )

    args = parser.parse_args()

    hparams = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.decay,
        "momentum": args.momentum,
        "nesterov": args.nesterov,
        "lr_scheduler": args.scheduler,
        "tune_on_val": args.tune_on_val,
        "batch_size": args.batch_size,
    }

    if args.scheduler == "StepLR":
        hparams["lr_step"] = args.lr_step
        hparams["lr_factor"] = args.lr_factor
    elif args.scheduler == "MultiStepLR":
        hparams["lr_step"] = [100, 150]
        hparams["lr_factor"] = args.lr_factor
    elif args.scheduler == "LambdaLR":
        hparams["lr_lambda"] = args.lr_lambda
    elif args.scheduler == "ReduceLROnPlateau":
        hparams["lr_factor"] = args.lr_factor

    attack_hparams = {
        # Bool settings
        "reconstruct_labels": args.reconstruct_labels,
        "signed_image": args.defense_instahide,
        "mini": args.mini,
        "large": args.large,

        # BN settings
        "BN_exact": args.BN_exact,
        "attacker_eval_mode": args.attacker_eval_mode,
        "defender_eval_mode": args.defender_eval_mode,

        # Hyper-params
        "total_variation": args.tv,
        "epoch": args.attack_epoch,
        "bn_reg": args.bn_reg,
        "attack_lr": args.attack_lr,
    }

    return args, hparams, attack_hparams


def parse_augmentation(args):
    return {
        "hflip":
        args.aug_hflip,
        "crop":
        args.aug_crop,
        "rotation":
        args.aug_rotation,
        "color_jitter": [float(i) for i in args.aug_colorjitter]
        if args.aug_colorjitter is not None else None,
        "affine":
        args.aug_affine,
    }


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def save_fig(img_arr, fname, save_npy=False, save_fig=True):
    if torch.is_tensor(img_arr):
        img_arr = img_arr.cpu().detach()
    if len(img_arr.shape) == 3:
        img_arr = np.transpose(img_arr, (1, 2, 0))
    elif len(img_arr.shape) == 4:
        img_arr = np.transpose(img_arr, (0, 2, 3, 1))
    # print(img_arr.shape)
    if save_npy:
        np.save(re.sub(r"(jpg|png|pdf)\b", "npy", fname), img_arr)
    if save_fig:
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        plt.imshow(img_arr)
        plt.axis("off")
        plt.savefig(fname, bbox_inches="tight")
        plt.show()


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    # view = [1] * len(x.shape)
    # view[1] = -1
    # print(view)
    x = (x - bn_mean) / torch.sqrt(bn_var + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var != 0).float()
    return x


class StandardizeLayer(nn.Module):
    def __init__(self, bn_stats=None, n_features=1024):
        super(StandardizeLayer, self).__init__()
        if bn_stats is None:
            mean = np.zeros(n_features, dtype=np.float32)
            var = np.ones(n_features, dtype=np.float32)
            bn_stats = (torch.from_numpy(mean), torch.from_numpy(var))
        self.bn_stats = bn_stats

    def forward(self, x):
        self.bn_stats = (self.bn_stats[0].to(x.device),
                         self.bn_stats[1].to(x.device))
        # print(x.shape)
        return standardize(x, self.bn_stats)


def cross_entropy_for_onehot(pred, target):
    # Prediction should be logits instead of probs
    return torch.mean(torch.sum(-target * log_softmax(pred, dim=-1), 1))


def patch_image(x, dim=(32, 32)):
    """Patch a batch of images for better visualization, keeping images in rows of 8. If the number of images isn't divisible by 8, pads the remaining space with whitespace."""
    if torch.is_tensor(x):
        x = x.cpu().detach().numpy()
    batch_size = len(x)
    if batch_size == 1:
        return x[0]
    else:
        if batch_size % 8 != 0:
            # Pad batch with white squares
            pad_size = int(math.ceil(batch_size / 8) * 8) - batch_size
            x = np.append(x, np.zeros((pad_size, *x[0].shape)), axis=0)
        batch_size = len(x)
        x = np.transpose(x, (0, 2, 3, 1))
        if int(np.sqrt(batch_size))**2 == batch_size:
            s = int(np.sqrt(batch_size))
            x = np.reshape(x, (s, s, dim[0], dim[1], 3))
            x = np.concatenate(x, axis=2)
            x = np.concatenate(x, axis=0)
            x = np.transpose(x, (2, 0, 1))
        else:
            x = np.reshape(x, (8, batch_size // 8, dim[0], dim[1], 3))
            x = np.concatenate(x, axis=2)
            x = np.concatenate(x, axis=0)
            x = np.transpose(x, (2, 0, 1))
    return x
