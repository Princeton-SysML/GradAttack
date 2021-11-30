from argparse import Namespace

vanilla_hparams = {
    "optimizer": "SGD",
    "lr": 0.05,
    "weight_decay": 0.0005,
    "momentum": 0.9,
    "nesterov": False,
    "lr_scheduler": "ReduceLROnPlateau",
    "tune_on_val": 0.02,
    "batch_size": 128,
    "lr_factor": 0.5,
}