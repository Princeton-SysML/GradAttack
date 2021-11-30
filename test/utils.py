import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn.functional import log_softmax


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * log_softmax(pred, dim=-1), 1))


early_stop_callback = EarlyStopping(monitor="epoch/val_loss",
                                    min_delta=0.00,
                                    patience=10,
                                    verbose=False,
                                    mode="min")

augment = {
    "hflip": False,
    "crop": True,
    "rotation": 0,
    "color_jitter": None,
    "affine": False,
}
