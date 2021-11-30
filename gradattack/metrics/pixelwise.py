import torch


def MeanPixelwiseError(x: torch.tensor, y: torch.tensor):
    """Computes the normalized square root of the average squared pixelwise error."""
    pixelwise_l2 = (x - y).pow(2)
    pixelwise_l2 /= pixelwise_l2.max()
    return torch.sqrt(torch.mean(pixelwise_l2))
