from __future__ import annotations

import math

import torch
from torch import nn

from gradattack.defenses.defense import GradientDefense

try:
    from gradattack.trainingpipeline import TrainingPipeline
except (ImportError, ModuleNotFoundError):
    TrainingPipeline = object


def quantize_gradient(
    grad: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
    stochastic: bool = False,
    preserve_norm: bool = False,
) -> torch.Tensor:
    """Quantize gradient tensor with block-wise uniform quantization as an inversion defense.

    Parameters:
        grad: Input gradient tensor.
        nbits: Quantization bit width (e.g. 4 for INT4, 8 for INT8).
        block_size: Number of elements per independent scaling block.
        stochastic: If True, uses unbiased stochastic rounding instead of deterministic rounding.
        preserve_norm: If True, rescales reconstructed gradient to match original L2 norm.

    Returns:
        Quantized gradient tensor in original float dtype and shape.
    """
    if grad.numel() == 0:
        return grad.clone()

    orig_shape = grad.shape
    orig_dtype = grad.dtype
    orig_device = grad.device

    flat = grad.detach().to(torch.float32).reshape(-1)
    numel = flat.numel()

    orig_norm = torch.norm(flat, p=2) if preserve_norm else None

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_len), mode="constant", value=0.0)
    else:
        padded = flat

    blocks = padded.view(-1, block_size)
    qmax = (1 << (nbits - 1)) - 1  # 7 for 4-bit
    qmin = -(1 << (nbits - 1))  # -8 for 4-bit

    max_vals = torch.max(torch.abs(blocks), dim=1, keepdim=True)[0]
    scales = torch.clamp(max_vals / float(qmax), min=1e-8)

    scaled_blocks = blocks / scales

    if stochastic:
        floor_val = torch.floor(scaled_blocks)
        prob = scaled_blocks - floor_val
        rand = torch.rand_like(prob)
        quantized = floor_val + (rand < prob).float()
    else:
        quantized = torch.round(scaled_blocks)

    quantized = torch.clamp(quantized, qmin, qmax)
    recon_blocks = quantized * scales
    recon_flat = recon_blocks.view(-1)[:numel]

    if preserve_norm and orig_norm is not None and orig_norm > 1e-8:
        recon_norm = torch.norm(recon_flat, p=2)
        if recon_norm > 1e-8:
            recon_flat = recon_flat * (orig_norm / recon_norm)

    return recon_flat.view(orig_shape).to(device=orig_device, dtype=orig_dtype)


class GradQuantDefense(GradientDefense):
    """Applies block-wise uniform integer quantization (INT4/INT8) to communicated gradients.

    Disrupts continuous gradient matching loss functions used in gradient inversion attacks
    while simulating communication compression.
    """

    def __init__(
        self,
        nbits: int = 4,
        block_size: int = 64,
        stochastic: bool = False,
        preserve_norm: bool = False,
    ):
        """
        Args:
            nbits (int): Quantization bit-width (default: 4 for INT4).
            block_size (int): Block size for independent scaling factors (default: 64).
            stochastic (bool): Whether to use unbiased stochastic rounding (default: False).
            preserve_norm (bool): Whether to rescale output to preserve L2 norm (default: False).
        """
        super().__init__()
        self.nbits = nbits
        self.block_size = block_size
        self.stochastic = stochastic
        self.preserve_norm = preserve_norm

    def apply(self, pipeline: TrainingPipeline):
        """Apply gradient quantization defense to the training pipeline.

        Args:
            pipeline (TrainingPipeline): the training pipeline to protect.
        """
        super().apply(pipeline)

        def do_gradquant(model: nn.Module) -> nn.Module:
            parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
            for p in parameters:
                qg = quantize_gradient(
                    p.grad.data,
                    nbits=self.nbits,
                    block_size=self.block_size,
                    stochastic=self.stochastic,
                    preserve_norm=self.preserve_norm,
                )
                p.grad.data.copy_(qg)
            return model

        pipeline.model._grad_transformations.append(do_gradquant)

    def calc_compression_stats(
        self, model_or_grads: nn.Module | list[torch.Tensor]
    ) -> dict[str, float]:
        """Calculate theoretical wire transport savings relative to float32."""
        if isinstance(model_or_grads, nn.Module):
            grads = [p.grad for p in model_or_grads.parameters() if p.grad is not None]
        else:
            grads = model_or_grads

        orig_bytes = sum(g.numel() * g.element_size() for g in grads)
        quant_bytes = 0

        for g in grads:
            numel = g.numel()
            num_blocks = math.ceil(numel / float(self.block_size)) if numel > 0 else 0
            if self.nbits == 4:
                packed_bytes = math.ceil(numel / 2.0)
            elif self.nbits == 8:
                packed_bytes = numel
            else:
                packed_bytes = math.ceil(numel * self.nbits / 8.0)
            scale_bytes = num_blocks * 4  # float32 scales
            quant_bytes += packed_bytes + scale_bytes

        ratio = float(orig_bytes) / max(quant_bytes, 1)
        savings = (1.0 - (quant_bytes / max(orig_bytes, 1))) * 100.0

        return {
            "orig_bytes": orig_bytes,
            "quant_bytes": quant_bytes,
            "compression_ratio": ratio,
            "bandwidth_savings_pct": savings,
        }
