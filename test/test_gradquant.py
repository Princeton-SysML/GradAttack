import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn

from gradattack.defenses.gradquant import GradQuantDefense, quantize_gradient


class DummyPipeline:
    def __init__(self, model):
        self.model = model
        self.applied_defenses = []


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self._grad_transformations = []


class TestGradQuant(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantize_gradient_roundtrip(self):
        grad = torch.randn(16, 8, 3, 3, dtype=torch.float32)
        qg = quantize_gradient(grad, nbits=4, block_size=64, stochastic=False)

        self.assertEqual(qg.shape, grad.shape)
        self.assertEqual(qg.dtype, grad.dtype)

        cos_sim = torch.cosine_similarity(grad.flatten(), qg.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.95)

    def test_quantize_int8(self):
        grad = torch.randn(128, dtype=torch.float32)
        qg = quantize_gradient(grad, nbits=8, block_size=32)
        cos_sim = torch.cosine_similarity(grad.flatten(), qg.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.99)

    def test_stochastic_quantization(self):
        grad = torch.randn(64, dtype=torch.float32)
        samples = [
            quantize_gradient(grad, nbits=4, block_size=64, stochastic=True)
            for _ in range(50)
        ]
        mean_sample = torch.stack(samples).mean(dim=0)
        error = torch.norm(mean_sample - grad) / torch.norm(grad)
        self.assertLess(error.item(), 0.35)

    def test_preserve_norm(self):
        grad = torch.randn(100, dtype=torch.float32) * 3.0
        orig_norm = torch.norm(grad, 2).item()
        qg = quantize_gradient(grad, nbits=4, block_size=32, preserve_norm=True)
        qg_norm = torch.norm(qg, 2).item()
        self.assertAlmostEqual(orig_norm, qg_norm, places=4)

    def test_defense_apply_to_pipeline(self):
        model = DummyModel()
        # populate gradients
        inputs = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(model.fc(inputs), labels)
        loss.backward()

        orig_grad = model.fc.weight.grad.clone()

        pipeline = DummyPipeline(model)
        defense = GradQuantDefense(nbits=4, block_size=32)
        defense.apply(pipeline)

        self.assertIn("GradQuantDefense", pipeline.applied_defenses)
        self.assertEqual(len(model._grad_transformations), 1)

        # Apply transformation
        transform_fn = model._grad_transformations[0]
        transformed_model = transform_fn(model)

        self.assertEqual(transformed_model.fc.weight.grad.shape, orig_grad.shape)
        # Cosine similarity should be high but not identical
        cos_sim = torch.cosine_similarity(
            orig_grad.flatten(), transformed_model.fc.weight.grad.flatten(), dim=0
        ).item()
        self.assertGreater(cos_sim, 0.95)

    def test_calc_compression_stats(self):
        defense = GradQuantDefense(nbits=4, block_size=64)
        grads = [torch.randn(128), torch.randn(256)]
        stats = defense.calc_compression_stats(grads)

        self.assertIn("compression_ratio", stats)
        self.assertGreater(stats["compression_ratio"], 5.0)

    def test_empty_tensor(self):
        grad = torch.empty((0,), dtype=torch.float32)
        qg = quantize_gradient(grad, nbits=4, block_size=64)
        self.assertEqual(qg.numel(), 0)


if __name__ == "__main__":
    unittest.main()
