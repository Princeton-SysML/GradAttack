import torch

from gradattack.defenses.defense import GradientDefense
from gradattack.trainingpipeline import TrainingPipeline


class GradPruneDefense(GradientDefense):
    def __init__(self, prune_ratio: float):
        """
        Args:
            prune_ratio (float): the ratio of gradients to be pruned
        """
        super().__init__()
        self.prune_ratio = prune_ratio

    def apply(self, pipeline: TrainingPipeline):
        """Apply the gradient pruning defense

        Args:
            pipeline (TrainingPipeline): the training pipeline to protect
        """
        super().apply(pipeline)

        def do_gradprune(model):
            """

            Args:
                model (nn.Module): the original model

            Returns:
                nn.Module: the model with pruned gradients
            """
            parameters = model.parameters()

            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
                parameters = list(
                    filter(lambda p: p.grad is not None, parameters))

            input_grads = [p.grad.data for p in parameters]

            threshold = [
                torch.quantile(torch.abs(input_grads[i]), self.prune_ratio)
                for i in range(len(input_grads))
            ]

            for i, p in enumerate(model.parameters()):
                p.grad[torch.abs(p.grad) < threshold[i]] = 0
            return model

        pipeline.model._grad_transformations.append(do_gradprune)
