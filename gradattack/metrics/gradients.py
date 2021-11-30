from typing import Union

import torch


class GradientMetric:
    def __init__(self, indices: Union[str, list] = "all") -> None:
        self.indices = indices

    def determine_indices(self, input_gradient):
        indices = self.indices
        if isinstance(indices, list):
            pass
        elif indices == "all":
            indices = torch.arange(len(input_gradient))
        elif indices == "batch":
            indices = torch.randperm(len(input_gradient))[:8]
        elif indices == "topk-1":
            _, indices = torch.topk(
                torch.stack([p.norm() for p in input_gradient], dim=0), 4)
        elif indices == "top10":
            _, indices = torch.topk(
                torch.stack([p.norm() for p in input_gradient], dim=0), 10)
        elif indices == "top50":
            _, indices = torch.topk(
                torch.stack([p.norm() for p in input_gradient], dim=0), 50)
        elif indices in ["first", "first4"]:
            indices = torch.arange(0, 4)
        elif indices == "first5":
            indices = torch.arange(0, 5)
        elif indices == "first10":
            indices = torch.arange(0, 10)
        elif indices == "first50":
            indices = torch.arange(0, 50)
        elif indices == "last5":
            indices = torch.arange(len(input_gradient))[-5:]
        elif indices == "last10":
            indices = torch.arange(len(input_gradient))[-10:]
        elif indices == "last50":
            indices = torch.arange(len(input_gradient))[-50:]
        else:
            raise ValueError()
        return indices


class CosineSimilarity(GradientMetric):
    def __init__(self, indices: Union[str, list] = "all") -> None:
        super().__init__(indices=indices)

    def __call__(self, trial_gradients, input_gradients):
        with torch.no_grad():
            indices = self.determine_indices(input_gradients)
            filtered_trial_gradients = [trial_gradients[i] for i in indices]
            filtered_input_gradients = [input_gradients[i] for i in indices]

        costs = sum((x * y).sum() for x, y in zip(filtered_input_gradients,
                                                  filtered_trial_gradients))

        trial_norm = sum(x.pow(2).sum()
                         for x in filtered_trial_gradients).sqrt()
        input_norm = sum(y.pow(2).sum()
                         for y in filtered_input_gradients).sqrt()
        costs = 1 - (costs / trial_norm / input_norm)
        return costs


class L2Diff(GradientMetric):
    def __init__(self, indices: Union[str, list] = "all") -> None:
        super().__init__(indices=indices)

    def __call__(self, trial_gradients, input_gradients):
        with torch.no_grad():
            indices = self.determine_indices(input_gradients)
            filtered_trial_gradients = [trial_gradients[i] for i in indices]
            filtered_input_gradients = [input_gradients[i] for i in indices]

        costs = sum((x - y).pow(2).sum() for x, y in zip(
            filtered_input_gradients, filtered_trial_gradients))

        return costs
