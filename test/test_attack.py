from test.test_pipeline import setup_CIFAR10_pipeline
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from gradattack.attacks.gradientinversion import GradientReconstructor


def test_gradient_attack_CIFAR10(fast_dev_run=True):

    cifar10_mean = torch.tensor(
        [0.4914672374725342, 0.4822617471218109, 0.4467701315879822])
    cifar10_std = torch.tensor(
        [0.24703224003314972, 0.24348513782024384, 0.26158785820007324])
    dm = cifar10_mean[:, None, None]
    ds = cifar10_std[:, None, None]

    pipeline = setup_CIFAR10_pipeline()
    pipeline.datamodule.setup("attack")

    example_batch = pipeline.get_datamodule_batch()

    batch_gradients, step_results = pipeline.model.get_batch_gradients(
        example_batch, 0)

    batch_inputs_transform, batch_targets_transform = step_results[
        "transformed_batch"]

    attack_instance = GradientReconstructor(
        pipeline,
        ground_truth_inputs=batch_inputs_transform,
        ground_truth_gradients=batch_gradients,
        ground_truth_labels=batch_targets_transform,
        mean_std=(dm, ds),
    )

    tb_logger = TensorBoardLogger("tb_logs", name="gradient_attack_test")
    attack_trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=10000,
        benchmark=True,
        checkpoint_callback=False,
        fast_dev_run=fast_dev_run,
    )

    attack_trainer.fit(attack_instance)
