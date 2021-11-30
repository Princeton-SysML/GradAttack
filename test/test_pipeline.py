from test.train_config import vanilla_hparams
from test.utils import augment, early_stop_callback

from torch.nn import CrossEntropyLoss

import pytorch_lightning as pl
from gradattack.datamodules import CIFAR10DataModule, ImageNetDataModule
from gradattack.models import create_lightning_module
from gradattack.trainingpipeline import TrainingPipeline

devices = [0]

# def setup_ImageNet_pipeline(
#     augment=augment,
#     batch_size=128,
#     tune_on_val=0.02,
#     loss=CrossEntropyLoss(reduction="mean"),
#     n_accumulation_steps=1,
#     freeze_extractor=False,
#     ckpt=None,
#     model="models.densenet121",
#     stage="train",
#     fast_dev_run=True,
#     max_epochs=200,
#     logger=None,
# ):
#     imagenet_datamodule = ImageNetDataModule(
#         augment=augment,
#         data_dir=
#         "/data/Hazel/Research_2021/GradAttack/data/imagenet",  # TODO: Please replace this with your own path to imagenet data
#         batch_size=batch_size,
#         tune_on_val=tune_on_val,
#     )
#     imagenet_datamodule.setup(stage=stage)
#     model = create_lightning_module(
#         model,
#         imagenet_datamodule.num_classes,
#         training_loss_metric=loss,
#         freeze_extractor=freeze_extractor,
#         ckpt=ckpt,
#     )

#     trainer = pl.Trainer(
#         gpus=devices,
#         check_val_every_n_epoch=1,
#         fast_dev_run=fast_dev_run,
#         logger=logger,
#         max_epochs=max_epochs,
#         callbacks=[early_stop_callback],
#         accumulate_grad_batches=n_accumulation_steps,
#     )

#     pipeline = TrainingPipeline(model, imagenet_datamodule, trainer)
#     return pipeline


def setup_CIFAR10_pipeline(
    augment=augment,
    batch_size=128,
    tune_on_val=0.02,
    loss=CrossEntropyLoss(reduction="mean"),
    n_accumulation_steps=1,
    freeze_extractor=False,
    ckpt=None,
    model="ResNet18",
    stage="train",
    hparams=vanilla_hparams,
    fast_dev_run=True,
    max_epochs=200,
    logger=None,
):

    cifar10_datamodule = CIFAR10DataModule(
        augment=augment,
        batch_size=batch_size,
        tune_on_val=tune_on_val,
        batch_sampler=None,
    )
    cifar10_datamodule.setup(stage=stage)
    model = create_lightning_module(
        model,
        cifar10_datamodule.num_classes,
        training_loss_metric=loss,
        freeze_extractor=freeze_extractor,
        ckpt=ckpt,
        **hparams,
    )
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        fast_dev_run=fast_dev_run,
        logger=logger,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback],
        accumulate_grad_batches=n_accumulation_steps,
    )

    pipeline = TrainingPipeline(model, cifar10_datamodule, trainer)
    return pipeline


def test_pipeline_creation():
    pipeline_cifar = setup_CIFAR10_pipeline()
    pipeline_cifar.run()
    # pipeline_imagenet = setup_ImageNet_pipeline()
    # pipeline_imagenet.run()
