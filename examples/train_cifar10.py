import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from gradattack.datamodules import CIFAR10DataModule
from gradattack.defenses.defense_utils import DefensePack
from gradattack.models import create_lightning_module
from gradattack.trainingpipeline import TrainingPipeline
from gradattack.utils import cross_entropy_for_onehot, parse_args, parse_augmentation

if __name__ == "__main__":

    args, hparams, _ = parse_args()

    logger = TensorBoardLogger(
        "tb_logs", name=f"{args.logname}/{args.optimizer}/{args.scheduler}")
    devices = [args.gpuid]

    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="epoch/val_loss",
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode="min",
        )

    augment = parse_augmentation(args)

    assert args.data == "CIFAR10"

    datamodule = CIFAR10DataModule(
        augment=augment,
        batch_size=args.batch_size,
        tune_on_val=args.tune_on_val,
        batch_sampler=None,
    )

    if args.defense_instahide or args.defense_mixup:
        loss = cross_entropy_for_onehot
    else:
        loss = torch.nn.CrossEntropyLoss(reduction="mean")

    if "multihead" in args.model:
        multi_head = True
        loss = torch.nn.CrossEntropyLoss(reduction="mean")
    else:
        multi_head = False

    model = create_lightning_module(
        args.model,
        datamodule.num_classes,
        pretrained=args.pretrained,
        ckpt=args.ckpt,
        freeze_extractor=args.freeze_extractor,
        training_loss_metric=loss,
        log_auc=args.log_auc,
        multi_class=datamodule.multi_class,
        multi_head=multi_head,
        **hparams,
    )

    trainer = pl.Trainer(
        gpus=devices,
        check_val_every_n_epoch=1,
        logger=logger,
        max_epochs=args.n_epoch,
        callbacks=[early_stop_callback],
        accumulate_grad_batches=args.n_accumulation_steps,
    )
    pipeline = TrainingPipeline(model, datamodule, trainer)

    defense_pack = DefensePack(args, logger)
    defense_pack.apply_defense(pipeline)

    pipeline.run()
    pipeline.test()