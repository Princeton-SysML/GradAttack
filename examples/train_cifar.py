import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from gradattack.datamodules import CIFAR100DataModule, CIFAR10DataModule
from gradattack.defenses.defense_utils import DefensePack
from gradattack.models import create_lightning_module
from gradattack.trainingpipeline import TrainingPipeline
from gradattack.utils import cross_entropy_for_onehot, parse_args, parse_augmentation

torch.set_float32_matmul_precision("high")
cifar_dm = {
    "CIFAR10": CIFAR10DataModule,
    "CIFAR100": CIFAR100DataModule
}

if __name__ == "__main__":
    args, hparams, _ = parse_args()
    method = ""
    if args.defense_mixup:
        method += 'mixup_'
    elif args.defense_instahide:
        method += 'instahide_'
    elif args.defense_gradprune:
        method += 'gradprune_'
    else:
        method += 'vanilla'
    exp_name = f"{args.data}/{method}{args.scheduler}"
    logger = WandbLogger(
        project='GradAttack',
        name=exp_name,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss_epoch",
        min_delta=0.00,
        patience=50,
        verbose=False,
        mode="min",
    )
    callback = ModelCheckpoint(
        exp_name, save_last=True, save_top_k=3, monitor="val/acc", mode="max",
    )

    augment = parse_augmentation(args)

    datamodule = cifar_dm[args.data](
        augment=augment,
        batch_size=args.batch_size,
        tune_on_val=args.tune_on_val,
        batch_sampler=None,
    )

    if args.defense_instahide or args.defense_mixup:
        loss = cross_entropy_for_onehot
    else:
        loss = torch.nn.CrossEntropyLoss()

    if "multihead" in args.model:
        multi_head = True
        loss = torch.nn.CrossEntropyLoss()
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
        log_dir=exp_name,
        **hparams,
    )

    trainer = pl.Trainer(
        default_root_dir=exp_name,
        devices=1,
        check_val_every_n_epoch=3,
        accelerator='auto',
        benchmark=True,
        logger=logger,
        num_sanity_val_steps=0,
        max_epochs=args.n_epoch,
        callbacks=[early_stop_callback, callback],
        accumulate_grad_batches=args.n_accumulation_steps,
    )
    pipeline = TrainingPipeline(model, datamodule, trainer)

    defense_pack = DefensePack(args)
    defense_pack.apply_defense(pipeline)

    pipeline.run()
    pipeline.test()
