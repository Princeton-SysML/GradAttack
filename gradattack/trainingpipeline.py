import pytorch_lightning as pl

from gradattack.models import LightningWrapper


class TrainingPipeline:
    def __init__(
        self,
        model: LightningWrapper,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
    ):
        self.model = model
        self.datamodule = datamodule
        self.trainer = trainer
        self.applied_defenses = []
        self._data_transformations = []  # Modifications to the dataloader
        self._model_transformations = (
            []
        )  # Modifications to the model architecture, trainable params ...
        self.datamodule.setup()

    # FIXME: @Samyak, are we actually using this funciton?
    def log_hparams(self):
        self.trainer.logger.log_hyperparams(self.model.hparams)

    def setup_pipeline(self):
        self.datamodule.prepare_data()

        for transform in self._data_transformations:
            transform(self.datamodule)
        for transform in self._model_transformations:
            transform(self.model)

        if len(self._data_transformations) > 0:
            assert self.datamodule.batch_sampler != None

    def run(self):
        self.setup_pipeline()
        # If we didn't call setup(), any updates to transforms (e.g. from defenses) wouldn't be applied
        return self.trainer.fit(self.model, self.datamodule)

    def test(self):
        return self.trainer.test(
            self.model, test_dataloaders=self.datamodule.test_dataloader())

    # FIXME: @Samyak, are we actually using this funciton?
    def get_datamodule_batch(self):
        self.datamodule.setup()
        trainloader = self.datamodule.train_dataloader()
        for batch in trainloader:
            return batch
