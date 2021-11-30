from test.test_pipeline import setup_CIFAR10_pipeline
from test.utils import cross_entropy_for_onehot

from gradattack.defenses import GradPruneDefense, InstahideDefense, MixupDefense


def test_mixup():
    pipeline = setup_CIFAR10_pipeline(loss=cross_entropy_for_onehot,
                                      fast_dev_run=True)

    mixup_defense = MixupDefense(pipeline.datamodule.train_set,
                                 klam=4,
                                 upper_bound=1,
                                 lower_bound=0,
                                 device="cpu",
                                 use_csprng=False)
    mixup_defense.apply(pipeline)
    pipeline.run()


def test_instahide():
    pipeline = setup_CIFAR10_pipeline(loss=cross_entropy_for_onehot,
                                      fast_dev_run=True)

    defense = InstahideDefense(pipeline.datamodule.train_set,
                               klam=4,
                               upper_bound=1,
                               lower_bound=0,
                               device="cpu",
                               use_csprng=False)
    defense.apply(pipeline)
    pipeline.run()


def test_gradprune():
    pipeline = setup_CIFAR10_pipeline(fast_dev_run=True)

    gradprune_defense = GradPruneDefense(prune_ratio=0.9)
    gradprune_defense.apply(pipeline)
    pipeline.run()
