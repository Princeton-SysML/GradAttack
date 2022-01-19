# Example bashes for attacking a CIFAR-10 training pipeline


Here we provide example scripts for attacking a CIFAR-10 training pipeline under various defenses, with or without strong assumption about the threat model.

## Vanilla attack (no defense is applied)

### The realistic setting

In a realistic Federated Learning setup, the attacker does **not** know private BatchNorm statistics or private labels of the victim batch. The script below runs the attack in such a setting. 

```
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --tv 0.05 --bn_reg 0.001 --reconstruct_labels
```

where `reconstruct_labels` asserts that attacker needs to reconstruct private labels when running the inversion attack; `tv` (coef. for total variation) and `bn_reg` (coef. for BatchNorm regularizer) are tuneable hyper-parameters. 

### The "strong assumption" setting (Geiping et al.'s)

Previous work (Geiping et al.'s) evaluted with a potentially strong assumption about the threat model, where the attacker knows the BatchNorm statistics and private labels. The script below runs the attack in such a setting. 

```
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005
```

where `BN_exact` allows the attacker to know the exact private BatchNorm statistics of the victim batch; `tv` (coef. for total variation) and `bn_reg` (coef. for BatchNorm regularizer) are tuneable hyper-parameters. 


## Run attack when defense(s) are applied

### Attack with gradient pruning being the defense

Attacking a pipeline with the gradient pruning defense is similar to attacking a vanilla pipeline.
```
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005 --defense_gradprune --p 0.5
```

where `defense_gradprune` enables the gradient pruning defense, and `p` sets the prune ratio. 

### Attack with an encoding-based defense

Attacking a pipeline with the encoding-based defense (InstaHide as an example) consists of 2 steps:

1. Recover encoded InstaHide samples from the gradient

```
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005 --defense_instahide --k 4 --c_1 0 --c_2 0.65
```

where `defense_instahide` enables the InstaHide defense, and `k`, `c_1`, `c_2` are hyper-parameters of the defense.

2. Run the decode attack ([Carlini et al.](https://arxiv.org/pdf/2011.05315.pdf)) on encoded InstaHide samples (of multiple training steps, e.g. 50)

```
python examples/attack_decode.py --dir PATH_TO_STEP1_RESULTS --instahide --k 4 --dest_dir PATH_TO_STEP2_RESULTS
```

where `PATH_TO_STEP1_RESULTS` is the path for recovered images from step 1 (e.g. `results/CIFAR10-16-InstaHideDefense-k\{4\}-c_1\{0.0\}-c_2\{0.65\}/tv\=0.01BN_exact-bn\=0.001/`), and `PATH_TO_STEP2_RESULTS` is the path for the final decoded results. 
