# Example bashes for attack evalution

Here we provide example scripts for evaluating the attack results.

## Evalute a non encoding-based pipeline
```
python examples/calc_metric.py --dir PATH_TO_ATTACK_RESULTS 
```
where `PATH_TO_ATTACK_RESULTS` is the path for dumped attack results, e.g. `results/CIFAR10-16-/tv\=0.1BN_exact-bn\=0.005-dataseed\=None/Epoch_0/`


## Evalute an encoding-based pipeline (i.e. which uses InstaHide or MixUp defense)
```
python examples/calc_metric.py --decode_defense --dir PATH_TO_ATTACK_RESULTS 
```
where `PATH_TO_ATTACK_RESULTS` is the path for decoded results (see the second step [here](attack_cifar10_bashes.md#attack-with-instahide-being-the-defense)), e.g. `decode/CIFAR10-16-InstaHideDefense-k\{4\}-c_1\{0.0\}-c_2\{0.65\}/tv\=0.01BN_exact-bn\=0.001/`