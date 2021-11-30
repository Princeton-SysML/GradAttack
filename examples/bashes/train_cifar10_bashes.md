# Example bashes for CIFAR-10 training

Here we provide example scripts for running a CIFAR-10 training pipeline under various defenses.

## Vanilla training
Let's start with a vanilla training example:

```
python3 examples/train_cifar10.py --scheduler ReduceLROnPlateau --tune_on_val 0.02 --lr 0.05 --lr_factor 0.5 --logname CIFAR10/Vanilla 
```
where `tune_on_val` is the ratio for the validation dataset, `lr` is the initial learning rate, `scheduler` and `lr_factor` defines how the learning rate changes during training (in the shown example, we decay the learning rate by a factor of 0.5 when the validation loss has stopped improving).

## Defend with Gradient Pruning
[Gradient pruning](https://arxiv.org/pdf/1906.08935.pdf) zeros out gradient entries with low magnitude: 

```
python3 examples/train_cifar10.py --scheduler ReduceLROnPlateau --tune_on_val 0.02 --lr 0.05 --lr_factor 0.5 --logname CIFAR10/GradPrune --defense_gradprune --p 0.9
```

where `defense_gradprune` enables the gradient pruning defense, and `p` sets the prune ratio (i.e. the fraction of gradient entries we want to prune). 

## Defend with MixUp
[MixUp](https://arxiv.org/pdf/1710.09412.pdf) generates an encoded training sample by linearly combining k randomly chosen sample using random coefficients (sum to 1):

```
python3 examples/train_cifar10.py --scheduler ReduceLROnPlateau --tune_on_val 0.02 --lr 0.1 --lr_factor 0.5 --logname CIFAR10/MixUp --defense_mixup --k 4 --c_1 0 --c_2 0.65  --patience 50
```
where `defense_mixup` enables the MixUp defense, `k` sets the number of samples to mix, `c_1` and `c_2` are lower and upper bounds for each mixing  coefficient.


## Defend with InstaHide
[InstaHide](https://arxiv.org/pdf/2010.02772.pdf) generates an encoded training sample by linearly combining k randomly chosen sample using random coefficients (sum to 1), and applying on it a random sign-flipping pattern:
```
python3 examples/train_cifar10.py --scheduler StepLR --lr_step 50 --tune_on_val 0.02 --lr 0.05 --lr_factor 0.1 --logname CIFAR10/InstaHide --defense_instahide --k 4 --c_1 0 --c_2 0.65 --patience 50
```

where `defense_instahide` enables the InstaHide defense, `k` sets the number of samples to mix, `c_1` and `c_2` are lower and upper bounds for each mixing  coefficient.

## Use Combined Defenses
Combining defenses is very simple in GradAttack:
```
# Gradient Pruning + MixUp 
python3 examples/train_cifar10.py --scheduler ReduceLROnPlateau --tune_on_val 0.02 --lr 0.1 --lr_factor 0.5 --logname CIFAR10/MixUp+GradPrune --defense_mixup --k 6 --c_1 0 --c_2 0.65 --defense_gradprune --p 0.9 --patience 50 

# Gradient Pruning + InstaHide  
python3 examples/train_cifar10.py --scheduler StepLR --lr_step 50 --tune_on_val 0.02 --lr 0.05 --lr_factor 0.1  --logname CIFAR10/InstaHide+GradPrune --defense_instahide --k 6 --c_1 0 --c_2 0.65 --defense_gradprune --p 0.9 --patience 50
```