
# Gradient Inversion Attacks and Defenses

Here we provide a (growing) list of research papers for gradient inversion attacks and defenses. Please feel free to submit an issue to report any new or missing papers.

## Papers for attacks
Recent research shows that sending gradients instead of data in Federated Learning can leak private information. These attacks demonstrate that an adversary eavesdropping on a client’s communications (i.e. observing the global modelweights and client update) can accurately reconstruct a client’s private data using a class of techniques known as “gradient inversion attacks", which raise serious concerns about such privacy leakage.


| Attack name | Paper 	| Venue        	| Additional Information Other Than Gradients| Supported 	| Official implementation 	|
|------- |-------	|--------------	|--------------	|----------- |----------- |
| DLG |  [Deep leakage from gradients](https://arxiv.org/pdf/1906.08935.pdf) 	| NeurIPS 2019 	| No | Yes       	| [link](https://github.com/mit-han-lab/dlg) | 
| iDLG | [iDLG: Improved Deep Leakage from Gradients](https://arxiv.org/pdf/2001.02610.pdf) | Arxiv   	| No | Yes       	| [link](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients)  |  
| Inverting Gradients | [Inverting Gradients -- How easy is it to break privacy in federated learning?](https://arxiv.org/pdf/2003.14053.pdf) | NeurIPS 2020   	| Batch Normalization statistics & private labels | Yes       	| [link](https://github.com/JonasGeiping/invertinggradients) |
| R-GAP | [R-GAP: Recursive Gradient Attack on Privacy](https://arxiv.org/pdf/2010.07733)  | ICLR 2021   	| The rank of the rank of the coefficient matrix (see Section 3.1.2 of its paper) | No (a relatively weak attack)       	| [link](https://github.com/JunyiZhu-AI/R-GAP) |
| GradInversion| [See through Gradients: Image Batch Recovery via GradInversion](https://arxiv.org/pdf/2104.07586) | CVPR 2021   	| Batch Normalization statistics & Good approximation of private labels | No (code unavailable)       	| No |
|  GIAS | [Gradient Inversion with Generative Image Prior](https://arxiv.org/pdf/2110.14962.pdf) | NeurIPS 2021   |  A GAN trained on the dirstribution of training data	| No (on our plan)       	| [link](https://github.com/ml-postech/gradient-inversion-generative-image-prior) | 
| CAFE | [CAFE: Catastrophic Data Leakage in Vertical Federated Learning](https://arxiv.org/pdf/2110.15122.pdf) | NeurIPS 2021   | Batch indicies	| No (on our plan)       	| [link](https://github.com/derafael/cafe) |

## Papers for defenses 
To counter these attacks, researchers have proposed defense mechanisms, including:
- **encrypting gradient** such assecure aggregation protocol or homomorphic encryption, which are secure, but require special setups and may introduce substantial overheads,
- **perturbing gradient** by adding deferentially private noise or gradient pruning which requires finding tradeoffs between accuracy and privacy leakage,
- **encoding the input** such as InstaHide by encoding input data to the model, which also requires finding tradeoffs between accuracy and privacy leakage.

### Defenses for  plain-text gradients
| Defense name | Paper 	| Venue        	| Supported 	| Official implementation 	|
|-------	|--------------	|-----------	|----------- |----------- | 
| DPSGD | [Deep Learning with Differential Privacy](https://arxiv.org/pdf/1607.00133.pdf) | CCS 2016   	| Yes       	| [link](https://github.com/pytorch/opacus) |
| Gradient Pruning | [Deep leakage from gradients](https://arxiv.org/pdf/1906.08935.pdf) 	| NeurIPS 2019 	| Yes       	| [link](https://github.com/mit-han-lab/dlg) |
| MixUp | [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf) | ICLR 2018   	| Yes       	| [link](https://github.com/facebookresearch/mixup-cifar10) |
| InstaHide | [InstaHide: Instance-hiding Schemes for Private Distributed Learning](https://arxiv.org/pdf/2010.02772.pdf) | ICML 2020   	| Yes       	| [link](https://github.com/Hazelsuko07/InstaHide) | 


### Defenses that encrypt gradients

| Defense name | Paper 	| Venue        	| Official implementation 	|
|-------	|--------------	|-----------	|----------- |
| Secure Aggregation | [Practical Secure Aggregation for Federated Learning on User-Held Data](https://arxiv.org/pdf/1611.04482.pdf)  | NeurIPS 2016  | No       	|
| FastSecAgg | [FastSecAgg: Scalable Secure Aggregation for Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2009.11248.pdf) | CCS 2020  | No       	|
| LightSecAgg | [LightSecAgg: Rethinking Secure Aggregation in Federated Learning](https://arxiv.org/pdf/2109.14236.pdf) | Arxiv  | No       	|
| Homomorphic Encryption | [Privacy-Preserving Deep Learning via Additively Homomorphic Encryption](https://eprint.iacr.org/2017/715.pdf) | ATIS 2017  | No       	|
