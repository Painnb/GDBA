# Defending Graph Neural Networks via Attribute Debiasing

## Overview

This repository contains python codes and datasets necessary to run the proposed GDBA approach. GDBA is a defense approach designed to defend GNNs that the main idea of the paper consists of refining graph edges using the innovative unbiased attribute-augmented PPR and mutual information regularization. Please refer to our paper for additional specifications.


## Requirements

Code is written in Python 3.6 and requires:
- PyTorch
- Torch Geometric
- NetworkX
- Deeprobust

## Datasets
For node classification, the used datasets are as follows:
- Cora
- CiteSeer
- ACM
- BlogCatalog
- UAI
- Flickr

All these datasets are part of the torch_geometric or deeprobust datasets and are directly downloaded when running the code.


## Training and Evaluation
To use our code, the user should first download the [DeepRobust](https://github.com/DSE-MSU/DeepRobust) package ( https://github.com/DSE-MSU/DeepRobust). Since we are using the [GNNGuard](https://github.com/mims-harvard/GNNGuard/tree/master) as a baseline, we are using the provided code from their official GitHub repository (https://github.com/mims-harvard/GNNGuard/tree/master).

As explained in the GNNGuard's original code, some files need to be substituted in the original DeepRobust implementation in the folder "deeprobust/graph/defense" by the one provided in our implementation. The file "migcnppr.py" contains our proposed framework.


To train and evaluate the model in the paper, the user should specify the following :

- Dataset : The dataset to be used

- Budget: The budget of the attack

  To run some typical experiments, you can use the following codes:

```bash
python main_mettack.py --dataset acm --ptb_rate 0.05
bash mettack.sh

python main_dice.py --dataset blogcatalog --ptb_rate 0.1
bash dice.sh

python main_random.py --dataset flickr --ptb_rate 0.2
bash random.sh
```

## Citing

If you find our proposed GDBA useful for your research, please consider citing our paper.

## License

GDBA is licensed under the MIT License.
