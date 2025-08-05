# CIFAR-10 part: Predicting memorization within Large Language Models fine-tuned for classification

Jérémie Dentan<sup>1</sup>, Davide Buscaldi<sup>1, 2</sup>, Aymen Shabou<sup>3</sup>, Sonia Vanier<sup>1</sup>

<sup>1</sup>LIX (École Polytechnique, IP Paris, CNRS) <sup>2</sup>LIPN (Sorbonne Paris Nord) <sup>3</sup>Crédit Agricole SA

This repository contains the source code for reproducing the CIFAR-10 part of the experiments for our paper "Predicting memorization within Large Language Models fine-tuned for classification", published at ECAI 2025. The main repository can be found at: [https://github.com/orailix/predict_llm_memorization](https://github.com/orailix/predict_llm_memorization).

## Acknowledgement

This repository is a fork of [misleading-privacy-evals](https://github.com/ethz-spylab/misleading-privacy-evals) from the SPY Lab at ETH Zurich. The original repository contains the source code for the paper [*Evaluations of Machine Learning Privacy Defenses are Misleading*](https://arxiv.org/abs/2404.17399), authored by [Michael Aerni](https://www.michaelaerni.com/), [Jie Zhang](https://zj-jayzhang.github.io/), and [Florian Tramèr](https://floriantramer.com/), and published at ACM CCS 2024.

This work received financial support from Crédit Agricole SA through the research chair “Trustworthy and responsible AI” with École Polytechnique. This work was performed using HPC resources from GENCI-IDRIS 2023-AD011014843. We thank Arnaud Grivet Sébert and Mohamed Dhouib for discussions on this paper.

## Copyright and License

At the time of publishing this source code, the repository [misleading-privacy-evals](https://github.com/ethz-spylab/misleading-privacy-evals) is unlicensed. Our work is Copyright 2023–present, Laboratoire d'Informatique de l'École Polytechnique, and is released under the Apache License v2.0.

Please cite this work as follows:

```bibtex
@inproceedings{dentan_predicting_2025,
  title = {Predicting Memorization within Large Language Models Fine-Tuned for Classification},
  author = {Dentan, Jérémie and Buscaldi, Davide and Shabou, Aymen and Vanier, Sonia},
  booktitle = {Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025)},
  year = {2025},
  note = {To appear},
  url = {https://arxiv.org/abs/2409.18858}
}
```

## Reproducing our results

### Setup environment

Our experiments were run under Python 3.12.8, using the requirements in `requirements.txt`

### Generate data

We provide scripts `scripts/train.sh` and `scripts/attack.sh` which enable the deployment of the experiments on a SLURM HPC cluster. You should first lauch the training of the models, and then the attack.

### Reproduce figures

After the training and the attack, you can use Bash script `scripts/hpc_sync.sh` to download the results from the HPC to you local machine, which is more convenient to reproduce our figures.

Then, we provide `figures/main_figures.ipynb` to analyze the data and reproduce our figures.
