# CSE490G1 Final Project
Group: Seth Vanderwilt, Zach Wilson, Zack Barnes, Richard Park

## Problem Description
Our project looks at implementing an end-to-end object detector using a modified
version of Facebook's DETR project. The project that we worked on is based on
the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
project by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai.
Deformable DETR modifies the original DETR model by using a sampling-based
attention mechanism.

## Project Setup
Our project uses [PyTorch](https://pytorch.org/) as the machine learning
framework and uses [Weights & Biases](https://wandb.ai/) to track training
progress. For training, we used Google Colab to run our training script.

The basis for our project is the Deformable DETR project, so our codebase can
be set up using the [installation instructions](https://github.com/fundamentalvision/Deformable-DETR#installation)
from the Deformable DETR repo.

## Dataset
The dataset that we used to train the model was the [COCO](https://cocodataset.org/)
2017 dataset, though it's worth noting that we didn't always use the full
dataset for all of our experiments. This was the dataset that the Deformable
DETR project used, which is why we used it for our project.

## Techniques
* What techniques did we apply to solve the problem?

## Pre-Existing Work
Our repository contains the Deformable DETR repository, which we used to get
started with our project. We then modified the Deformable DETR code to start
work on our project and try out different experiments. The original Deformable
DETR code still provides the foundation that our project stands on, even with
our changes.

## New Components
We added new code to send results to Weights & Biases so we could track how
our experiments performed, which required changes to the main training script
from the Deformable DETR codebase. We also found that we needed to adjust the
print frequency for the training scripts as the excessive printing was causing
Colab to crash and become unusable.

## Experiments
* What changes did we try in order to solve the problem?
* How did we modify our dataset to facilitate the changes?

## Video
* 2-3 minute video explaining the work

## GitHub Repository
https://github.com/sethv/dl-dales-project
