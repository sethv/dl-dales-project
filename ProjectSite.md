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
progress. For training, we used Google Colab to run our training scripts.

The basis for our project is the Deformable DETR project, so our codebase can
be set up using the [installation instructions](https://github.com/fundamentalvision/Deformable-DETR#installation)
from the Deformable DETR repo, with our revised `Deformable-DETR/requirements.txt` file.

## Dataset
The dataset that we used to train the model was the [COCO](https://cocodataset.org/)
2017 dataset (118287 train/5000 val images), though it's worth noting that we didn't always use the full
dataset for all of our experiments. This was the dataset that the Deformable
DETR project used, which is why we used it for our project.

Our reduced subset consisted of 5 classes (zebra, airplane, train, bear, giraffe)
and we eliminated all annotations with 'area' < 100 pixels.
This resulted in 11746 train/479 val images.

## Techniques
Deformable-DETR uses the multiscale feature maps output by intermediate layers
of a standard Resnet-50. We do the same using MobileNetV2 feature maps!

## Pre-Existing Work
Our repository contains the Deformable DETR repository, which we used to get
started with our project. We then modified the Deformable DETR code to start
work on our project and try out different experiments. The original Deformable
DETR code still provides the foundation that our project stands on, even with
our changes. We only modified the training script and arguments specifically
for our experiment, and left most of the hyperparameters etc. untouched.

## New Components
We added support in the Deformable DETR backbone code to load the standard
pretrained torchvision MobileNetV2 model and extract intermediate feature maps.

We added new code to send results to Weights & Biases so we could track how
our experiments performed, which required changes to the main training script
from the Deformable DETR codebase. We also found that we needed to adjust the
print frequency for the training scripts as the excessive printing was causing
Colab to crash and become unusable.

## Experiments

### Easier COCO subset

Our best model on the reduced COCO dataset used many of the Deformable-DETR defaults
and incorporated their iterative box refinement implementation.

We changed the backbone to MobileNetV2 and shrunk the number of encoder and decoder
layers to 3 each, with a reduced hidden dimension (for the weight matrices) of 128
(their default is 256). This results in 4683631 trainable parameters, as a few of
of the early convolutional layers in MobileNetV2 remain frozen.
Due to Colab issues we trained for 50 epochs with the first 16 at learning rate of 2e-4,
the next 24 being at 2e-5, and the last 10 at 2e-6. This differs from the Deformable-DETR
default training schedule of 40 epochs at lr=2e-4 followed by 10 at lr=2e-5.

Here are the official COCO precision/recall metrics on the COCO subset described above:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.642
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.853
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.702
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.348
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.530
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.722
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.463
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.728
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.802
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.525
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.725
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.884
```

### 

## Video
https://github.com/sethv/dl-dales-project/blob/main/project-video.mp4

## GitHub Repository
https://github.com/sethv/dl-dales-project
