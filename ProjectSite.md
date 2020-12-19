# CSE490G1 Final Project
Group: Seth Vanderwilt, Zach Wilson, Zack Barnes, Richard Park

## Video
https://github.com/sethv/dl-dales-project/blob/main/project-video.mp4

## Demo/try it out
[TODO] working on producing a video like Joe's [YOLOv3](https://www.youtube.com/watch?v=MPU2HistivI)

[Example Colab notebook for training](https://colab.research.google.com/drive/1pVafgPzaRkVP_nuXT-Oub90Gy5xTn9Rf?usp=sharing)

## GitHub Repository
https://github.com/sethv/dl-dales-project

## Loss and Precision-Recall Plots
_Note: due to Colab crashes we often had to resume runs, so the time series may look strange_
https://wandb.ai/dl-project/dl-final-project/reports/Metrics-and-losses-for-our-models--VmlldzozNzI5Mzk?accessToken=qmag1px2080flylj23ha71ifed05bgqwdd22s4ys2pz33z76nyvsgnf485195vkk

## Problem Description
Our project looks at implementing an end-to-end object detector using a modified
version of Facebook AI's [DETR](https://github.com/facebookresearch/detr/)
(**DE**tection **TR**ansformer) project.
The project that we worked on is a modification of the new
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
work by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai.

DETR (Detection Transformer) implements end-to-end object detection by feeding
feature maps produced by a standard Resnet-50 CNN into several transformer
encoder-decoder layers to predict bounding boxes and class labels directly.
Most detectors use one or many complicated components like non-maximum suppression
and anchor generation, so DETR is a much simpler approach.
However, DETR is very slow to converge and does poorly on small objects.
Deformable DETR modifies the original DETR model 
by using a sampling-based attention mechanism described in their preprint, which
you can find [here](https://arxiv.org/abs/2010.04159)
Zhu et al. use the same basic architecture of feeding one or more CNN feature maps
to a stack of transformer encoders and decoders, but they change the attention modules
to only attend to a small sample of points around the reference, rather than every
pixel in the feature map. This reduces the complexity and allows the model
to converge in 90% less time (see Figure 2 of their paper)

We created a smaller version of Deformable-DETR with a smaller backbone and 
fewer encoder-decoder layers, so that trains faster on our limited GPU resources!

## Project Setup
Our project uses [PyTorch](https://pytorch.org/) as the machine learning
framework and uses [Weights & Biases](https://wandb.ai/) to track training
progress. For training, we used Google Colab to run our training scripts.

The basis for our project is the Deformable DETR project, so our codebase can
be set up using the [installation instructions](https://github.com/fundamentalvision/Deformable-DETR#installation)
from the Deformable DETR repo, with our revised `Deformable-DETR/requirements.txt` file.

## Dataset
The dataset that we used to train the model was the [COCO](https://cocodataset.org/)
2017 dataset (118287 train/5000 val images), though it's worth noting that
we didn't always use the full dataset for all of our experiments.
This was the dataset that most detection researchers (including Zhu et al.) use,
and it's well supported by [TorchVision](https://pytorch.org/docs/stable/torchvision/index.html)
and other supporting libraries.

Our reduced subset consisted of 5 classes (zebra, airplane, train, bear, giraffe)
and we eliminated all annotations with 'area' < 100 pixels.
This resulted in 11746 train/479 val images.

## Techniques
Deformable-DETR uses the multi-scale feature maps output by intermediate layers
of a standard Resnet-50. The network gradually increases the number of channels
and progressively downscales the feature maps, so there may be important
spatial information that we can use from earlier layers with larger feature maps.

We modify `Deformable-DETR/models/backbone.py` to support the
[MobileNetV2](https://arxiv.org/abs/1801.04381) backbone, which is often used in
low-resource or low-latency settings (e.g. phones or other deviecs).
In order to make this work, we perform network surgery by extracting intermediate
layers of MobileNetV2, with 32 channels at "layer2", 96 channels at "layer3", and
320 channels at what we call "layer4" (the last feature map before the classifier
head, which we remove).

We used the standard `torchvision` implementation of [MobileNetV2](https://pytorch.org/docs/stable/_modules/torchvision/models/mobilenet.html#mobilenet_v2)
initialized with pretrained weights (from ImageNet).

## Pre-Existing Work
Our repository contains the Deformable DETR repository, which we used to get
started with our project. We then modified the Deformable DETR code to start
work on our project and try out different experiments. The original Deformable
DETR code still provides the foundation that our project stands on, even with
our changes. We only modified the training script and arguments specifically
for our experiment, and left most of the hyperparameters etc. untouched.

**Warning: we likely have introduced new bugs, especially since we did not
consider any distributed training.**

## New Components
We added support in the Deformable DETR backbone code to load the standard
pretrained torchvision MobileNetV2 model and extract intermediate feature maps.

We added new code to send results to Weights & Biases so we could track how
our experiments performed, which required changes to the main training script
from the Deformable DETR codebase. We also found that we needed to adjust the
print frequency for the training scripts as the excessive printing was causing
Colab to crash and become unusable, among other small changes.

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
default training schedule of 40 epochs at lr=2e-4 followed by 10 at lr=2e-5. Each epoch
takes approximately 12-13 minutes on a Colab NVIDIA Tesla V100 GPU.

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

This made it easy to disambiguate which dataset we were using, because to our knowledge
nobody produces such high scores on the full COCO `val2017` dataset!

### Full COCO dataset

Our best model on the full COCO dataset is still training, but is using a
MobileNetV2 backbone with 2 encoder layers and 3 decoder transformers all with dimension 64.
We enable the box refinement option as it appears to give a free boost in accuracy without
a notable decrease in speed. This results in 2940399 trainable parameters.

Here are the metrics after 13 epochs at a learning rate of 2e-4 - each epoch takes
approximately ~90 minutes on a Colab NVIDIA Tesla V100 GPU.
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.241
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.126
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.260
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.676 
```

We hope the run will survive on Colab and continue to improve!
