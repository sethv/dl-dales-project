# CSE490G1 Final Project - smaller & faster version of Deformable DETR
Group: Seth Vanderwilt, Zach Wilson, Zack Barnes, Richard Park

![coco detection example](coco_detection_example.png)
Example prediction output from our model, filtered by confidence threshold of 0.25

## Video
https://github.com/sethv/dl-dales-project/blob/main/project-video.mp4

## Demo/try it out
[TODO] working on producing a video like Joe's [YOLOv3](https://www.youtube.com/watch?v=MPU2HistivI)!

See Weights & Biases report link below for some visualized predictions!

[Example Colab notebook for training](https://colab.research.google.com/drive/1pVafgPzaRkVP_nuXT-Oub90Gy5xTn9Rf?usp=sharing)

## GitHub Repository
https://github.com/sethv/dl-dales-project

## Visualizations, Losses and Precision-Recall Plots
_Note: due to Colab crashes we often had to resume runs, so the time series are messed up_

https://wandb.ai/dl-project/dl-final-project/reports/Metrics-and-losses-for-our-models--VmlldzozNzI5Mzk?accessToken=qmag1px2080flylj23ha71ifed05bgqwdd22s4ys2pz33z76nyvsgnf485195vkk

## Problem Description
Our project looks at implementing an end-to-end object detector using a modified
version of the new
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
project by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai,
which in turn is a modification of Facebook AI's [DETR](https://github.com/facebookresearch/detr/)
(**DE**tection **TR**ansformer) project.

DETR (Detection Transformer) [preprint here](https://arxiv.org/abs/2010.04159) implements end-to-end object detection by feeding
feature maps produced by a standard Resnet-50 CNN into several transformer
encoder-decoder layers to predict bounding boxes and class labels directly.
Most detectors use one or many complicated components like non-maximum suppression
and anchor generation, so DETR is a much simpler approach.
However, DETR is very slow to converge and does poorly on small objects.
Deformable DETR modifies the original DETR model by using a sampling-based attention mechanism
which is described in detail in their preprint.
Zhu et al. use the same basic architecture of feeding one or more CNN feature maps
to a stack of transformer encoders and decoders, but they change the attention modules
to only attend to a small sample of points around the reference, rather than every
pixel in the feature map. This reduces the complexity and allows the model
to converge in 90% less time (see Figure 2](https://raw.githubusercontent.com/fundamentalvision/Deformable-DETR/main/figs/convergence.png of their paper)

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
2017 detection dataset (118287 train/5000 validation images, over 300000 individual boxes),
though we didn't use the full dataset for all of our experiments.
This is the dataset that most detection researchers (including Zhu et al.) use,
and it's well supported by [TorchVision](https://pytorch.org/docs/stable/torchvision/index.html)
and other supporting libraries.

Our reduced subset (results below) consisted of 5 classes (zebra, airplane, train, bear, giraffe)
and we eliminated all annotations with an `'area'` field of < 100 pixels.
This resulted in 11746 train/479 validation images, which was much easier to work with.

## Techniques
Deformable-DETR uses the multi-scale feature maps output by intermediate layers
of a standard Resnet-50. The network gradually increases the number of channels
and progressively downscales the feature maps, so there may be important
spatial information that we can use from earlier layers with larger feature maps.

We modify `Deformable-DETR/models/backbone.py` to support the
[MobileNetV2](https://arxiv.org/abs/1801.04381) backbone, which is often used in
low-resource or low-latency settings (e.g. phones or other deviecs).
In order to make this work, we perform network surgery by extracting intermediate
layers of MobileNetV2, with 32 channels x H x W at "layer2", 96 channels x H/2 x W/2
at "layer3", and 320 channels x H/4 x W/4 at what we call "layer4"
(the last feature map before the classifier head, which we of course remove).
This is to mimic what Zhu et al. do with their Resnet-50 experiments,
where they use 512 channels at "layer2", 1024 channels at "layer3", and
2048 channels at "layer4" (these are the Resnet layer names).

We used the standard `torchvision` implementation of [MobileNetV2](https://pytorch.org/docs/stable/_modules/torchvision/models/mobilenet.html#mobilenet_v2)
initialized with pretrained weights (from ImageNet).

We follow Zhu et al. and set the learning rate for the backbone layers to 0.1x the
learning rate of the rest of the network. We also were going to freeze the backbone
completely to see if the transformers could take in fixed feature maps that are optimized
for ImageNet classification and learn to do the full detection task based on these inputs,
but we forgot to run these experiments...

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

Command to reproduce (with our subset of annotations in place, see the Colab notebook):
```
!python -u main.py \
--output_dir output \
--backbone mobilenet_v2 \
--wb_name "[subset COCO] mobilenet_v2 3_enc_3_dec_dim64" \
--wb_notes "Default parameters & box refine but mobilenet_v2 backbone, reduce # of encoder decoder layers and their hidden dim" \
--enc_layers 3 \
--dec_layers 3 \
--hidden_dim 128 \
--with_box_refine \
--batch_size 4
```

### Full COCO dataset

Our best model on the full COCO dataset is still training, but is using a
MobileNetV2 backbone with 2 encoder layers and 3 decoder transformers all with dimension 64.
We enable the box refinement option as it appears to give a free boost in accuracy without
a notable decrease in speed. This results in 2940399 trainable parameters.

Here are the metrics after 17 epochs at a learning rate of 2e-4 - each epoch takes
approximately ~90 minutes on a Colab NVIDIA Tesla V100 GPU.
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.266
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.279
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.447
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.266
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.698
```

#### Post-project experiments
Fine-tuning from that checkpoint for 7 more epochs at a lower learning rate of 2e-5
for all trainable layers followed by 5 more epochs (bringing total to 29)
with the backbone learning rate
further lowered to 2e-6 further boosts the COCO metrics as suggested by Figure 2.
![figure 2](https://raw.githubusercontent.com/fundamentalvision/Deformable-DETR/main/figs/convergence.png)

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.298
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.463
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.158
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.318
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.737
```

Command to reproduce:
```
!python -u main.py \
--output_dir output \
--backbone mobilenet_v2 \
--wb_name "[full COCO] mobilenet_v2 2_enc_3_dec_dim64" \
--wb_notes "(Resume with same LR) Default parameters & box refine but mobilenet_v2 backbone, reduce # of encoder decoder layers and their hidden dim" \
--enc_layers 2 \
--dec_layers 3 \
--hidden_dim 64 \
--with_box_refine \
--batch_size 8
```

Running `benchmark.py` with a batch size of 1:
```Inference Speed: 36.5 FPS```

And with a batch size of 32:
```Inference Speed: 101.1 FPS```

## Ideas for further experiments
* More search of the transformer hyperparameters
(dimension, number of layers, number of sampling points)
* Try two-stage version of Deformable-DETR
* Disable regularization (dropout in transformer layers, weight decay)
* **Replace MobileNetV2 with newer backbones** that offer better speed/accuracy
e.g. [MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf),
[MobileDets](https://arxiv.org/pdf/2004.14525.pdf) (especially for phones?),
and [EfficientNet](https://arxiv.org/abs/1905.11946)
to compare to [EfficientDet](https://arxiv.org/abs/1911.09070)
* Measure performance on embedded devices, e.g. Nvidia Jetson (can't try on phone,
Deformable DETR has CUDA dependency).
* Train a segmentation head (instance or panoptic)
* ??? - suggestions?
