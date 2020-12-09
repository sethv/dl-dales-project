# Deformable DETR - CSE deep learning project
Seth Vanderwilt, Zach Wilson, Zack Barnes, Richard Park

## What is this?
* We want to try out a new object detector called Deformable DETR on the COCO dataset (or some subset)
* Once we can successfully instantiate & train a model, we will modify the internals, hyperparameters, etc. and run some experiments

## Installation
* [TODO should just have a regular copy of this repo] `git submodule update --init --recursive` because we have set up Deformable DETR as a submodule
* Just follow the [Deformable DETR instructions](https://github.com/fundamentalvision/Deformable-DETR#installation) here to set up a conda environment
* Can `pip install -r requirements.txt` if we add anything else
* [detectron2 - TBD] if the Deformable DETR training script works for us, we may just want to use that and forget about detectron2, or just pull in some of its useful functions.

## Dataset
* Whoever is able to do a full training run should download COCO 2017 train/val images and annotations, should be about 20GB total. Start by just downloading the train2017 annotations https://cocodataset.org/#download
* Use [these instructions](https://github.com/fundamentalvision/Deformable-DETR#dataset-preparation) to get the right directory structure within Deformable-DETR
* Run the `make_coco_subset.py` script to shrink the `instances_train2017.json` file down to a few images & only "person" boxes. You might have to rename to `instances_train2017_orig.json` first, wanted to keep the original dataset around but need that filename for the training script.
* Run `visualize_dataset.py` to check if the boxes are showing up correctly. Probably should improve this script!

## Training
* See https://github.com/fundamentalvision/Deformable-DETR#training
* We are using this [Weights & Biases project](https://wandb.ai/dl-project/dl-final-project) but have to edit the code to log losses, metrics, etc. there. Not sure that we want to upload all the checkpoints.
