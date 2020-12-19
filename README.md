# Deformable DETR - CSE deep learning project
Seth Vanderwilt, Zach Wilson, Zack Barnes, Richard Park

## `ProjectSite.md` has more information than this stale README - please go there instead

Paper here: https://arxiv.org/abs/2010.04159

## What is this?
* We want to try out a new object detector called Deformable DETR on the COCO dataset (or some subset)
* Once we can successfully instantiate & train a model, we will modify the internals, hyperparameters, etc. and run some experiments

## Installation
* We have Deformable-DETR in here as a regular folder because we're making modifications
* Just follow the [Deformable DETR instructions](https://github.com/fundamentalvision/Deformable-DETR#installation) here to set up a conda environment
* Can `pip install -r requirements.txt` if we add anything else
* [detectron2 - TBD] if the Deformable DETR training script works for us, we may just want to use that and forget about detectron2, or just pull in some of its useful functions.

## Dataset
* Whoever is able to do a full training run should download COCO 2017 train/val images and annotations, should be about 20GB total. Start by just downloading the train2017 annotations https://cocodataset.org/#download
* Use [these instructions](https://github.com/fundamentalvision/Deformable-DETR#dataset-preparation) to get the right directory structure within `Deformable-DETR/data/coco`
* Run the `make_coco_subset.py` script to shrink the `instances_train2017.json` file down to a few images & only "person" boxes. You might have to rename to `instances_train2017_orig.json` first, wanted to keep the original dataset around but need that filename for the training script.
* Run `visualize_dataset.py` to check if the boxes are showing up correctly. Probably should improve this script!

## Training
* See https://github.com/fundamentalvision/Deformable-DETR#training, but we are just running main.py directly e.g.
```
python -u main.py --wb_name "batchsize20_resnet18_3encs_3decs" \
--wb_notes "Try batch size of 20 for frozen resnet18 with 3 layers of transformer encoders + 3 layers of decoders" \
--lr_backbone 0 --backbone resnet18 --enc_layers 3 --dec_layers 3 \
--batch_size 20 \
--num_feature_levels 1 --output_dir output_r18
```
or something like that
* We are using this [Weights & Biases project](https://wandb.ai/dl-project/dl-final-project) and have edited the code to log losses, metrics, etc. there. Not sure that we want to upload all the checkpoints, there may be some COCO evaluation detection boxes kind of stuff for precision + recall in `eval` folder
