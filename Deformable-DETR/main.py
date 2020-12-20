# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

import torchvision.transforms
import wandb
from datasets.torchvision_datasets.coco import CocoDetection
import datasets.transforms as T
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)

    # Our wandb metadata
    parser.add_argument('--no_wb', default=False, action='store_true') # include this to turn off W&B
    parser.add_argument('--wb_name', required=True, help='Weights & Biases experiment name - just 1-2 words', type=str)
    parser.add_argument('--wb_notes', required=True, help='Weights & Biases description of your experiment - length of a git commit msg', type=str)

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

@torch.no_grad()
def visualize_video(model, postprocessors):
    """
    Unfinished but manages to label selected frames from a video & log to W&B
    
    Ideally would be able to produce an output video with labels
    """

    model.eval()

    # COCO classes
    class_id_to_label = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    vid_frame_count = 0
    max_vid_frames = 10
    # Try reading the video
    # path = "../project-video.mp4"
    path = 'NEW YORK CITY 2019 - BUSY TIMES SQUARE! [4K]-WPvMlyr4H_Y.mp4'
    assert os.path.exists(path)
    video = cv2.VideoCapture(path)
    # output_video = cv2.VideoWriter
    assert video.isOpened()
    print(video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    batch_size = 1
    for frame_idx in range(10000):
        # print(frame_idx)
        ret, frame = video.read()
        if frame_idx % 30: # want to get further through video
            continue
        if not ret:
            break
        frames = [frame]

        inputs = torchvision.transforms.ToTensor()(frame).to(device)
        # inputs /= 255 # comes in as [0,255], convert to [0,1]
        # inputs = inputs.permute(0, 3, 1, 2) # from NHWC to NCHW
        # x = x.permute(2,0,1) # from HWC to CHW
        inputs = torchvision.transforms.Resize((height//4, width//4))(inputs)
        # TODO normalize!
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        inputs = normalize(inputs)
        
        outputs = model([inputs])
        orig_target_sizes = torch.stack([torch.tensor([height,width])], dim=0).to(device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        result = results[0]

        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes'].cpu()

        box_data = []

        # each box
        for j in range(len(labels)):
            score = scores[j].item()
            label = labels[j].item()
            box = boxes[j]

            if score < 0.1 :
                continue

            box_data_j = {
                "position": {
                    "minX": box[0].item(),
                    "maxX": box[2].item(),
                    "minY": box[1].item(),
                    "maxY": box[3].item()
                },
                "class_id": label,
                "box_caption": "%s (%.3f)" % (class_id_to_label.get(label, f"unknown-label-{label}"), score),
                "domain": "pixel",
                "scores": {"score": score}
            }


            box_data.append(box_data_j)

        boxes = {
            "predictions": {
                "box_data": box_data,
                "class_labels": class_id_to_label
            },
            #"ground_truth": {} # would be nice to have
        }
        # print("boxes:", boxes)

        if not args.no_wb:
            img = wandb.Image(frame, boxes=boxes)
            wandb.log({f"video{vid_frame_count}": img})
        else:
            from google.colab.patches import cv2_imshow
            cv2_imshow(frame) # draw bboxes
        
        vid_frame_count += batch_size

        if vid_frame_count >= max_vid_frames:
            return

@torch.no_grad()
def visualize_bbox(model: torch.nn.Module, postprocessors, dataloader, device, dataset_val_without_resize):
    # We want the raw images without any normalization or random resizing

    model.eval()

    # COCO classes
    class_id_to_label = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    num_img_log = 50
    count = 0

    for samples, targets in dataloader:
        #print(samples)
        #print(targets)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        batch_size = len(targets)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        images, masks = samples.decompose()

        # each image
        for i in range(batch_size):
            image = images[i]
            result = results[i]
            target = targets[i]

            # Save for debug
            torch.save(result, f"result{count}.pth")
            torch.save(target, f"target{count}.pth")
            
            if len(target) == 0:
                # print("no boxes, skip this image")
                continue
            
            # Find the raw image matching these boxes "image_id" field
            # TODO this will be broken for our reduced COCO dataset
            # if we ever want to visualize results on that.
            image_id = target["image_id"].item()  # otherwise KeyError
            orig_size = target["orig_size"].tolist()
            # print("original size is", orig_size)

            image_path = dataset_val_without_resize.coco.loadImgs([image_id])[0]['file_name']
            
            # TODO use the gt
            ann_ids = dataset_val_without_resize.coco.getAnnIds(imgIds=image_id)
            gt = dataset_val_without_resize.coco.loadAnns(ann_ids)

            image = dataset_val_without_resize.get_image(image_path)
            assert abs(orig_size[0] - image.height) <= 1, "height mismatch"
            assert abs(orig_size[1] - image.width) <= 1, "width mismatch"

            # TODO could also show gt boxes?

            scores = result['scores']
            labels = result['labels']
            boxes = result['boxes'].cpu()
            
            box_data = []

            # each box
            for j in range(len(labels)):
                score = scores[j].item()
                label = labels[j].item()
                box = boxes[j]

                if score < 0.05 :
                    continue

                box_data_j = {
                    "position": {
                        "minX": box[0].item(),
                        "maxX": box[2].item(),
                        "minY": box[1].item(),
                        "maxY": box[3].item()
                    },
                    "class_id": label,
                    "box_caption": "%s (%.3f)" % (class_id_to_label.get(label, f"unknown-label-{label}"), score),
                    "domain": "pixel",
                    "scores": {"score": score}
                }

                box_data.append(box_data_j)

            boxes = {
                "predictions": {
                    "box_data": box_data,
                    "class_labels": class_id_to_label
                },
                #"ground_truth": {} # would be nice to have
            }

            img = wandb.Image(image, boxes=boxes)
            wandb.log({f"{count}": img})
            # wandb.log({"visualized_predictions": img})
            count += 1

            if count > num_img_log - 1: return

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # Save our Wandb metadata
    if not args.no_wb:
        wandb.init(entity='dl-project', project='dl-final-project', name=args.wb_name, notes=args.wb_notes, reinit=True)
    wandb.config.epochs = args.epochs

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    # visualize_video(model, postprocessors)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)
    wandb.config.n_parameters = n_parameters
    wandb.config.n_trainable_parameters = n_parameters  # better name

    # Log total # of model parameters (including frozen) to W&B
    n_total_parameters = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', n_total_parameters)
    wandb.config.n_total_parameters = n_total_parameters

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # For visualization we want the raw images without any normalization or random resizing
    dataset_val_without_resize = CocoDetection(
        "data/coco/val2017",
        annFile="data/coco/annotations/instances_val2017.json",
        transforms=T.Compose([T.ToTensor()])
    )

    # Save metadata about training + val datasets and batch size
    wandb.config.len_dataset_train = len(dataset_train)
    wandb.config.len_dataset_val = len(dataset_val)
    wandb.config.batch_size = args.batch_size

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    # Not sure if we should save all hyperparameters in wandb.config?
    # just start with a few important ones
    wandb.config.lr = args.lr
    wandb.config.lr_backbone = args.lr_backbone

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
    
    if args.eval:

        print("Generating visualizations...")
        visualize_bbox(model, postprocessors, data_loader_val, device, dataset_val_without_resize)
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_file_for_wb = str(output_dir / f'{wandb.run.id}_checkpoint{epoch:04}.pth')
            checkpoint_paths = [
                output_dir / 'checkpoint.pth',
                checkpoint_file_for_wb
            ]

            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            
             # Save model checkpoint to W&B
            wandb.save(checkpoint_file_for_wb)

        # Generate visualizations for fixed(?) set of images every epoch
        print("Generating visualizations...")
        visualize_bbox(model, postprocessors, data_loader_val, device, dataset_val_without_resize)
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        # Save the COCO metrics properly
        metric_name = ["AP", "AP50", "AP75", "APs", "APm", "APl",
                "AR@1", "AR@10", "AR@100", "ARs", "ARm", "ARl"]
        for i, metric_val in enumerate(log_stats["test_coco_eval_bbox"]):
            log_stats[metric_name[i]] = metric_val
        
        if not args.no_wb:
            wandb.log(log_stats)
        print("train_loss: ", log_stats['train_loss'])

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            wandb.save(str(output_dir / "log.txt"))

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    eval_filename_for_wb = f'{wandb.run.id}_eval_{epoch:04}.pth'
                    eval_path_for_wb = str(output_dir / "eval" / eval_filename_for_wb)
                    filenames = ['latest.pth', eval_filename_for_wb]
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

                    # TODO not sure if this file will end up being too big
                    # I think it's the COCO precision/recall metrics
                    # in some format...
                    # let's track it just in case to start!
                    wandb.save(eval_path_for_wb)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # print("Generating visualizations...")
    #visualize_bbox(model, postprocessors, data_loader_val, device, dataset_val_without_resize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
