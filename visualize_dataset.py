# Show some images and labels from a COCO dataset
# Download all of the images referenced by your annotation file first
# Look at make_coco_subset.py
import random

import torchvision
from PIL import ImageDraw

coco_train = torchvision.datasets.CocoDetection(
    "Deformable-DETR/data/coco/val2017",
    annFile="Deformable-DETR/data/coco/annotations/instances_val2017_5classes_areagt100.json",
)

MAX_IMAGES = 20
i = 0
for img, target in coco_train:
    if random.random() > 0.05: # TODO easier way to get 20 random images
        continue
    for instance in target:
        x, y, w, h = instance["bbox"]  # ??? coords?
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + w, y + h], outline="red")

    img.show()

    i += 1
    if i > MAX_IMAGES:
        break
