import torchvision
from PIL import ImageDraw

coco_train = torchvision.datasets.CocoDetection(
    "Deformable-DETR/data/coco/train2017",
    annFile="Deformable-DETR/data/coco/annotations/instances_train2017.json",
)

MAX_IMAGES = 20
i = 0
for img, target in coco_train:
    for instance in target:
        x, y, w, h = instance["bbox"]  # ??? coords?
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + w, y + h], outline="red")

    img.show()

    i += 1
    if i > MAX_IMAGES:
        break
