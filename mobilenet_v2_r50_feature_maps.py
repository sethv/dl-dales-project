# Figure out the shapes of intermediate layers produced by
# mobilenet_v2
# resnet50
# From https://github.com/pytorch/vision/issues/3048
import torch
import torchvision

print(torchvision.__version__)
print("")

import torchvision.models as models

from torchvision.models._utils import IntermediateLayerGetter

backbone = models.mobilenet_v2(pretrained=None)

# We won't use backbone.forward() anyway, because it's for classification
# Have to call these sequentially then
# https://pytorch.org/docs/stable/_modules/torchvision/models/mobilenet.html
backbone.layer1 = backbone.features[0:4]
backbone.layer2 = backbone.features[4:7]
backbone.layer3 = backbone.features[7:14]
backbone.layer4 = backbone.features[14:-1]
backbone.features = None
backbone.classifier = None

return_layers = {
    "layer2": "0",
    "layer3": "1",
    "layer4": "2",
}  # they commented  out layer1 in backbone.py

backbone2 = IntermediateLayerGetter(backbone, return_layers)


x = torch.rand([1, 3, 512, 1024])
res = backbone2(x)

print("\nIntermediateLayerGetter mobilenet_v2 FMs")
for k in res:
    print(k, res[k].shape)

print("\nInitial mobilenet_v2 FMs")
m = models.mobilenet_v2()
a = torch.rand([1, 3, 512, 1024])
for i, l in enumerate(m.features):
    a = l(a)
    print(i, l.__class__.__name__, a.shape, end="")
    if i in [3, 6, 13, 17]:
        print(" <---- desired feature map layer?")

    print("")


# Now try with resnet50, should produce same feature map dimensions
# at some layers but with more channels
# (resnet18 would be in between mobilenet_v2 and resnet50)
print("\nr50 shape")

x = torch.rand([1, 3, 512, 1024])
print("x", x.shape)

r50 = torchvision.models.resnet50(pretrained=True)
z = r50.conv1(x)
z = r50.bn1(z)
z = r50.relu(z)
z = r50.maxpool(z)

# Potential layers used as feature maps by Deformable-DETR
print("before layer1", z.shape)
z = r50.layer1(z)
print("after layer1", z.shape)
z = r50.layer2(z)
print("after layer2", z.shape)
z = r50.layer3(z)
print("after layer3", z.shape)
z = r50.layer4(z)
print("after layer4", z.shape)
