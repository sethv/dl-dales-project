import torch
import torchvision
from PIL import Image

img = Image.open("cat.jpg")
tensor = torchvision.transforms.ToTensor()(img)
print(tensor.shape)

# they appear to be saving the model's .state_dict()
model_state_dict = torch.load("r50_deformable_detr_single_scale-checkpoint.pth")

# https://pytorch.org/tutorials/beginner/saving_loading_models.html

# https://github.com/fundamentalvision/Deformable-DETR/blob/main/configs/r50_deformable_detr_single_scale.sh
# from this script we can see they ran python main.py with
# --num_feature_levels 1 (what we want)
model = DeformableDETR(*args, **kwargs)
model.load_state_dict(model_state_dict)
model.eval()
model(tensor)
