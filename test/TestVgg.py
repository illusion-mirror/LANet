from model import vgg16
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
to_tensor = ToTensor()
to_pil = ToPILImage
img1 = Image.open(r'D:\Data\ship_data\data\ship_saliency\Ship_Data2\imgs\000000.jpg')
#img1.show()
input = to_tensor(img1).unsqueeze(0)
print(input.size())
base = {'352': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]}
vgg = nn.ModuleList(vgg16(base['352'],3))
for layer in vgg:
    input = layer(input)

out = input
print(out.size())