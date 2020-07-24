from model import SPP
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
to_tensor = ToTensor()
to_pil = ToPILImage
img1 = Image.open(r'D:\Data\ship_data\data\ship_saliency\Ship_Data2\imgs\000000.jpg')
#img1.show()
input = to_tensor(img1).unsqueeze(0)
print(input.size())
# laNet = LANet()
# out = laNet(input)
# pil = to_pil(out.data.squeeze(0))
spp = SPP(3)
out = spp(input)
print(out.size())

