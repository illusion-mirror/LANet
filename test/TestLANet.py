from model import LinearBottleneck
from torchvision.transforms import ToTensor, ToPILImage
from model import LANet
from PIL import Image
to_tensor = ToTensor()
to_pil = ToPILImage
img1 = Image.open(r'D:\Data\ship_data\data\ship_saliency\Ship_Data2\imgs\000000.jpg')
#img1.show()
input = to_tensor(img1).unsqueeze(0)
laNet = LANet()
out = laNet(input)
print(out.size())

