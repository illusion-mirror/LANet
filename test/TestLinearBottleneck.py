from model import LinearBottleneck
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
to_tensor = ToTensor()
to_pil = ToPILImage
img1 = Image.open(r'D:\Data\ship_data\data\ship_saliency\Ship_Data2\imgs\000000.jpg')
#img1.show()
input = to_tensor(img1).unsqueeze(0)
linearBottleneck1 = LinearBottleneck(3,3,1,6)
out = linearBottleneck1(input)
pil = to_pil(out.data.squeeze(0))
print(pil)

