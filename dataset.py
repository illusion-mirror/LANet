import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image

class ImageData(data.Dataset):
    def __init__(self,image_path,label_path,transform,t_transform):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        label = Image.open(self.label_path[item]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image,label
    def __len__(self):

        return len(self.image_path)

'''
img_root:   root/to/image such as   d:/imgs
label_root: root/to/label           d:/G_T
filename:                           train.txt 
'''
'''
数据集的RGB平均值为
[0.18862618857240948,0.18871731868931105,0.18797078542627094]
数据集的RGB方差为
[0.20141003702612348,0.2015158028693538,0.20092256383873736]
'''
def dataloader(img_root,label_root,image_size,batch_size,filename =None,mode='train',num_thread=0):
    mean = torch.Tensor([128.68, 116.779, 103.939]).view(3, 1, 1)/255
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x:x-mean)
    ])
    if mode=='train':
        t_transform = transforms.Compose({
            transforms.Resize((image_size // 2,image_size//2)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : torch.round(x))
        })
    else:
        t_transform = transforms.Compose([
            #transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))
        ])
    #取得路径列表
    image_path = []
    label_path = []
    if filename is None:
        image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
        label_path = list(
            map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), image_path))
    else:
        lines = [line.rstrip('\n')[:-3] for line in open(filename)]
        image_path = list(map(lambda x: os.path.join(img_root, x + 'jpg'), lines))
        label_path = list(map(lambda x: os.path.join(label_root, x + 'png'), lines))
    dataset = ImageData(image_path, label_path, transform, t_transform)
    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,num_thread=num_thread)
    return data_loader