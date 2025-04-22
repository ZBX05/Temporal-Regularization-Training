import os,sys

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from torchvision import datasets,transforms
import numpy as np
import torch

class Cutout(object):
    def __init__(self,n_holes:int,length:int) -> None:
        self.n_holes=n_holes
        self.length=length
    def __call__(self,img:torch.Tensor) -> torch.Tensor:
        h=img.size(1)
        w=img.size(2)

        mask=np.ones((h,w),np.float32)

        for n in range(self.n_holes):
            y=np.random.randint(h)
            x=np.random.randint(w)

            y1=np.clip(y-self.length//2,0,h)
            y2=np.clip(y+self.length//2,0,h)
            x1=np.clip(x-self.length//2,0,w)
            x2=np.clip(x+self.length//2,0,w)

            mask[y1:y2,x1:x2]=0.

        mask=torch.from_numpy(mask)
        mask=mask.expand_as(img)
        img=img*mask

        return img

def get_dataset(root_path:str,augment:bool=False) -> tuple[datasets.CIFAR100,datasets.CIFAR100]:
    train_transform=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1,length=16),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ]) if augment else transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    train_data=datasets.CIFAR100(root=root_path,train=True,transform=train_transform,download=True)
    test_data=datasets.CIFAR100(root=root_path,train=False,transform=test_transform,download=True)
    return train_data,test_data

# if __name__=='__main__':
#     datasets.CIFAR10(root=root_path+'/data',train=True,transform=train_transform,download=True)
#     datasets.CIFAR10(root=root_path+'/data',train=False,transform=test_transform,download=True)