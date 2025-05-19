import os,sys

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.datasets import split_to_train_test_set
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import random

class FromNP:
    def __call__(self,x:np.ndarray):
        return torch.from_numpy(x)

# class Cutout(object):
#     def __init__(self,n_holes:int,length:int) -> None:
#         self.n_holes=n_holes
#         self.length=length
#     def __call__(self,img:torch.Tensor) -> torch.Tensor:
#         h=img.size(1)
#         w=img.size(2)

#         mask=np.ones((h,w),np.float32)

#         for n in range(self.n_holes):
#             y=np.random.randint(h)
#             x=np.random.randint(w)

#             y1=np.clip(y-self.length//2,0,h)
#             y2=np.clip(y+self.length//2,0,h)
#             x1=np.clip(x-self.length//2,0,w)
#             x2=np.clip(x+self.length//2,0,w)

#             mask[y1:y2,x1:x2]=0.

#         mask=torch.from_numpy(mask)
#         mask=mask.expand_as(img)
#         img=img*mask

#         return img

# class EventAugmentation(object):
#     def __init__(self,augmentation_space:dict) -> None:
#         self.augmentation_space=augmentation_space
#         self.transforms=self._build_transforms()

#     def _build_transforms(self) -> transforms.Compose:
#         aug_list=[]
#         for aug_name,(param_range,apply) in self.augmentation_space.items():
#             if not apply:
#                 continue

#             params=eval(param_range)
#             if isinstance(params,torch.Tensor):
#                 params=params.numpy()
            
#             param=np.random.choice(params)
            
#             if aug_name=='ShearX':
#                 aug_list.append(transforms.RandomAffine(
#                     degrees=0,shear=(param*180/np.pi,param*180/np.pi)))
#             elif aug_name=='ShearY':
#                 aug_list.append(transforms.RandomAffine(
#                     degrees=0,shear=(0,0,param*180/np.pi,param*180/np.pi)))
#             elif aug_name=='TranslateX':
#                 aug_list.append(transforms.RandomAffine(
#                     degrees=0,translate=(param/48,0)))
#             elif aug_name=='TranslateY':
#                 aug_list.append(transforms.RandomAffine(
#                     degrees=0,translate=(0,param/48)))
#             elif aug_name=='Rotate':
#                 aug_list.append(transforms.RandomRotation(degrees=(param,param)))
#             elif aug_name=='Cutout':
#                 aug_list.append(Cutout(n_holes=1,length=int(param)))

#         return transforms.Compose(aug_list)

#     def __call__(self,x:torch.Tensor) -> torch.Tensor:
#         return self.transforms(x)

# class Rolling:
#     def __init__(self,c:int) -> None:
#         self.c=c

#     def __call__(self,img:torch.Tensor) -> torch.Tensor:
#         # Sample shift values
#         a=torch.randint(low=-self.c,high=self.c,size=(1,)).item()
#         b=torch.randint(low=-self.c,high=self.c,size=(1,)).item()

#         C,H,W=img.shape

#         # Horizontal shift
#         if a!=0:
#             if a>0:
#                 # Right shift with zero padding on left
#                 if a>=W:
#                     img=torch.zeros_like(img)
#                 else:
#                     img=torch.cat([torch.zeros_like(img[...,:a]),img[...,:-a]],dim=-1)
#             else:
#                 # Left shift with zero padding on right
#                 a_abs=-a
#                 if a_abs>=W:
#                     img=torch.zeros_like(img)
#                 else:
#                     img=torch.cat([img[...,a_abs:],torch.zeros_like(img[...,:a_abs])],dim=-1)

#         # Vertical shift
#         if b!=0:
#             if b>0:
#                 # Down shift with zero padding on top
#                 if b>=H:
#                     img=torch.zeros_like(img)
#                 else:
#                     img=torch.cat([torch.zeros_like(img[:,:b]),img[:,b:]],dim=1)
#             else:
#                 # Up shift with zero padding on bottom
#                 b_abs=-b
#                 if b_abs>=H:
#                     img=torch.zeros_like(img)
#                 else:
#                     img=torch.cat([img[:,b_abs:],torch.zeros_like(img[:,:b_abs])], dim=1)

#         return img

# class EDA(object):
#     def __init__(self,augmentation_space:dict) -> None:
#         self.augmentation_space=augmentation_space
#         self.transforms=self._build_transforms()

#     def _build_transforms(self) -> list:
#         aug_list=[]
#         for aug_name,(param,apply) in self.augmentation_space.items():
#             if not apply:
#                 continue
            
#             if aug_name=='ShearX':
#                 aug_list.append(transforms.RandomAffine(degrees=0,translate=None,scale=None,shear=(-param,param)))
#             elif aug_name=='Rolling':
#                 aug_list.append(Rolling(param))
#             elif aug_name=='Rotate':
#                 param=torch.empty(1).uniform_(-param,param).item()
#                 aug_list.append(transforms.RandomRotation(degrees=(param,param)))
#             elif aug_name=='Cutout':
#                 param=torch.randint(1,param,size=(1,)).item()
#                 aug_list.append(Cutout(n_holes=1,length=int(param)))

#         return aug_list

#     def __call__(self,x:torch.Tensor) -> torch.Tensor:
#         x_aug=[]
#         T=x.shape[0]
#         for t in range(T):
#             self.transforms=self._build_transforms()
#             x_aug.append(random.choice(self.transforms)(x[t,...]))
#         return torch.stack(x_aug,dim=0)

class TransformedDataset(Dataset):
    def __init__(self,dataset:Dataset,transform:transforms.Compose=None) -> None:
        self.dataset=dataset
        self.transform=transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self,idx:int) -> tuple[torch.Tensor,torch.Tensor]:
        data,target=self.dataset[idx]
        if self.transform:
            data=self.transform(data)
        return data,target

def get_dataset(root_path:str,T:int) -> tuple[NCaltech101,NCaltech101]:
    train_transform=transforms.Compose([FromNP(),transforms.Resize((48,48)),transforms.RandomHorizontalFlip()])
    test_transform=transforms.Compose([FromNP(),transforms.Resize((48,48))])
    data=NCaltech101(root=root_path,data_type='frame',frames_number=T,split_by='number')
    print('Spliting dataset...')
    train_data,test_data=split_to_train_test_set(train_ratio=0.9,origin_dataset=data,num_classes=101)
    print('Spliting done.')
    train_data=TransformedDataset(train_data,train_transform)
    test_data=TransformedDataset(test_data,test_transform)
    return train_data,test_data

# if __name__=='__main__':
#     train_data,test_data=get_dataset(r'F:\Temporal_Regularization_Training\DVSGesture128\data',10)
#     from torch.utils.data import DataLoader
#     train_loader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
#     for x,y in train_loader:
#         print(x.shape)
#         exit()