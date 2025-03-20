import os,sys

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any

class DVSCifar10(Dataset):
    def __init__(self,root:str,train:bool=True,transform:bool=False,target_transform:Any=None) -> None:
        self.root=os.path.expanduser(root)
        self.transform=transform
        self.target_transform=target_transform
        self.train=train
        self.resize=transforms.Resize(size=(48,48))  # 48 48
        self.tensorx=transforms.ToTensor()
        self.imgx=transforms.ToPILImage()

    def __getitem__(self,index:int) -> tuple[torch.Tensor,torch.Tensor]:
        '''
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        '''
        data_path=self.root+'/{}/{}.pt'.format('train' if self.train else 'test',index)
        data,target=torch.load(data_path)
        new_data=[]
        for t in range(data.size(0)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t,...]))))
        data=torch.stack(new_data,dim=0)
        if self.transform:
            flip=random.random()>0.5
            if flip:
                data=torch.flip(data,dims=(3,))
            off1=random.randint(-5,5)
            off2=random.randint(-5,5)
            data=torch.roll(data,shifts=(off1,off2),dims=(2,3))

        if self.target_transform is not None:
            target=self.target_transform(target)
        return data,target.long().squeeze(-1)

    def __len__(self) -> int:
        return len(os.listdir(self.root+'/train' if self.train else self.root+'/test'))

def get_dataset(root_path:str,augment:bool) -> tuple[DVSCifar10,DVSCifar10]:
    train_data=DVSCifar10(root=root_path,train=True,transform=augment)
    test_data=DVSCifar10(root=root_path,train=False)
    return train_data,test_data