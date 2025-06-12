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

def get_dataset(root_path:str,T:int) -> tuple[NCaltech101,NCaltech101]:
    transform=transforms.Compose([FromNP(),transforms.Resize((48,48))])
    data=NCaltech101(root=root_path,data_type='frame',frames_number=T,split_by='number',transform=transform)
    print('Spliting dataset...')
    train_data,test_data=split_to_train_test_set(train_ratio=0.9,origin_dataset=data,num_classes=101)
    print('Spliting done.')
    return train_data,test_data

# if __name__=='__main__':
#     train_data,test_data=get_dataset(r'F:\Temporal_Regularization_Training\DVSGesture128\data',10)
#     from torch.utils.data import DataLoader
#     train_loader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
#     for x,y in train_loader:
#         print(x.shape)
#         exit()