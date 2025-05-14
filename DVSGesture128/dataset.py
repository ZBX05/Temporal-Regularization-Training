import os,sys

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torchvision import transforms
import torch
import numpy as np

class FromNP:
    def __call__(self,x:np.ndarray):
        return torch.from_numpy(x)

def get_dataset(root_path:str,T:int) -> tuple[DVS128Gesture,DVS128Gesture]:
    transform=transforms.Compose([FromNP(),transforms.Resize(48)])
    train_data=DVS128Gesture(root=root_path,train=True,data_type='frame',frames_number=T,split_by='number',transform=transform)
    test_data=DVS128Gesture(root=root_path,train=False,data_type='frame',frames_number=T,split_by='number',transform=transform)
    return train_data,test_data

# if __name__=='__main__':
#     train_data,test_data=get_dataset(r'F:\Temporal_Regularization_Training\DVSGesture128\data',10)
#     from torch.utils.data import DataLoader
#     train_loader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
#     for x,y in train_loader:
#         print(x.shape)
#         exit()