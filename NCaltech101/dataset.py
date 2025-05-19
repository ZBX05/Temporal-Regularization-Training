import os,sys

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.datasets import split_to_train_test_set
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

class FromNP:
    def __call__(self,x:np.ndarray):
        return torch.from_numpy(x)

class Cutout(object):
    def __init__(self,n_holes:int,length:int) -> None:
        self.n_holes=n_holes
        self.length=length
    def __call__(self,img:torch.Tensor) -> torch.Tensor:
        h=img.size(2)
        w=img.size(3)

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

class EventAugmentation(object):
    def __init__(self,augmentation_space:dict) -> None:
        self.augmentation_space=augmentation_space
        self.transforms=self._build_transforms()

    def _build_transforms(self) -> transforms.Compose:
        aug_list=[]
        for aug_name,(param_range,apply) in self.augmentation_space.items():
            if not apply:
                continue

            params=eval(param_range)
            if isinstance(params,torch.Tensor):
                params=params.numpy()
            
            param=np.random.choice(params)
            
            if aug_name=='ShearX':
                aug_list.append(transforms.RandomAffine(
                    degrees=0,shear=(param*180/np.pi,param*180/np.pi)))
            elif aug_name=='ShearY':
                aug_list.append(transforms.RandomAffine(
                    degrees=0,shear=(0,0,param*180/np.pi,param*180/np.pi)))
            elif aug_name=='TranslateX':
                aug_list.append(transforms.RandomAffine(
                    degrees=0,translate=(param/48,0)))
            elif aug_name=='TranslateY':
                aug_list.append(transforms.RandomAffine(
                    degrees=0,translate=(0,param/48)))
            elif aug_name=='Rotate':
                aug_list.append(transforms.RandomRotation(degrees=(param,param)))
            elif aug_name=='Cutout':
                aug_list.append(Cutout(n_holes=1,length=int(param)))

        return transforms.Compose(aug_list)

    def __call__(self,x:torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

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
    augmentation_space={
        "Identity":['torch.tensor(0.0)',False],
        "ShearX":['torch.linspace(-0.3, 0.3, 31)',True],
        "ShearY":['torch.linspace(-0.3, 0.3, 31)',True],
        "TranslateX":['torch.linspace(-0.5, 5.0, 31)',True],
        "TranslateY":['torch.linspace(-0.5, 5.0, 31)',True],
        "Rotate":['torch.linspace(-30.0, 30.0, 31)',True],
        "Cutout":['torch.linspace(1.0, 30.0, 31)',True],
    }
    train_transform=transforms.Compose([FromNP(),transforms.Resize((48,48)),transforms.RandomHorizontalFlip(),
                                        EventAugmentation(augmentation_space)])
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