from torch.utils.data import DataLoader
from typing import Any

def get_data(path:str,dataset:str,augment:bool=False) -> tuple[Any,Any]:
    if dataset=='MNIST':
        import MNIST.dataset
        train_data,test_data=MNIST.dataset.get_dataset(path)
    elif dataset=='FMNIST':
        import FMNIST.dataset
        train_data,test_data=FMNIST.dataset.get_dataset(path)
    elif dataset=='CIFAR10':
        import CIFAR10.dataset
        train_data,test_data=CIFAR10.dataset.get_dataset(path,augment)
    elif dataset=='CIFAR100':
        import CIFAR100.dataset
        train_data,test_data=CIFAR100.dataset.get_dataset(path,augment)
    elif dataset=='DVSCIFAR10':
        import DVSCIFAR10.dataset
        train_data,test_data=DVSCIFAR10.dataset.get_dataset(path,augment)
    elif dataset=='ImageNet100':
        import ImageNet100.dataset
        train_data,test_data=ImageNet100.dataset.get_dataset(path)
    return train_data,test_data

def get_dataloader(train_data:Any,test_data:Any,batch_size:int,shuffle:bool) -> tuple[DataLoader,DataLoader]:
    train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=shuffle,pin_memory=True)
    test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,pin_memory=True)
    return train_loader,test_loader

def load_dataset_mnist(path:str,batch_size:int,shuffle:bool) -> tuple[DataLoader,DataLoader,tuple[int,int,int]]:
    train_data,test_data=get_data(path,'MNIST')
    train_loader,test_loader=get_dataloader(train_data,test_data,batch_size,shuffle)
    data_shape=(1,28,28)
    return train_loader,test_loader,data_shape

def load_dataset_fmnist(path:str,batch_size:int,shuffle:bool) -> tuple[DataLoader,DataLoader,tuple[int,int,int]]:
    train_data,test_data=get_data(path,'FMNIST')
    train_loader,test_loader=get_dataloader(train_data,test_data,batch_size,shuffle)
    data_shape=(1,28,28)
    return train_loader,test_loader,data_shape

def load_dataset_cifar10(augment:bool,path:str,batch_size:int,shuffle:bool) -> tuple[DataLoader,DataLoader,tuple[int,int,int]]:
    train_data,test_data=get_data(path,'CIFAR10',augment)
    train_loader,test_loader=get_dataloader(train_data,test_data,batch_size,shuffle)
    data_shape=(3,32,32)
    return train_loader,test_loader,data_shape

def load_dataset_cifar100(augment:bool,path:str,batch_size:int,shuffle:bool) -> tuple[DataLoader,DataLoader,tuple[int,int,int]]:
    train_data,test_data=get_data(path,'CIFAR100',augment)
    train_loader,test_loader=get_dataloader(train_data,test_data,batch_size,shuffle)
    data_shape=(3,32,32)
    return train_loader,test_loader,data_shape

def load_dataset_dvscifar10(augment:bool,path:str,batch_size:int,shuffle:bool) -> tuple[DataLoader,DataLoader,tuple[int,int,int]]:
    train_data,test_data=get_data(path,'DVSCIFAR10',augment)
    train_loader,test_loader=get_dataloader(train_data,test_data,batch_size,shuffle)
    data_shape=(2,48,48)
    return train_loader,test_loader,data_shape

def load_dataset_imagenet100(path:str,batch_size:int,shuffle:bool) -> tuple[DataLoader,DataLoader,tuple[int,int,int]]:
    train_data,test_data=get_data(path,'ImageNet100')
    train_loader,test_loader=get_dataloader(train_data,test_data,batch_size,shuffle)
    data_shape=(3,224,224)
    return train_loader,test_loader,data_shape