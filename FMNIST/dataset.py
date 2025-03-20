import os,sys

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from torchvision import datasets,transforms

def get_dataset(root_path:str) -> tuple[datasets.MNIST,datasets.MNIST]:
    train_data=datasets.MNIST(root=root_path,train=True,transform=transforms.ToTensor(),download=True)
    test_data=datasets.MNIST(root=root_path,train=False,transform=transforms.ToTensor(),download=True)
    return train_data,test_data


# if __name__=='__main__':
#     datasets.MNIST(root=root_path+'/data',train=True,transform=transforms.ToTensor(),download=True)
#     datasets.MNIST(root=root_path+'/data',train=False,transform=transforms.ToTensor(),download=True)