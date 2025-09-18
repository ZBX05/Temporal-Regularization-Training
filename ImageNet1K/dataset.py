import os,sys

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from torchvision import datasets,transforms

def get_dataset(root_path:str) -> tuple[datasets.ImageFolder,datasets.ImageFolder]:
    train_transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    test_transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    train_data=datasets.ImageFolder(root=root_path+'/train',transform=train_transform)
    test_data=datasets.ImageFolder(root=root_path+'/val',transform=test_transform)
    return train_data,test_data

# if __name__=='__main__':
#     datasets.CIFAR10(root=root_path+'/data',train=True,transform=train_transform,download=True)
#     datasets.CIFAR10(root=root_path+'/data',train=False,transform=test_transform,download=True)