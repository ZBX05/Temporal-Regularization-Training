import os
import sys

root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from model import SNN
from train import *
from dataset import *
import argparse
import random
import numpy as np

def main():
    parser=argparse.ArgumentParser(description='Training fully-connected and convolutional spiking neural networks.')
    # Device
    parser.add_argument('--cpu',type=int,default=0,help='Disable CUDA and run on CPU.')
    parser.add_argument('--parallel',type=int,default=0,help='Whither to use multiple GPUs.')
    parser.add_argument('--gpu',type=str,default='0',help='GPU(s) ID. When using parallel training, the IDs must be specified as a string of comma-separated integers, like 0-1-2-3. Default: 0.')
    parser.add_argument('--seed',type=int,default=42)
    # Dataset
    parser.add_argument('--dataset',type=str,choices=['MNIST','FMNIST','CIFAR10','CIFAR100','DVSCIFAR10','ImageNet100'],default='MNIST',help='Choice of the dataset: MNIST (MNIST), Fashion-MNIST (FMNIST), CIFAR-10 (CIFAR10), CIFAR10-DVS (DVSCIFAR10). Default: MNIST.')
    parser.add_argument('--augment',type=int,default=1,help='Whether to use cutout for CIFAR-10 and CIFAR10-DVS. Default: True.')
    #Training
    parser.add_argument('--save_checkpoint',type=int,default=0,help='Whether to save the checkpoints. Default: False.')
    parser.add_argument('--checkpoint_epochs',type=str,default='150-200-250',help='Epochs for saving the checkpoints. Default: 150-200-250.')
    parser.add_argument('--observe_fi',type=int,default=0,help='Whether to observe the Fisher Information. Default: False.')
    parser.add_argument('--fi_epochs',type=str,default='20-40-60-80-100-120-200-300',help='Checkpoints for FI observation. Default: 20-40-60-80-100-120-200-300.')
    parser.add_argument('--optimizer',choices=['SGD','AdamW','Adam','RMSprop'],default='Adam',help='Choice of the optimizer - stochastic gradient descent with 0.9 momentum (SGD), SGD with 0.9 momentum and AdamW (AdamW), Adam (Adam), and RMSprop (RMSprop). Default: AdamW.')
    parser.add_argument('--l1',type=float,default=0,help='L1 regularization coefficient. Default: 0.')
    parser.add_argument('--l2',type=float,default=0,help='L2 regularization coefficient. Default: 0.')
    parser.add_argument('--criterion',choices=['MSE','BCE','CE'], default='CE',help='Choice of criterion (loss function) - mean squared error (MSE), binary cross entropy (BCE), cross entropy (CE, which already contains a logsoftmax activation function). Default: MSE.')
    parser.add_argument('--tetloss',type=int,default=0)
    parser.add_argument('--loss_means',type=float,default=1.0)
    parser.add_argument('--resumeloss',type=int,default=0)
    parser.add_argument('--regloss',type=int,default=1,help='Whether to use the Temporal Regularization Training method. Default: False.')
    parser.add_argument('--loss_decay',type=float,default=0.5,help='Means afctor for Temporal Regularization Training loss function, make all the potential increment around the means. Default: 0.5.')
    parser.add_argument('--loss_lambda',type=float,default=1e-5,help='Lambda factor for Temporal Regularization Training loss function. Default: 0.00001.')
    parser.add_argument('--loss_epsilon',type=float,default=1e-5)
    parser.add_argument('--loss_eta',type=float,default=0.05)
    parser.add_argument('--mean_reduce',type=int,default=1,help='Whether to reduce the mean of the loss function. Default: True.')
    parser.add_argument('--dropout',type=float,default=0.0,help='Dropout probability (applied only to fully-connected layers). Default: 0.')
    parser.add_argument('--epochs',type=int,default=300,help='Number of training epochs Default: 300.')
    parser.add_argument('--batch_size',type=int,default=64,help='Input batch size for training. Default: 256.')
    parser.add_argument('--lr',type=float,default=1e-3,help='Learning rate. Default: 1e-3.')
    parser.add_argument('--scheduler',type=int,default=1,help='Whether to use a learning rate scheduler (CosineAnnealingLR). Default: False.')
    parser.add_argument('--amp',type=int,default=1)
    parser.add_argument('--resume',type=int,default=0,help='Whether to resume training from a checkpoint. Default: False.')
    parser.add_argument('--resume_path',type=str,default='',help='Path to the checkpoint for resuming training.')
    # Network 'CONV_32_5_1_2_FC_1000_FC_10'
    parser.add_argument('--init',type=int,default=1)
    parser.add_argument('--topology',type=str,default='CONV-28-5-1-2_FC-1000_FC-10',help='Choice of network topology. Format for convolutional layers: CONV_{output channels}-{kernel size}-{stride}-{padding}. Format for fully-connected layers: FC-{output units}.')
    parser.add_argument('--norm',type=str,choices=['tdBN','No'],default='tdBN',help='Choice of normalization method - batch normalization (BN), temporal decoupled batch normalization (tdBN). Default: BN.')
    parser.add_argument('--T',type=int,default=10,help='The time window for spiking neurons. Default: 10.')
    parser.add_argument('--v_threshold',type=float,default=1.0,help='Threshold potential for spiking neurons. Default: 0.5.')
    parser.add_argument('--v_reset',type=float,default=0.0,help='Reset potential for spiking neurons. Default: 0.')
    parser.add_argument('--tau',type=float,default=2.0,help='Time constant for spiking neurons, decay factor equals to 1/tau. Default: 5.0')
    parser.add_argument('--surrogate_type',choices=['sigmoid','zo','pseudo','triangle','asgl'],default='triangle',help='Choice of surrogate function for spiking neurons - sigmoid (sigmoid), dynamic zeroth order method (dzo), zeroth order method (zo), and pseudo-bp (pseudo). Default: dzo.')
    parser.add_argument('--surrogate_m',type=float,default=5)
    parser.add_argument('--surrogate_param',type=float,default=1.0)

    args=parser.parse_args()

    seed=args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    if args.parallel:
        gpus=args.gpu.split('-')
        os.environ['CUDA_VISIBLE_DEVICES']=gpus[0] if len(gpus)==1 else ','.join(args.gpu.split('-'))
        device=torch.device('cuda')
    else:
        device=torch.device(f'cuda:{args.gpu}') if not args.cpu else torch.device('cpu')
    experiment_path=os.path.dirname(os.path.abspath(__file__))+f'/{args.dataset}'

    if args.v_reset<0:
        args.v_reset=None

    args.model='Custom'

    if args.dataset=='MNIST':
        train_data_loader,test_data_loader,input_shape=load_dataset_mnist(experiment_path+'/data',args.batch_size,True)
        args.label_size=10
        args.expend_time=True
    elif args.dataset=='FMNIST':
        train_data_loader,test_data_loader,input_shape=load_dataset_fmnist(experiment_path+'/data',args.batch_size,True)
        args.label_size=10
        args.expend_time=True
    elif args.dataset=='CIFAR10':
        train_data_loader,test_data_loader,input_shape=load_dataset_cifar10(args.augment,experiment_path+'/data',args.batch_size,True)
        args.label_size=10
        args.expend_time=True
        if args.topology=='ResNet-19':
            args.model='ResNet-19'
            args.topology=f'CONVNP-64-3-1-1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-256-3=3-2=1-1=1_RES-256-3=3-1=1-1=1_RES-256-3=3-1=1-1=1_RES-512-3=3-2=1-1=1_RES-512-3=3-1=1-1=1-_FC-256_L-{args.label_size}'
    elif args.dataset=='CIFAR100':
        train_data_loader,test_data_loader,input_shape=load_dataset_cifar100(args.augment,experiment_path+'/data',args.batch_size,True)
        args.label_size=100
        args.expend_time=True
        if args.topology=='ResNet-19':
            args.model='ResNet-19'
            args.topology=f'CONVNP-64-3-1-1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-256-3=3-2=1-1=1_RES-256-3=3-1=1-1=1_RES-256-3=3-1=1-1=1_RES-512-3=3-2=1-1=1_RES-512-3=3-1=1-1=1-_FC-256_L-{args.label_size}'
    elif args.dataset=='DVSCIFAR10':
        train_data_loader,test_data_loader,input_shape=load_dataset_dvscifar10(args.augment,experiment_path+'/data',args.batch_size,True)
        args.label_size=10
        args.expend_time=False
        if args.topology=='VGGSNN':
            args.model='VGGSNN'
            args.topology=f'CONVNP-64-3-1-1_CONVAP-128-3-1-1_CONVNP-256-3-1-1_CONVAP-256-3-1-1_CONVNP-512-3-1-1_CONVAP-512-3-1-1_CONVNP-512-3-1-1_CONVAP-512-3-1-1_L-{args.label_size}'
    elif args.dataset=='ImageNet100':
        train_data_loader,test_data_loader,input_shape=load_dataset_imagenet100(experiment_path+'/data',args.batch_size,True)
        args.label_size=100
        args.expend_time=True
        if args.topology=='ResNet-34':
            args.model='ResNet-34'
            args.topology=f'CONV-64-7-2-3-3-2-1_RES-64-3=3-1=1-1=1_RES-64-3=3-1=1-1=1_RES-64-3=3-1=1-1=1_RES-128-3=3-2=1-1=1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-256-3=3-2=1-1=1_RES-256-3=3-1=1-1=1_RES-256-3=3-1=1-1=1_RES-256-3=3-1=1-1=1_RES-256-3=3-1=1-1=1_RES-256-3=3-1=1-1=1_RES-512-3=3-2=1-1=1_RES-512-3=3-1=1-1=1_RES-512-3=3-1=1-1=1-_L-{args.label_size}'
        if args.topology=='SEW-ResNet-34':
            # args.expend_time=False
            args.model='SEW-ResNet-34'
            args.topology=f'CONV-64-7-2-3-3-2-1_SEWRES~ADD-64-3=3-1=1-1=1_SEWRES~ADD-64-3=3-1=1-1=1_SEWRES~ADD-64-3=3-1=1-1=1_SEWRES~ADD-128-3=3-2=1-1=1_SEWRES~ADD-128-3=3-1=1-1=1_SEWRES~ADD-128-3=3-1=1-1=1_SEWRES~ADD-128-3=3-1=1-1=1_SEWRES~ADD-256-3=3-2=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-512-3=3-2=1-1=1_SEWRES~ADD-512-3=3-1=1-1=1_SEWRES~ADD-512-3=3-1=1-1=1-_L-{args.label_size}'
    else:
        raise ValueError('Unsupported dataset: '+args.dataset+'.')

    surrogate_param=args.surrogate_param
    
    if args.surrogate_type!='asgl':
        assert args.surrogate_m>0,'m must be an integer greater than 0.'
    else:
        assert 0<=args.surrogate_m<=1.0,'m indicates the probability of the spike mask, must be in [0,1].' 
    assert args.tau>=1,'tau must be greater than or equal to 1.0.'
    assert not(args.regloss and args.criterion=='MSE'),'MSE is not supported in TET.'

    args.weight_decay=None
    if args.l1!=0 and args.l2!=0:
        raise ValueError('Only one type of weight decay can be used in one experiment!')
    if args.l1:
        args.weight_decay={"type":'l1',"decay":args.l1}
    elif args.l2:
        args.weight_decay={"type":'l2',"decay":args.l2}

    model=SNN(args.topology,args.T,input_shape,args.dropout,args.norm,args.v_threshold,args.v_reset,args.tau,args.surrogate_type,
              surrogate_param,args.surrogate_m,args.expend_time,args.init)
    if args.parallel:
        model=torch.nn.DataParallel(model)
    train(args,model,train_data_loader,test_data_loader,device,experiment_path)

if __name__=='__main__':
    main()