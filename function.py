import torch
import torch.nn.functional as F
import warnings
from typing import Any
from math import exp
from copy import deepcopy
from tqdm import tqdm
from tensorboardX import SummaryWriter
import logging

warnings.filterwarnings('ignore',category=Warning)

def rectangle(x:torch.Tensor,param:torch.Tensor) -> torch.Tensor:
    y=torch.clamp(torch.exp(param)*x+0.5,0,1.0)
    return y

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,surrogate_type,param):
        if not isinstance(param,torch.Tensor):
            param=torch.tensor([param],device=input.device)
        ctx.save_for_backward(input,param)
        ctx.in_1=surrogate_type
        output=input.gt(0).float()
        return output # spike=(mem-self.v_threshold)>0

    @staticmethod
    def backward(ctx,grad_output):
        grad_input=grad_output.clone()
        input,param=ctx.saved_tensors
        surrogate_type=ctx.in_1
        param=param.item() if param.shape[0]==1 else param
        param_grad=None
        if surrogate_type=='sigmoid':
            sgax=1/(1+torch.exp(-param*input))
            grad_surrogate=param*(1-sgax)*sgax
        elif surrogate_type=='triangle':
            grad_surrogate=(1/param)*(1/param)*((param-input.abs()).clamp(min=0))
        else:
            raise NameError('Surrogate type '+str(surrogate_type)+' is not supported!')
        return grad_surrogate.float()*grad_input,None,param_grad,None

def TRT_Loss(model:torch.nn.Module,outputs:torch.Tensor,labels:torch.Tensor,criterion:Any,decay:float,lamb:float,epsilon:float,
             eta:float=0.05) -> torch.Tensor:
    T=outputs.size(1)
    loss=0
    sup_loss=0
    mse_loss=torch.nn.MSELoss()
    labels_one_hot=F.one_hot(labels,outputs.size(-1)).float()
    for t in range(T):
        reg=0
        label_loss=criterion(outputs[:,t,...].float(),labels)
        for name,param in model.named_parameters():
            if 'bias' not in name:
                decay_factor=lamb/(1+(exp(decay*t)-1)*(torch.abs(param)+epsilon))
                reg+=torch.sum(param**2*decay_factor)
        if eta!=0:
            sup_loss=mse_loss(outputs[:,t,...].float(),labels_one_hot)
        loss+=(1-eta)*label_loss+eta*sup_loss+reg
    loss=loss/T
    return loss

def ERT_Loss(model:torch.nn.Module,outputs:torch.Tensor,labels:torch.Tensor,criterion:Any,decay:float,lamb:float,epsilon:float,) -> torch.Tensor:
    T=outputs.size(1)
    loss=0
    for t in range(T):
        reg=0
        label_loss=criterion(outputs[:,t,...].float(),labels)
        for name,param in model.named_parameters():
            if 'bias' not in name:
                decay_factor=lamb/(1+(exp(decay*t)-1)*(torch.abs(param)+epsilon))
                reg+=torch.sum(param**2*decay_factor)
        loss+=label_loss+reg
    loss=loss/T
    return loss

def TET_Loss(outputs:torch.Tensor,labels:torch.Tensor,criterion:Any,means:float,lamb:float) -> torch.Tensor:
    T=outputs.size(1)
    Loss_es=0
    for t in range(T):
        Loss_es+=criterion(outputs[:,t,...],labels)
    Loss_es=Loss_es/T # L_TET
    if lamb!=0:
        MMDLoss=torch.nn.MSELoss()
        y=torch.zeros_like(outputs).fill_(means)
        Loss_mmd=MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd=0
    return (1-lamb)*Loss_es+lamb*Loss_mmd # L_Total

def FI_Observation(model:torch.nn.Module,train_data_loader:torch.utils.data.DataLoader,epoch:int,T:int,device:torch.device,logging:logging,
                   writer:SummaryWriter) -> None:
    print('Start to calculate the Fisher Information in epoch {:3d}'.format(epoch))
    logging.info('Start to calculate the Fisher Information in epoch {:3d}'.format(epoch))
    # fisherlist=[[] for _ in range(T)]
    ep_fisher_list=[]
    N=len(train_data_loader.dataset)
    for t in range(1,T+1):
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        precision_matrices={}
        for n,p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data
        model.eval()

        for step,(img,labels) in enumerate(tqdm(train_data_loader)):
            model.zero_grad()
            img=img.to(device)
            labels=labels.to(device)
            output=model(img,True)
            loss=F.nll_loss(F.log_softmax(torch.sum(output[:,:t,...],dim=1)/t,dim=1),labels)
            loss.backward()

            for n,p in model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data+=p.grad.data**2/100

            if step==100:
                break

        precision_matrices={n:p for n,p in precision_matrices.items()}
        fisher_trace_info=0
        for p in precision_matrices:
            weight=precision_matrices[p]
            fisher_trace_info+=weight.sum()
        # fisher_trace_info/=N

        print('Time: {:2d} | FisherInfo: {:4f}'.format(t,fisher_trace_info))
        logging.info('Time: {:2d} | FisherInfo: {:4f}'.format(t,fisher_trace_info))
        # fisherlist[t-1].append(float(fisher_trace_info.cpu().data.numpy()))
        ep_fisher_list.append(float(fisher_trace_info.cpu().data.numpy()))
        writer.add_scalar(f'fisher_trace_info_curve_{epoch}',ep_fisher_list[-1],t)

        print('Fisher list: ',ep_fisher_list)
        logging.info('Fisher list: '+str(ep_fisher_list))