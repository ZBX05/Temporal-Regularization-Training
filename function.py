import torch
import torch.nn.functional as F
import warnings
from typing import Any
from math import exp

warnings.filterwarnings('ignore',category=Warning)

def rectangle(x:torch.Tensor,param:torch.Tensor) -> torch.Tensor:
    y=torch.clamp(torch.exp(param)*x+0.5,0,1.0)
    return y

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,surrogate_type,param,m=None):
        if not isinstance(param,torch.Tensor):
            param=torch.tensor([param],device=input.device)
        ctx.save_for_backward(input,param)
        ctx.in_1=surrogate_type
        ctx.in_2=m
        if surrogate_type!='asgl':
            output=input.gt(0).float()
        else:
            h_s=rectangle(input,param)
            output=h_s+((input.gt(0).float()-h_s)*m).detach()
        return output # spike=(mem-self.v_threshold)>0

    @staticmethod
    def backward(ctx,grad_output):
        grad_input=grad_output.clone()
        input,param=ctx.saved_tensors
        surrogate_type=ctx.in_1
        surrogate_m=ctx.in_2
        param=param.item() if param.shape[0]==1 else param
        param_grad=None
        if surrogate_type=='sigmoid':
            sgax=1/(1+torch.exp(-param*input))
            grad_surrogate=param*(1-sgax)*sgax
        elif surrogate_type=='zo':
            sample_size=surrogate_m
            abs_z=torch.abs(torch.randn((sample_size,)+input.size(),device=input.device,dtype=torch.float))
            t=torch.abs(input[None,:,:])<abs_z*param
            grad_surrogate=torch.mean(t*abs_z,dim=0)/(2*param)
        elif surrogate_type=='triangle':
            grad_surrogate=(1/param)*(1/param)*((param-input.abs()).clamp(min=0))
        elif surrogate_type=='pseudo':
            grad_surrogate=abs(input)<param
        elif surrogate_type=='asgl':
            t=torch.abs(input)<1/(2*torch.exp(param))
            grad_surrogate=t*torch.exp(param)
            param_grad=t*torch.exp(param)*input*grad_input
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
            # sup_loss+=mse_loss(outputs[:,t,...].float(),labels_one_hot)
        # loss+=label_loss+reg
        loss+=(1-eta)*label_loss+eta*sup_loss+reg
    # if eta!=0:
        # loss=(1-eta)*loss+eta*sup_loss
    loss=loss/T
    return loss