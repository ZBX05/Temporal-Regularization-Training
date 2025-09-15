import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *

class SNN(nn.Module):
    def __init__(self,topology:str,T:int,input_shape:tuple,dropout:float,norm:str,v_threshold:float,v_reset:float,tau:float,
                 surrogate_type:str,surrogate_param:float,expend_time:bool,init:bool) -> None:
        super(SNN,self).__init__()
        self.expend_time=expend_time
        self.layers=nn.Sequential()
        self.flatten=FlattenBlock(start_dim=2,end_dim=-1)
        self.T=T
        self.v_threshold=v_threshold
        self.v_reset=v_reset
        self.tau=tau
        self.surrogate_type=surrogate_type
        self.surrogate_param=surrogate_param
        
        # Build custom SNN
        self.topology=topology.split('_')
        for i,layer in zip(range(len(self.topology)),self.topology):
            layer=layer.split('-')
            try:
                if layer[0]=='CONV' or layer[0]=='CONVNP' or layer[0]=='CONVAP' or layer[0]=='CONVEN':
                    in_channels=input_shape[0] if i==0 else out_channels
                    out_channels=int(layer[1])
                    # input_dim=input_shape if i==0 else int(
                    #     output_dim / 2)  # /2 accounts for pooling operation of the previous convolutional layer
                    # output_dim = int((input_dim - int(layer[2]) + 2 * int(layer[4])) / int(layer[3])) + 1
                    input_dim=input_shape if i==0 else (self.layers[i-1].output_height,self.layers[i-1].output_width)
                    if len(layer)>5:
                        pool_kernel_size=int(layer[5])
                        pool_stride=int(layer[6])
                        pool_padding=int(layer[7])
                    else:
                        pool_kernel_size=2
                        pool_stride=2
                        pool_padding=0
                    if layer[0]=='CONV':
                        self.layers.append(Conv2dBlock(
                            norm=norm,
                            input_shape=input_dim,
                            in_channels=in_channels,
                            out_channels=int(layer[1]),
                            conv_kernel_size=int(layer[2]),
                            conv_stride=int(layer[3]),
                            conv_padding=int(layer[4]),
                            pool_kernel_size=pool_kernel_size,
                            pool_stride=pool_stride,
                            pool_padding=pool_padding,
                            bias=True if norm=='No' else False,
                            v_threshold=v_threshold,
                            v_reset=v_reset,
                            tau=tau,
                            surrogate_type=surrogate_type,
                            surrogate_param=surrogate_param
                        ))
                    elif layer[0]=='CONVNP':
                        self.layers.append(Conv2dNoPoolingBlock(
                            norm=norm,
                            input_shape=input_dim,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            conv_kernel_size=int(layer[2]),
                            conv_stride=int(layer[3]),
                            conv_padding=int(layer[4]),
                            bias=True if norm=='No' else False,
                            v_threshold=v_threshold,
                            v_reset=v_reset,
                            tau=tau,
                            surrogate_type=surrogate_type,
                            surrogate_param=surrogate_param
                        ))
                    elif layer[0]=='CONVAP':
                        self.layers.append(Conv2dAPBlock(
                            norm=norm,
                            input_shape=input_dim,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            conv_kernel_size=int(layer[2]),
                            conv_stride=int(layer[3]),
                            conv_padding=int(layer[4]),
                            pool_kernel_size=pool_kernel_size,
                            pool_stride=pool_stride,
                            pool_padding=pool_padding,
                            bias=True if norm=='No' else False,
                            v_threshold=v_threshold,
                            v_reset=v_reset,
                            tau=tau,
                            surrogate_type=surrogate_type,
                            surrogate_param=surrogate_param
                        ))
                    elif layer[0]=='CONVEN':
                        self.layers.append(Conv2dEncoderBlock(
                            T=T,
                            norm=norm,
                            input_shape=input_dim,
                            in_channels=in_channels,
                            out_channels=int(layer[1]),
                            conv_kernel_size=int(layer[2]),
                            conv_stride=int(layer[3]),
                            conv_padding=int(layer[4]),
                            pool_kernel_size=pool_kernel_size,
                            pool_stride=pool_stride,
                            pool_padding=pool_padding,
                            bias=True if norm=='No' else False,
                            v_threshold=v_threshold,
                            v_reset=v_reset,
                            tau=tau,
                            surrogate_type=surrogate_type,
                            surrogate_param=surrogate_param
                        ))
                    output_dim=self.layers[-1].output_height*self.layers[-1].output_width*self.layers[-1].out_channels
                elif layer[0]=='RES' or 'SEWRES' in layer[0]:
                    try:
                        pool=True if self.topology[i+1].split('-')[0]=='FC' or self.topology[i+1].split('-')[0]=='L' else False
                    except IndexError:
                        raise ValueError('Residual block cannot be the last layer of ResNet!')
                    in_channels=out_channels
                    try:
                        input_dim=(self.layers[i-1].output_height,self.layers[i-1].output_width)
                    except IndexError:
                        raise ValueError('Residual block cannot be the first layer of ResNet!')
                    out_channels=int(layer[1])
                    to_num=lambda l:tuple([int(s) for s in l])
                    conv_kernel_size=to_num(layer[2].split('='))
                    conv_stride=to_num(layer[3].split('='))
                    conv_padding=to_num(layer[4].split('='))
                    downsample=True if conv_stride[0]!=conv_stride[1] or in_channels!=out_channels else False
                    if layer[0]=='RES':
                        self.layers.append(ResidualBlock(
                            norm=norm,
                            input_shape=input_dim,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            conv_kernel_size=conv_kernel_size,
                            conv_stride=conv_stride,
                            conv_padding=conv_padding,
                            pool=pool,
                            downsample=downsample,
                            bias=True if norm=='No' else False,
                            v_threshold=v_threshold,
                            v_reset=v_reset,
                            tau=tau,
                            surrogate_type=surrogate_type,
                            surrogate_param=surrogate_param
                        ))
                    elif 'SEWRES' in layer[0]:
                        connect_function=layer[0].split('~')[1]
                        self.layers.append(SEWResidualBlock(
                            connect_function=connect_function,
                            norm=norm,
                            input_shape=input_dim,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            conv_kernel_size=conv_kernel_size,
                            conv_stride=conv_stride,
                            conv_padding=conv_padding,
                            pool=pool,
                            downsample=downsample,
                            bias=True if norm=='No' else False,
                            v_threshold=v_threshold,
                            v_reset=v_reset,
                            tau=tau,
                            surrogate_type=surrogate_type,
                            surrogate_param=surrogate_param
                        ))
                elif layer[0]=='FC' or layer[0]=='L':
                    if i==0:
                        input_dim=1
                        for i in range(len(input_shape)):
                            input_dim*=input_shape[i] 
                        self.layers.append(self.flatten)
                    elif 'CONV' in self.topology[i-1].split('-')[0] or 'RES' in self.topology[i-1].split('-')[0]:
                        input_dim=self.layers[i-1].output_height*self.layers[i-1].output_width*self.layers[i-1].out_channels
                        self.layers.append(self.flatten)
                    else:
                        input_dim=output_dim
                    output_dim=int(layer[1])
                    if layer[0]=='FC':
                        self.layers.append(FCBlock(
                            input_size=input_dim,
                            output_size=output_dim,
                            bias=True,
                            dropout=dropout,
                            v_threshold=v_threshold,
                            v_reset=v_reset,
                            tau=tau,
                            surrogate_type=surrogate_type,
                            surrogate_param=surrogate_param
                        ))
                    elif layer[0]=='L':
                        self.layers.append(SeqToANNContainer(nn.Linear(input_dim,output_dim,True)))
                else:
                    raise NameError('Layer construct '+str(layer[0])+' not supported!')
            except ValueError as e:
                raise ValueError('Unsupported layer parameter format: '+str(e)+'.')
        if init:
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    
    def forward(self,input:torch.Tensor,full_output:bool=False) -> torch.Tensor:
        x=input.unsqueeze(1).repeat(1,self.T,1,1,1) if self.expend_time else input
        x=self.layers(x)
        # for i in range(len(self.layers)):
        #     x=self.layers[i](x)
        #     if not isinstance(self.layers,FlattenBlock):
        #         print(x.shape)
        # exit()
        #    layer_output=[]
        #    for i in range(len(self.layers)):
        #        layer_output.append(self.layers[i](x))
        return x.mean(1) if not full_output else x
    
    def get_spike_firing_patterns(self,input:torch.Tensor,full_output:bool=False) -> tuple[torch.Tensor,list]:
        self.eval()
        spike_patterns=[]
        handles=[]
        neuron_hook=lambda module,input,output:spike_patterns.append(output.detach())
        for n,m in self.layers.named_modules():
            if isinstance(m,LIFNode):
                handles.append(m.register_forward_hook(neuron_hook))
        # for i in range(len(self.layers)):
        #     if hasattr(self.layers[i],'lif'):
        #         handles.append(self.layers[i].lif.register_forward_hook(neuron_hook))
        with torch.no_grad():
            output=self.forward(input,full_output)
        for handle in handles:
            handle.remove()
        return output,spike_patterns

if __name__=='__main__':
    # model=SNN('CONV-64-7-2-3-3-2-1_RES-64-3=3-1=1-1=1_RES-64-3=3-1=1-1=1_RES-128-3=3-2=1-1=1_RES-128-3=3-1=1-1=1_RES-256-3=3-2=1-1=1_RES-256-3=3-1=1-1=1_RES-512-3=3-2=1-1=1_RES-512-3=3-1=1-1=1-_FC-256_L-1000',4,(3,224,224),0.0,'tdBN',0.5,0.0,5.0,'sigmoid',2.0,5,True,True)
    model=SNN('CONV-64-7-2-3-3-2-1_SEWRES~ADD-64-3=3-1=1-1=1_SEWRES~ADD-64-3=3-1=1-1=1_SEWRES~ADD-64-3=3-1=1-1=1_SEWRES~ADD-128-3=3-2=1-1=1_SEWRES~ADD-128-3=3-1=1-1=1_SEWRES~ADD-128-3=3-1=1-1=1_SEWRES~ADD-128-3=3-1=1-1=1_SEWRES~ADD-256-3=3-2=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-256-3=3-1=1-1=1_SEWRES~ADD-512-3=3-2=1-1=1_SEWRES~ADD-512-3=3-1=1-1=1_SEWRES~ADD-512-3=3-1=1-1=1-_L-100',4,(3,224,224),0.0,'tdBN',0.5,0.0,5.0,'sigmoid',2.0,5,True,True)
    # model=SNN('CONVNP-64-3-1-1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-256-3=3-2=1-1=1_RES-256-3=3-1=1-1=1_RES-256-3=3-1=1-1=1_RES-512-3=3-2=1-1=1_RES-512-3=3-1=1-1=1-_FC-256_L-10',4,(3,32,32),0.0,'tdBN',0.5,0.0,5.0,'sigmoid',2.0,5,True,True)
    # print(model)
    # exit()
    out=model(torch.randn(2,3,224,224))