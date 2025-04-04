import torch
from neuron import *
import torch.nn as nn

class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly.
    # This container enables seq input to layers contained in it(e.g. [B,T,C,H,W] -> nn.Conv2d).
    def __init__(self,*args:nn.Module) -> None:
        super(SeqToANNContainer,self).__init__()
        if len(args)==1:
            self.module=args[0]
        else:
            self.module=nn.Sequential(*args)

    def forward(self,x_seq:torch.Tensor) -> torch.Tensor:
        y_shape=[x_seq.shape[0],x_seq.shape[1]]
        y_seq=self.module(x_seq.flatten(0,1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class TimeExpensionBlock(nn.Module):
    def __init__(self,T:int,dimension:int) -> None:
        super(TimeExpensionBlock,self).__init__()
        self.T=T
        self.dimension=dimension
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        return input.unsqueeze(self.dimension).repeat_interleave(self.T,self.dimension)

class BaseBlock(nn.Module):
    def __init__(self) -> None:
        super(BaseBlock,self).__init__()

    def reset(self) -> None:
        if hasattr(self,'lif'):
            self.lif.reset()

class BatchNorm2dBlock(BaseBlock):
    def __init__(self,num_features:int) -> None:
        super(BatchNorm2dBlock,self).__init__()
        self.bn=nn.BatchNorm2d(num_features)

    def forward(self,input:torch.Tensor) -> torch.Tensor:
        T=input.shape[1]
        output_pot=[]
        for t in range(T):
            output_pot.append(self.bn(input[:,t,...]))
        return torch.stack(output_pot,dim=1)

class FlattenBlock(BaseBlock):
    def __init__(self,start_dim:int=1,end_dim:int=-1) -> None:
        super(FlattenBlock,self).__init__()
        self.flatten=nn.Flatten(start_dim=start_dim,end_dim=end_dim)
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        return self.flatten(input)

class FCBlock(BaseBlock):
    def __init__(self,input_size:int,output_size:int,bias:bool,dropout:float,v_threshold:float=1.0,v_reset:float=0.0,tau:float=5,
                 surrogate_type:str='sigmoid',surrogate_param:float=2.0,surrogate_m:int=5) -> None:
        super(FCBlock,self).__init__()
        self.fc=SeqToANNContainer(
            nn.Linear(input_size,output_size,bias=bias)
        )
        self.lif=LIFNode(v_threshold=v_threshold,v_reset=v_reset,tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param,
                         surrogate_m=surrogate_m)
        if dropout!=0:
            self.dropout=nn.Dropout(p=dropout)
        else:
            self.dropout=None
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        x=self.fc(input)
        if self.dropout is not None:
            x=self.dropout(x)
        output=self.lif(x)
        return output

class Conv2dBlock(BaseBlock):
    def __init__(self,norm:str,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:int,conv_stride:int,conv_padding:int,
                 pool_kernel_size:int,pool_stride:int,pool_padding:int,bias:bool,v_threshold:float=1.0,v_reset:float=0.0,tau:float=5,
                 surrogate_type:str='sigmoid',surrogate_param:float=2.0,surrogate_m:int=5) -> None:
        super(Conv2dBlock,self).__init__()
        self.out_channels=out_channels
        self.conv_kernel_size=conv_kernel_size
        self.conv_stride=conv_stride
        self.conv_padding=conv_padding
        self.pool_kernel_size=pool_kernel_size
        self.pool_stride=pool_stride
        self.pool_padding=pool_padding
        self.norm=norm

        if self.norm=='tdBN':
            self.conv2d_layer=SeqToANNContainer(
                nn.Conv2d(in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,bias=bias),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv2d_layer=SeqToANNContainer(
                nn.Conv2d(in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,bias=bias)
            )
        self.maxpool2d_layer=SeqToANNContainer(
            nn.MaxPool2d(pool_kernel_size,pool_stride,pool_padding)
        )
        if len(input_shape)==2:
            input_height=input_shape[0]
            input_width=input_shape[1]
        else:
            input_height=input_shape[1]
            input_width=input_shape[2]
        self.conv2d_output_height=self.get_conv2d_output_dim(input_height)
        self.conv2d_output_width=self.get_conv2d_output_dim(input_width)
        self.output_height,self.output_width=self.get_conv2d_block_output_dim(input_height,input_width)
        self.lif=LIFNode(v_threshold=v_threshold,v_reset=v_reset,tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param,
                         surrogate_m=surrogate_m)
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        x=self.conv2d_layer(input)
        x=self.lif(x)
        output=self.maxpool2d_layer(x)
        return output
        
    def get_conv2d_output_dim(self,dim_size:int) -> int:
        padding=self.conv_padding
        kernel_size=self.conv_kernel_size
        stride=self.conv_stride
        output_dim=(dim_size+2*padding-kernel_size)//stride+1
        return output_dim

    def get_maxpool2d_output_dim(self,dim_size:int) -> int:
        kernel_size=self.pool_kernel_size
        padding=self.pool_padding
        stride=self.pool_stride
        output_dim=(dim_size+2*padding-kernel_size)//stride+1
        return output_dim

    def get_conv2d_block_output_dim(self,input_height:int,input_width:int) -> tuple[int,int]:
        output_height=self.get_maxpool2d_output_dim(self.get_conv2d_output_dim(input_height))
        output_width=self.get_maxpool2d_output_dim(self.get_conv2d_output_dim(input_width))
        return output_height,output_width

class Conv2dAPBlock(Conv2dBlock):
    def __init__(self,norm:str,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:int,conv_stride:int,conv_padding:int,
                 pool_kernel_size:int,pool_stride:int,pool_padding:int,bias:bool,v_threshold:float=1.0,v_reset:float=0.0,tau:float=5,
                 surrogate_type:str='sigmoid',surrogate_param:float=2.0,surrogate_m:int=5) -> None:
        super(Conv2dAPBlock,self).__init__(norm,input_shape,in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,
                                                  pool_kernel_size,pool_stride,pool_padding,bias,v_threshold,v_reset,tau,surrogate_type,
                                                  surrogate_param,surrogate_m)
        self.maxpool2d_layer=SeqToANNContainer(
            nn.AvgPool2d(pool_kernel_size,pool_stride,pool_padding)
        )

class Conv2dNoPoolingBlock(Conv2dBlock):
    def __init__(self,norm:str,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:int,conv_stride:int,conv_padding:int,
                 bias:bool,v_threshold:float=1.0,v_reset:float=0.0,tau:float=5,surrogate_type:str='sigmoid',
                 surrogate_param:float=2.0,surrogate_m:int=5) -> None:
        super(Conv2dNoPoolingBlock,self).__init__(norm,input_shape,in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,
                                                  2,2,0,bias,v_threshold,v_reset,tau,surrogate_type,surrogate_param,surrogate_m)
        self.maxpool2d_layer=None
        
        if len(input_shape)==2:
            input_height=input_shape[0]
            input_width=input_shape[1]
        else:
            input_height=input_shape[1]
            input_width=input_shape[2]
        self.output_height=self.get_conv2d_output_dim(input_height)
        self.output_width=self.get_conv2d_output_dim(input_width)
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        x=self.conv2d_layer(input)
        output=self.lif(x)
        return output

class Conv2dEncoderBlock(Conv2dBlock):
    def __init__(self,T:int,norm:str,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:int,conv_stride:int,conv_padding:int,
                 pool_kernel_size:int,pool_stride:int,pool_padding:int,bias:bool,v_threshold:float=1.0,v_reset:float=0.0,tau:float=5,
                 surrogate_type:str='sigmoid',surrogate_param:float=2.0,surrogate_m:int=5) -> None:
        super(Conv2dEncoderBlock,self).__init__(norm,input_shape,in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,
                                                pool_kernel_size,pool_stride,pool_padding,bias,v_threshold,v_reset,tau,surrogate_type,
                                                surrogate_param,surrogate_m)
        self.time_expension_layer=TimeExpensionBlock(T,1)
        if self.norm=='tdBN':
            self.conv2d_layer=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,bias=bias),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv2d_layer=nn.Conv2d(in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,bias=bias)
        self.maxpool2d_layer=SeqToANNContainer(
            nn.MaxPool2d(pool_kernel_size,pool_stride,pool_padding)
        )

    def forward(self,input:torch.Tensor) -> torch.Tensor:
        x=self.conv2d_layer(input)
        x=self.time_expension_layer(x)
        x=self.lif(x)
        output=self.maxpool2d_layer(x)
        return output

class ResidualBlock(BaseBlock):
    def __init__(self,norm:str,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:tuple,conv_stride:tuple,conv_padding:tuple,
                 pool:bool,downsample:bool,bias:bool,v_threshold:float=1.0,v_reset:float=0.0,tau:float=5,surrogate_type:str='sigmoid',
                 surrogate_param:float=2.0,surrogate_m:int=5) -> None:
        super(ResidualBlock,self).__init__()
        self.norm=norm
        self.pool=pool
        self.downsample=downsample
        self.out_channels=out_channels
        conv_1=nn.Conv2d(in_channels,out_channels,conv_kernel_size[0],conv_stride[0],conv_padding[0],bias=bias)
        conv_2=nn.Conv2d(out_channels,out_channels,conv_kernel_size[1],conv_stride[1],conv_padding[1],bias=bias)
        self.conv_kernel_size=conv_kernel_size
        self.conv_stride=conv_stride
        self.conv_padding=conv_padding
        if len(input_shape)==2:
            input_height=input_shape[0]
            input_width=input_shape[1]
        else:
            input_height=input_shape[1]
            input_width=input_shape[2]
        self.output_height,self.output_width=(1,1) if self.pool else self.get_residual_block_output_dim(input_height,input_width)
        if self.norm=='tdBN':
            self.conv2d_layer_1=SeqToANNContainer(
                conv_1,
                nn.BatchNorm2d(out_channels)
            )
            self.conv2d_layer_2=SeqToANNContainer(
                conv_2,
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv2d_layer_1=SeqToANNContainer(
                conv_1
            )
            self.conv2d_layer_2=SeqToANNContainer(
                conv_2
            )
        if self.downsample and self.norm=='tdBN':
            self.downsample_layer=SeqToANNContainer(
                nn.Conv2d(in_channels,out_channels,1,conv_stride[0],0,bias=bias),
                nn.BatchNorm2d(out_channels)
            )
        elif self.downsample and self.norm!='tdBN':
            self.downsample_layer=SeqToANNContainer(
                nn.Conv2d(in_channels,out_channels,1,conv_stride[0],0,bias=bias)
            )
        if self.pool:
            self.adaptive_avgpool_layer=SeqToANNContainer(
                nn.AdaptiveAvgPool2d((1,1))
            )
        self.lif_1=LIFNode(v_threshold=v_threshold,v_reset=v_reset,tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param,
                           surrogate_m=surrogate_m)
        self.lif_2=LIFNode(v_threshold=v_threshold,v_reset=v_reset,tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param,
                           surrogate_m=surrogate_m)
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        identity=input
        x=self.conv2d_layer_1(input)
        x=self.lif_1(x)
        x=self.conv2d_layer_2(x)
        if self.downsample:
            identity=self.downsample_layer(input)
        x+=identity
        output=self.lif_2(x)
        if self.pool:
            output=self.adaptive_avgpool_layer(output)
        return output

    def get_conv2d_output_dim(self,kernel_size:int,stride:int,padding:int,dim_size:int) -> int:
        padding=padding
        kernel_size=kernel_size
        stride=stride
        output_dim=(dim_size+2*padding-kernel_size)//stride+1
        return output_dim
    
    def get_residual_block_output_dim(self,input_height:int,input_width:int) -> tuple[int,int]:
        padding=self.conv_padding[0]
        kernel_size=self.conv_kernel_size[0]
        stride=self.conv_stride[0]
        input_height=(input_height+2*padding-kernel_size)//stride+1
        input_width=(input_width+2*padding-kernel_size)//stride+1
        padding=self.conv_padding[1]
        kernel_size=self.conv_kernel_size[1]
        stride=self.conv_stride[1]
        output_height=(input_height+2*padding-kernel_size)//stride+1
        output_width=(input_width+2*padding-kernel_size)//stride+1
        return output_height,output_width

class SEWResidualBlock(ResidualBlock):
    def __init__(self,connect_function:str,norm:str,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:tuple,conv_stride:tuple,
                 conv_padding:tuple,pool:bool,downsample:bool,bias:bool,v_threshold:float=1.0,v_reset:float=0.0,tau:float=5,
                 surrogate_type:str='sigmoid',surrogate_param:float=2.0,surrogate_m:int=5) -> None:
        super(SEWResidualBlock,self).__init__(norm,input_shape,in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,pool,
                                              downsample,bias,v_threshold,v_reset,tau,surrogate_type,surrogate_param,surrogate_m)
        if connect_function=='ADD':
            self.connect_function=lambda x,identity:x+identity
        elif connect_function=='AND':
            self.connect_function=lambda x,identity:x*identity
        elif connect_function=='IAND':
            self.connect_function=lambda x,identity:identity*(1.0-x)
        
        if self.downsample and self.norm=='tdBN':
            self.downsample_layer=nn.Sequential(
                SeqToANNContainer(
                    nn.Conv2d(in_channels,out_channels,1,conv_stride[0],0,bias=bias),
                    nn.BatchNorm2d(out_channels)
                ),
                LIFNode(v_threshold=v_threshold,v_reset=v_reset,tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param,
                        surrogate_m=surrogate_m)
            )
        elif self.downsample and self.norm!='tdBN':
            self.downsample_layer=nn.Sequential(
                SeqToANNContainer(
                    nn.Conv2d(in_channels,out_channels,1,conv_stride[0],0,bias=bias)
                ),
                LIFNode(v_threshold=v_threshold,v_reset=v_reset,tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param,
                        surrogate_m=surrogate_m)
            )
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        identity=input
        x=self.conv2d_layer_1(input)
        x=self.lif_1(x)
        x=self.conv2d_layer_2(x)
        if self.downsample:
            identity=self.downsample_layer(input)
        x+=identity
        output=self.lif_2(x)
        if self.pool:
            output=self.adaptive_avgpool_layer(output)
        return output

class MaxPool2dBlock(nn.Module):
    def __init__(self,kernel_size:int,stride:int,padding:int=0) -> None:
        super(MaxPool2dBlock,self).__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.maxpool2d_layer=SeqToANNContainer(nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding))
    
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        return self.maxpool2d_layer(X)

class AveragePool2dBlock(nn.Module):
    def __init__(self,kernel_size:int,stride:int,padding:int=0) -> None:
        super(AveragePool2dBlock,self).__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.averagepool2d_layer=SeqToANNContainer(nn.AvgPool2d(kernel_size=kernel_size,stride=stride,padding=padding))
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.averagepool2d_layer(X)