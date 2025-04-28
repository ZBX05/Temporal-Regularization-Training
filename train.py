import os
import torch
from torch.amp import GradScaler
from torch.optim import AdamW,Adam,SGD,RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from typing import Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import logging
from argparse import Namespace
from function import TRT_Loss,TET_Loss,FI_Observation

def train(args:Namespace,model:torch.nn.Module,train_data_loader:DataLoader,test_data_loader:DataLoader,device:torch.device,
          experiment_path:str) -> None:
    # first_str=[string for string in filter(lambda x:x!='',re.split(r'[.>\']+',str(type(model))))][-1]
    norm_str=args.norm+'_' if args.norm!='No' and 'CONV' in args.topology else ''
    topology=args.model if args.model!='Custom' else args.topology
    first_str='T'+str(args.T)+'_'+args.surrogate_type+f'_{norm_str}'+topology
    if args.regloss:
        first_str=f'REG({args.criterion})_'+first_str
    else:
        first_str=args.criterion+'_'+first_str
    if args.resume:
        checkpoint=torch.load(args.resume_path)
        epoch_resume=checkpoint["epoch"]
        first_str=f'Resume_{epoch_resume+1}_'+first_str
    time_str=time.strftime(r'%Y-%m-%d_%H-%M-%S')
    result_root_path=experiment_path+'/result/'+first_str+'_'+time_str
    result_logs_path=experiment_path+'/result/'+first_str+'_'+time_str+'/logs'
    result_weight_path=experiment_path+'/result/'+first_str+'_'+time_str+'/weights'
    if not os.path.exists(experiment_path+'/result/'+first_str+time_str):
        os.makedirs(result_root_path)
        os.makedirs(result_logs_path)
        os.makedirs(result_weight_path)
    
    logging.basicConfig(level=logging.INFO,filename=result_root_path+'/train.log',filemode='w')
    for arg in args._get_kwargs():
        logging.info(f'{arg[0]}={arg[1]}')
    logging.info(f'Model structure:\n{model}')

    model.to(device)
    writer=SummaryWriter(result_logs_path)
    with torch.no_grad():
        example_input,_=next(iter(train_data_loader))
        writer.add_graph(model,example_input.to(device))
    
    epochs=args.epochs

    if args.optimizer=='SGD':
            optimizer=SGD(model.parameters(),lr=args.lr,momentum=0.9,nesterov=False)
    elif args.optimizer=='AdamW':
            optimizer=AdamW(model.parameters(),lr=args.lr)
    elif args.optimizer=='Adam':
            optimizer=Adam(model.parameters(),lr=args.lr)
    elif args.optimizer=='RMSprop':
            optimizer=RMSprop(model.parameters(),lr=args.lr)
    else:
        raise NameError('Optimizer '+str(args.optimizer)+' not supported!')
    
    if args.criterion=='MSE':
        criterion=lambda output,labels:F.mse_loss(output,F.one_hot(labels,args.label_size).float())
    elif args.criterion=='BCE':
        criterion=F.binary_cross_entropy
    elif args.criterion=='CE':
        criterion=F.cross_entropy
    else:
        raise NameError('Loss '+str(args.loss)+' not supported!')
    
    if args.scheduler:
        scheduler=CosineAnnealingLR(optimizer,eta_min=0,T_max=epochs)
    else:
        scheduler=None
    
    amp=False
    if args.amp:
         amp=True
         scaler=GradScaler(device='cuda' if not args.cpu else 'cpu')
    
    reg_loss=args.regloss
    if reg_loss:
        loss_lambda=args.loss_lambda
        loss_decay=args.loss_decay
        loss_epsilon=args.loss_epsilon
        loss_eta=args.loss_eta
        # loss_means=args.loss_means
    
    tet_loss=args.tetloss
    if tet_loss:
        tet_lambda=args.loss_lambda
        tet_means=args.loss_means
    
    observe_fi=False
    if args.observe_fi:
        observe_fi=True
        fi_epochs=[int(tic_epoch) for tic_epoch in args.fi_epochs.split('-')]
    
    weight_decay=False
    if args.weight_decay is not None:
        weight_decay=True
        decay_dict=args.weight_decay
    
    if args.mean_reduce:
        get_backward_loss=lambda loss:loss.mean()
    else:
        get_backward_loss=lambda loss:loss
    
    save_checkpoint=False
    if args.save_checkpoint:
        save_checkpoint=True
        checkpoint_epochs=[int(epoch) for epoch in args.checkpoint_epochs.split('-')]

    if args.resume:
        # checkpoint=torch.load(args.resume_path)
        # epoch_resume=checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        loss=checkpoint["loss"]
        if amp:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        epoch_resume=0

    best_test_acc=0
    best_test_loss=0
    best_epoch=0
    train_acc_list=[]
    test_acc_list=[]
    train_loss_list=[]
    test_loss_list=[]
    for epoch in range(epoch_resume,epochs):
        model.train()
        train_loss=0
        train_acc=0
        train_samples=0
        for img,labels in tqdm(train_data_loader):
            img=img.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            if amp:
                with torch.amp.autocast(device_type='cuda' if not args.cpu else 'cpu'):
                    if reg_loss or tet_loss:
                        output=model(img,True)
                        if reg_loss:
                            loss=TRT_Loss(model,output,labels,criterion,loss_decay,loss_lambda,loss_epsilon,loss_eta)
                        elif tet_loss:
                            loss=TET_Loss(output,labels,criterion,tet_means,tet_lambda)
                        output=output.mean(1)
                    else:
                        output=model(img)
                        loss=criterion(output,labels)
                        if weight_decay:
                            if decay_dict["type"]=='l1':
                                norm=lambda x:torch.sum(torch.abs(x))
                            elif decay_dict["type"]=='l2':
                                norm=lambda x:torch.sum(x**2)
                            for name,param in model.named_parameters():
                                if 'weight' in name:
                                    loss+=decay_dict["decay"]*norm(param)
                    loss=get_backward_loss(loss)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if reg_loss or tet_loss:
                    output=model(img,True)
                    if reg_loss:
                        loss=TRT_Loss(model,output,labels,criterion,loss_decay,loss_lambda,loss_epsilon,loss_eta)
                    elif tet_loss:
                        loss=TET_Loss(output,labels,criterion,tet_means,tet_lambda)
                    output=output.mean(1)
                else:
                    output=model(img)
                    loss=criterion(output,labels)
                    if weight_decay:
                        if decay_dict["type"]=='l1':
                            norm=lambda x:torch.sum(torch.abs(x))
                        elif decay_dict["type"]=='l2':
                            norm=lambda x:torch.sum(x**2)
                        for name,param in model.named_parameters():
                            if 'weight' in name:
                                loss+=decay_dict["decay"]*norm(param)
                loss=get_backward_loss(loss)
                loss.backward()
                optimizer.step()
            train_samples+=labels.size(0)
            train_loss+=loss.item()*labels.size(0)

            train_acc+=(output.argmax(1)==labels).float().sum().item()
        test_loss,test_acc=evaluate(model,test_data_loader,criterion,device)
        if test_acc>best_test_acc or (test_acc==best_test_acc and test_loss<best_test_loss):
            best_test_acc=test_acc
            best_test_loss=test_loss
            best_epoch=epoch+1
            # torch.save(model.cpu().state_dict(),
            #            experiment_path+f'/result/{surrogate_type}_{epoch+1}_{test_loss}_{test_acc}.pth')
            # first_str=model.train_mode+'_'+[string for string in filter(lambda x:x!='',re.split(r'[.>\']+',str(type(model))))][-1]
            param=args.surrogate_param
            torch.save(model.cpu().state_dict(),
                        result_weight_path+f'/{first_str}_{args.surrogate_type}{param}_{epoch+1}_{test_loss}_{test_acc}.pth')
            model.to(device)
        if observe_fi and epoch+1 in fi_epochs:
            torch.save(model.cpu().state_dict(),
                       result_weight_path+f'/FI_{first_str}_{epoch+1}.pth')
            model.to(device)
        if save_checkpoint and epoch+1 in checkpoint_epochs:
            torch.save({
                "epoch":epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "scheduler_state_dict":scheduler.state_dict() if scheduler is not None else None,
                "loss":loss,
                "scaler":scaler.state_dict() if amp else None
            },result_weight_path+f'/checkpoint_{first_str}_{epoch+1}.pth')
        if scheduler is not None:
            scheduler.step()
        train_loss/=train_samples
        train_acc/=train_samples
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        tags=['train_loss','train_acc','test_loss','test_acc']
        writer.add_scalar(tags[0],train_loss,epoch+1)
        writer.add_scalar(tags[1],train_acc,epoch+1)
        writer.add_scalar(tags[2],test_loss,epoch+1)
        writer.add_scalar(tags[3],test_acc,epoch+1)
        print('Epoch {:3d}: Train loss: {:4f} | Train acc: {:3f} | Test loss: {:4f} | Test acc: {:3f}'
              .format(epoch+1,train_loss,train_acc,test_loss,test_acc))
        logging.info('Epoch {:3d}: Train loss: {:4f} | Train acc: {:3f} | Test loss: {:4f} | Test acc: {:3f}'
              .format(epoch+1,train_loss,train_acc,test_loss,test_acc))
    print(f'Best test accuracy: {best_test_acc} at epoch {best_epoch}')
    logging.info(f'Best test accuracy: {best_test_acc} at epoch {best_epoch}')
    # pd.DataFrame({"train_loss":train_loss_list,"test_loss":test_loss_list,"train_acc":train_acc_list,"test_acc":test_acc_list}).to_csv(
    #     experiment_path+'/result/curve.csv')
    if observe_fi:
        for epoch in fi_epochs:
            model.load_state_dict(torch.load(result_weight_path+f'/FI_{first_str}_{epoch}.pth'))
            FI_Observation(model,train_data_loader,epoch,args.T,device,logging,writer)
    writer.close()

def evaluate(model:torch.nn.Module,test_data_loader:DataLoader,criterion:Any,device:torch.device) -> tuple:
    model.eval()
    test_loss=0
    test_acc=0
    test_samples=0
    model.to(device)
    with torch.no_grad():
        for img,labels in tqdm(test_data_loader):
            img=img.to(device)
            labels=labels.to(device)

            output=model(img)
            loss=criterion(output,labels)

            test_samples+=labels.size(0)
            test_loss+=loss.item()*labels.size(0)

            test_acc+=(output.argmax(1)==labels).float().sum().item()
    
    return test_loss/test_samples,test_acc/test_samples
