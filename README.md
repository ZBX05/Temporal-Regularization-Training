# Temporal Regularization Training

Here is the PyTorch implementation for paper **Temporal Regularization Training: Unleashing the Potential of Spiking Neural Networks**.  

[arXiv](https://arxiv.org/abs/2506.19256)

## Requirements

- Python >= 3.11
- Pytorch >= 2.5.0
- CUDA >= 12.4
- SpikingJelly == 0.0.0.0.15

## Datasets

We conduct experiments on these datasets:

- [CIFAR10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet-100](https://www.image-net.org/challenges/LSVRC/2012/browse-synsets)
- [DVS-CIFAR10](https://figshare.com/s/d03a91081824536f12a8)
- [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101)

## Running

CIFAR10 (T=6):  

```bash
python main.py --dataset CIFAR10 --topology ResNet-19 --T 6 --trtloss 1 --loss_lambda 0.00001 --loss_decay 0.25 --loss_eta 0.05 --epochs 300 --batch_size 64
```

ImageNet-100 (T=4):

```bash
python main.py --dataset ImageNet100 --topology SEW-ResNet-34 --T 4 --tau 1.0 --trtloss 1 --loss_lambda 0.00005 --loss_decay 0.5 --loss_eta 0.001 --epochs 300 --batch_size 64
```

DVS-CIFAR10 (T=10 automatically):

```bash
python main.py --dataset DVSCIFAR10 --topology VGGSNN --trtloss 1 --loss_lambda 0.00005 --loss_decay 0.5 --loss_eta 0.001 --epochs 300 --batch_size 64
```

N-Caltech101 (T=10):

```bash
python main.py --dataset NCaltech101 --topology VGGSNN -T 10 --trtloss 1 --loss_lambda 0.00005 --loss_decay 0.5 --loss_eta 0.05 --epochs 300 --batch_size 64
```

## Abstract

Spiking Neural Networks (SNNs) have received widespread attention due to their event-driven and low-power characteristics, making them particularly effective for processing event-based neuromorphic data. Recent studies have shown that directly trained SNNs suffer from severe overfitting issues due to the limited scale of neuromorphic datasets and the gradient mismatching problem, which fundamentally constrain their generalization performance. In this paper, we propose a temporal regularization training (TRT) method by introducing a time-dependent regularization mechanism to enforce stronger constraints on early timesteps. We compare the performance of TRT with other state-of-the-art methods performance on datasets including CIFAR10/100, ImageNet100, DVS-CIFAR10, and N-Caltech101. To validate the effectiveness of TRT, we conducted ablation studies and analyses including loss landscape visualization and learning curve analysis, demonstrating that TRT can effectively mitigate overfitting and flatten the training loss landscape, thereby enhancing generalizability. Furthermore, we establish a theoretical interpretation of TRT’s temporal regularization mechanism based on the results of Fisher information analysis. We analyze the temporal information dynamics inside SNNs by tracking Fisher information during the TRT training process, revealing the Temporal Information Concentration (TIC) phenomenon, where Fisher information progressively concentrates in early timesteps. The time-decaying regularization mechanism implemented in TRT effectively guides the network to learn robust features in early timesteps with rich information, thereby leading to significant improvements in model generalization. Code is available at [https://github.com/ZBX05/Temporal-Regularization-Training](https://github.com/ZBX05/Temporal-Regularization-Training).

## Notice

We apply the following smooth function to the data of learning curves and accuracy curves before plotting.  

```python
def smooth(data_path:str, weight:int=0.5) -> list[np.float64]:
    data = pd.read_csv(filepath_or_buffer=data_path, header=0, names=['Step','Value'], dtype={'Step' : np.int64, 'Value' : np.float64})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
```

## Acknowledgements

Dataset N-Caltech101 is loaded via [SpikingJelly](https://github.com/fangwei123456/spikingjelly/tree/master), the code of Fisher information analysis is based on [https://github.com/Intelligent-Computing-Lab-Yale/Exploring-Temporal-Information-Dynamics-in-Spiking-Neural-Networks](https://github.com/Intelligent-Computing-Lab-Yale/Exploring-Temporal-Information-Dynamics-in-Spiking-Neural-Networks), and the loss landscape visualization is based on [https://github.com/tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape). Thanks for their great work!  
