from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_fn(batch):
    """ 
    This is a function which changes the batching function of the PyTorch data loader, 
    which would normally collate a sample of datapoints from the dataset into a mini-batch. This function
    customizes the default behavior by adding padding of the captions. It pads each caption to the length 
    of the longest caption in the batch. 
    """
    imgs, captions, split = zip(*batch)
    ordering, _ = zip(*sorted(enumerate(captions), reverse=True, key=lambda it: len(it[1])))
    captions = [captions[i] for i in ordering]
    imgs = [imgs[i] for i in ordering]
    seq_lengths = [len(seq) for seq in captions]
    max_seq_len = max(seq_lengths)

    # Apply padding. NOTE: the value of the padding token is assumed to be 0.
    captions_pad = np.zeros((len(captions), max_seq_len))
    for i, seq in enumerate(captions):
        captions_pad[i, :seq_lengths[i]] = seq
        
    # Convert the data to PyTorch tensors.
    to_long_tensor = partial(torch.tensor, dtype=torch.long)
    imgs = torch.stack(imgs, 0)
    captions_pad, seq_lengths = map(to_long_tensor, (captions_pad, seq_lengths))
    
#     return imgs, captions_pad, seq_lengths
    return imgs, split, captions_pad, seq_lengths


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    
    Taken from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8 '''

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and ("encoder.resnet" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0 ,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()