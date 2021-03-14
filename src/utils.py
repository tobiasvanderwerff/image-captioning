from functools import partial

import torch
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_fn(batch):
    """ 
    This is a function which changes the batching function of the PyTorch data loader, 
    which would normally collate a sample of datapoints from the dataset into a mini-batch. This function
    customizes the default behavior by adding padding of the captions. It pads each caption to the length 
    of the longest caption in the batch. 
    """
    imgs, captions, targets = zip(*batch)
    captions = sorted(captions, reverse=True, key=lambda seq: len(seq))
    targets = sorted(targets, reverse=True, key=lambda seq: len(seq))
    seq_lengths = [len(seq) for seq in captions]
    max_seq_len = max(seq_lengths)

    # Apply padding. NOTE: the value of the padding token is assumed to be 0.
    captions_pad = np.zeros((len(captions), max_seq_len))
    targets_pad = np.zeros((len(captions), max_seq_len-1))
    for i, (seq, target) in enumerate(zip(captions, targets)):
        captions_pad[i, :seq_lengths[i]] = seq
        targets_pad[i, :seq_lengths[i]-1] = target
        
    # Convert the data to PyTorch tensors.
    to_long_tensor = partial(torch.tensor, dtype=torch.long)
    imgs = torch.stack(imgs, 0)
    captions_pad, seq_lengths, targets_pad = map(to_long_tensor, (captions_pad, seq_lengths, targets_pad))
    
    return imgs, captions_pad, seq_lengths, targets_pad