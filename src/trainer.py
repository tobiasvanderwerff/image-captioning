"""
Basic training loop. This code is meant to be generic and can be used to train different types of neural networks.
"""

import logging
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    batch_size = 128
    epochs = 10
    grad_norm_clip = 5.0
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
class Trainer:
    def __init__(self, config, model, optimizer, train_ds, eval_ds=None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    def train(self):
        model, optimizer, config = self.model, self.optimizer, self.config
        trainloader = DataLoader(self.train_ds, config.batch_size, shuffle=True, 
                                 num_workers=config.num_workers, pin_memory=True)
        if self.eval_ds is not None:
            evalloader = DataLoader(self.eval_ds, 2*config.batch_size, shuffle=False,  # double the batch size since evaluation takes less memory
                                    num_workers=config.num_workers, pin_memory=True)
    
        def run_epoch(split):
            is_train = True if split == 'train' else False
            dataloader = trainloader if split == 'train' else evalloader
            losses, total_samples, total_correct = [], 0, 0
            for data in dataloader:
                model.train(is_train)  # put model in training or evaluation mode

                data = [el.to(self.device) for el in data]  # put data on the appropriate device (cpu or gpu)

                optimizer.zero_grad()  # set the gradients to zero

                logits, loss, num_correct, num_samples = model(*data)  # forward pass
                
                losses.append(loss.item())
                total_correct += num_correct
                total_samples += num_samples
                
                if is_train:
                    loss.backward()  # calculate gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)  # clip gradients to avoid exploding gradients
                    optimizer.step()  # update weights
            epoch_loss = np.mean(losses)
            info_str = f"Epoch {ep} - {split}_loss: {epoch_loss:.4f}"
            if split == 'eval':
                accuracy = total_correct / total_samples
                info_str += f" - accuracy: {accuracy:.4f}"
            logger.info(info_str)

        for ep in range(config.epochs):
            run_epoch('train')
            if self.eval_ds is not None:
                with torch.no_grad():
                    run_epoch('eval')