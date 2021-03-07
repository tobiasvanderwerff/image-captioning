"""
Basic training loop. This code is meant to be generic and can be used to train different types of neural networks.
"""

import logging
import itertools

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
        evalloader = DataLoader(self.eval_ds, 2*config.batch_size, shuffle=False,  # double the batch size since evaluation takes less memory
                                num_workers=config.num_workers, pin_memory=True)
    
        for ep in range(config.epochs):
            running_loss = 0
            for data in trainloader:
                model.train()  # put model in training mode (rather than evaluation mode)
                
                data = [el.to(self.device) for el in data]  # put data on the appropriate device (cpu or gpu)

                optimizer.zero_grad()  # set the gradients to zero

                logits, loss = model(*data)

                loss.backward()  # calculate gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)  # clip gradients to avoid exploding gradients
                optimizer.step()  # update weights

                running_loss += loss.item()
            logger.info(f"Epoch {ep} - train_loss: {running_loss / len(trainloader):.4f}")
            
            # TODO: evaluate model on evaluation set. 