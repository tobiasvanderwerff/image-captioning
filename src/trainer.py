"""
Basic training loop. This code is meant to be generic and can be used to train different types of neural networks.
"""

import logging
import itertools

import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils import plot_grad_flow

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    batch_size = 128
    epochs = 10
    grad_norm_clip = 5.0
    num_workers = 0
    track_loss = False
    track_grad_norm = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
class Trainer:
    def __init__(self, config, model, optimizer, train_ds, eval_ds=None, train_collate_fn=None,
                 eval_collate_fn = None, evaluation_callback_fn=None, metrics_callback_fn=None, 
                 max_epochs_no_change=5):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn
        self.evaluation_callback_fn = evaluation_callback_fn
        self.metrics_callback_fn = metrics_callback_fn
        self.max_epochs_no_change = max_epochs_no_change
        
        self.grad_norms = []    
        self.best_score = 0
        self.epochs_no_change = 0
        self.losses = {'train': [], 'eval': [], 'test': []}
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    def train(self):
        model, optimizer, config = self.model, self.optimizer, self.config
        trainloader = DataLoader(self.train_ds, config.batch_size, shuffle=True, 
                                 num_workers=config.num_workers, pin_memory=True,
                                 collate_fn=self.train_collate_fn)
        if self.eval_ds is not None:
            evalloader = DataLoader(self.eval_ds, 2*config.batch_size, shuffle=False,  # double the batch size since evaluation takes less memory
                                    num_workers=config.num_workers, pin_memory=True,
                                    collate_fn=self.eval_collate_fn)
    
        def run_epoch(split):
            is_train = True if split == 'train' else False
            dataloader = trainloader if split == 'train' else evalloader
            losses, scores = [], []
            for data in dataloader:
                model.train(is_train)  # put model in training or evaluation mode

                # put data on the appropriate device (cpu or gpu)
                data = [el.to(self.device) if hasattr(el, 'device') else el for el in data]  

                optimizer.zero_grad()  # set the gradients to zero

                if self.metrics_callback_fn is not None:
                    data.append(self.metrics_callback_fn)

                logits, loss, score, *_ = model(*data)  # forward pass
                
                if loss is not None:
                    losses.append(loss.item())
                    if config.track_loss:
                        self.losses[split].append(loss.item())
                        
                if score is not None:
                    if isinstance(score, list):
                        scores.extend(score)
                    else:
                        scores.append(score)
                
                if is_train:
                    loss.backward()  # calculate gradients
                    if config.track_grad_norm:
                        self._track_grad_norm()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)  # clip gradients to avoid exploding gradients
                    optimizer.step()  # update weights
            info_str = ""
            if loss is not None:
                epoch_loss = np.mean(losses)
                info_str += (f"Epoch {ep} - {split}_loss: {epoch_loss:.4f}")
            if scores != []:
                epoch_score = np.mean(scores)
                if epoch_score > self.best_score:
                    self.best_score = epoch_score
                    self.epochs_no_change = 0
                else:
                    self.epochs_no_change += 1
                all_scores.append(epoch_score)
                info_str += f"score: {epoch_score:.1f}"
            if info_str != "":
                logger.info(info_str)

        all_scores = []
        for ep in range(config.epochs):
            if self.epochs_no_change >= self.max_epochs_no_change:  # stop early
                # TODO: save best configuration when the score was best
                logger.info(f"Stopped early at epoch {ep}.")
                break
            run_epoch('train')
#             plot_grad_flow(model.named_parameters())
            if self.eval_ds is not None:
                with torch.no_grad():
                    run_epoch('eval')
                    if self.evaluation_callback_fn is not None:  # this can be used to show intermediate predictions of the model
                        self.evaluation_callback_fn(model, self.eval_ds)
                    
    @torch.no_grad()
    def _track_grad_norm(self):
        """ Calculate and store the 2-norm of the gradients in the model. """
        grad_norm = np.sqrt(np.sum(LA.norm(p.grad.cpu().numpy()) ** 2 for p in self.model.parameters() if p.requires_grad))
        self.grad_norms.append(grad_norm)