"""
Basic training loop. This code is meant to be generic and can be used to train different types of neural networks.
"""

import logging
import itertools
from pathlib import Path

import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils import plot_grad_flow
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    batch_size = 128
    epochs = 10
    grad_norm_clip = 5.0
    num_workers = 0
    track_loss = False
    track_grad_norm = False
    checkpoint_path = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
class Trainer:
    def __init__(self, config, model, optimizer, train_ds, eval_ds=None, test_ds=None, train_collate_fn=None,
                 eval_collate_fn = None, evaluation_callback_fn=None, metrics_callback_fn=None, 
                 max_epochs_no_change=10):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.test_ds = test_ds
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn
        self.evaluation_callback_fn = evaluation_callback_fn
        self.metrics_callback_fn = metrics_callback_fn
        self.max_epochs_no_change = max_epochs_no_change
        
        self.grad_norms = []    
        self.best_score = float('-inf')
        self.best_state_dict = None
        self.epoch = 0
        self.epochs_no_change = 0
        self.losses = {'train': [], 'eval': [], 'test': []}
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                    
    def save_checkpoint(self, score, **kwargs):
        assert Path(self.config.checkpoint_path).exists(), 'checkpoint path does not exist'
        cpt = {'model_state_dict': self.model.state_dict(), 
               'optimizer_state_dict': self.optimizer.state_dict(),
               'score': score}
        cpt.update(**kwargs)
        torch.save(cpt, Path(self.config.checkpoint_path) / f'epoch{self.epoch}_{score}')
                            
    @torch.no_grad()
    def _track_grad_norm(self):
        """ Calculate and store the 2-norm of the gradients in the model. """
        grad_norm = np.sqrt(np.sum(LA.norm(p.grad.cpu().numpy()) ** 2 for p in self.model.parameters() if p.requires_grad))
        self.grad_norms.append(grad_norm)
        
    def train(self):
        model, optimizer, config = self.model, self.optimizer, self.config
        trainloader = DataLoader(self.train_ds, config.batch_size, shuffle=True, 
                                 num_workers=config.num_workers, pin_memory=True,
                                 collate_fn=self.train_collate_fn)
        if self.eval_ds is not None:
            evalloader = DataLoader(self.eval_ds, 2*config.batch_size, shuffle=False,  # double the batch size since evaluation takes less memory
                                    num_workers=config.num_workers, pin_memory=True,
                                    collate_fn=self.eval_collate_fn)
        if self.test_ds is not None:
            testloader = DataLoader(self.test_ds, 2*config.batch_size, shuffle=False,  # double the batch size since evaluation takes less memory
                                    num_workers=config.num_workers, pin_memory=True,
                                    collate_fn=self.eval_collate_fn)
            
    
        def run_epoch(split):
            is_train = True if split == 'train' else False
            losses, scores = [], {}
            
            if split == 'train':
                dataloader = trainloader
            elif split == 'eval':
                dataloader = evalloader
            else:
                dataloader = testloader
            
            pbar = tqdm(dataloader, total=len(dataloader)) if is_train else dataloader
            for data in pbar:
                model.train(is_train)  # put model in training or evaluation mode

                # put data on the appropriate device (cpu or gpu)
                data = [el.to(self.device) if hasattr(el, 'device') else el for el in data]  

                if self.metrics_callback_fn is not None:
                    data.append(self.metrics_callback_fn)

                logits, loss, score, *_ = model(*data)  # forward pass
                
                if loss is not None:
                    losses.append(loss.item())
                    if config.track_loss:
                        self.losses[split].append(loss.item())
                        
                if score is not None:
                    for metric, value in score.items():
                        if metric in scores:
                            scores[metric].extend(value)
                        else:
                            scores.update({metric: value})
                        
                if is_train:
                    loss.backward()  # calculate gradients
                    if config.track_grad_norm:
                        self._track_grad_norm()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)  # clip gradients to avoid exploding gradients
                    optimizer.step()  # update weights
                    optimizer.zero_grad()  # set the gradients back to zero
            info_str = ""
            if loss is not None:
                epoch_loss = np.mean(losses)
                info_str += (f"epoch {ep} - {split}_loss: {epoch_loss:.4f}")
            if scores != {}:
                for metric, values in scores.items():
                    logger.info(f"{metric}: {np.mean(values):.1f}")
                eval_score = scores['BLEU-2']  # TODO: this should be application independent, change this
                epoch_score = np.mean(eval_score)
                if epoch_score > self.best_score:
                    logger.info(f"New best score: {epoch_score:1f}")
                    self.best_score = epoch_score
                    self.best_state_dict = model.state_dict()
                    self.epochs_no_change = 0
                    if self.config.checkpoint_path is not None:  # save the new best model
                        logger.info("Saving checkpoint.")
                        self.save_checkpoint(epoch_score, epoch=self.epoch)
                else:
                    self.epochs_no_change += 1
            if info_str != "":
                logger.info(info_str)

        for ep in range(config.epochs):
            self.epoch = ep
            run_epoch('train')
#             plot_grad_flow(model.named_parameters())
            if self.eval_ds is not None:
                with torch.no_grad():
                    run_epoch('eval')
                    if self.evaluation_callback_fn is not None:  # this can be used to show intermediate predictions of the model
                        self.evaluation_callback_fn(model, self.eval_ds)
#             if self.epochs_no_change >= self.max_epochs_no_change:  # stop early
            if self.epochs_no_change >= 1:  # stop early
                logger.info(f"Stopped early at epoch {ep}. Best score: {self.best_score}")
                if self.test_ds is not None:
                    logger.info("Calculating results on test set...")
                    model.load_state_dict(self.best_state_dict)
                    with torch.no_grad():
                        run_epoch('test')
                break