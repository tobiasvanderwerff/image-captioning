"""
Encoder-decoder classes/functions for image captioning.
The main idea is:
- calculate image features using CNN encoder
- feed calculated image features into the initial state of an LSTM language model, which makes 
  use of an attention mechanism in order to generate a caption for the image.
"""

import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models
from src.utils import count_parameters

logger = logging.getLogger(__name__)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, max_seq_len=30):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_seq_len = max_seq_len
        
        logger.info(f"Number of trainable parameters: {count_parameters(self)}")  # TODO: this is not right for the encoder
        
    def forward(self, imgs, split, captions=None, seq_lengths=None):
        # This follows the implementation in the 2015 paper 'Show and Tell' by Vinyals et al. We feed the image
        # through the encoder, after which the LSTM is initialized by running the image features through the LSTM
        # and using the consequent hidden and cell state to run the LSTM for generating words.
        
        if type(split) == list or type(split) == tuple:
            split = split[0] # TODO: fix this?
        assert split in ['train', 'eval', 'test']
        
        batch, *_ = imgs.shape
        max_seq_len = captions.size(1)
        loss, num_correct, sampled_ids = None, None, None
        
        img_features = self.encoder(imgs)  # img_features: (batch, num_hidden)
        if split == 'train':
            all_logits = self.decoder(captions, img_features, seq_lengths.cpu())
            loss = F.cross_entropy(all_logits.transpose(2, 1), captions, ignore_index=self.decoder.pad_token_idx)
        if split == 'eval' or split == 'test':
            all_logits, sampled_ids = [], []
            hiddens = None
            lstm_in = img_features.unsqueeze(1)
            for t in range(max_seq_len):
                lstm_out, hiddens = self.decoder.lstm(lstm_in, hiddens)
                logits = self.decoder.fc(lstm_out)
                _, sample = logits.max(-1)
                lstm_in = self.decoder.emb(sample)
                sampled_ids.append(sample.squeeze(1))
                all_logits.append(logits)
            sampled_ids = torch.stack(sampled_ids, 1)
            all_logits = torch.cat(all_logits, 1)
            if captions is not None:
                loss = F.cross_entropy(all_logits.transpose(2, 1), captions, ignore_index=self.decoder.pad_token_idx)
                # TODO: num_correct is no longer correct since padding values are also included here
        if captions is not None:
            num_correct = (all_logits.argmax(-1) == captions).sum().item()  
        return all_logits, loss, num_correct, batch * max_seq_len, sampled_ids
    
    def sample(self, imgs):
        # TODO
        pass
    

class LSTMDecoder(nn.Module):
    def __init__(self, num_hidden, embedding_dim, vocab_size, pad_token_idx,
                 start_token_idx, num_layers=2, bidirectional=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.pad_token_idx = pad_token_idx
        self.start_token_idx = start_token_idx
        self.num_directions = 2 if bidirectional else 1
        
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, num_hidden, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(num_hidden, vocab_size)
    
    def forward(self, captions, img_features, seq_lengths):
        embs = self.emb(captions)  # (batch, seq_len) -> (batch, seq_len, embedding_dim) 
        embs = torch.cat([img_features.unsqueeze(1), embs], 1)
        packed = pack_padded_sequence(embs, seq_lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)  # undo packing
        logits = self.fc(hiddens)
        return logits
                     
def get_encoder(num_hidden):
    """ Returns a pretrained resnet34 image encoder. """
    encoder = models.resnet34(pretrained=True)
    in_features = encoder.fc.in_features
    encoder.fc = nn.Linear(in_features, num_hidden)  # ensure that the output of the encoder is of the right size 
    for p in encoder.parameters():  # freeze the weights
        p.requires_grad = False
    encoder.fc.weight.requires_grad = True  # only train the last linear layer of the encoder
    encoder.fc.bias.requires_grad = True
    return encoder

class ResNetEncoder(nn.Module):
    """ Pretrained resnet34 image encoder """
    
    def __init__(self, num_hidden):
        super().__init__()
        resnet = models.resnet34(pretrained=True) 
        modules = list(resnet.children())
        in_features = modules[-1].in_features
        
        self.resnet = nn.Sequential(*modules[:-1])  # remove the last fc layer
        self.fc = nn.Linear(in_features, num_hidden, bias=False)
        self.bn = nn.BatchNorm1d(num_hidden)
    
    def forward(self, imgs):
        with torch.no_grad():  # do not keep track of resnet gradients, because we want to freeze those weights
            features = self.resnet(imgs)
        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        return features
        
