"""
Encoder-decoder classes/functions for image captioning.
The main idea is:
- calculate image features using CNN encoder
- feed calculated image features into the initial state of an LSTM language model, which makes 
  use of an attention mechanism in order to generate a caption for the image.
"""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, img, caption, batch_size, targets=None):
        img_features = self.encoder(img)
        c0 = self.encoder.init_cell_state(batch_size, self.device)
        logits = self.decoder(caption, img_features, c0, seq_lengths) 
        
        loss = None
        if targets is not None:  # calculate the loss
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

class LSTMDecoder(nn.Module):
    def __init__(self, num_hidden, embedding_dim, vocab_size, num_layers=2, bidirectional=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, num_hidden, num_layers=2, batch_first=True, dropout=0, bidirectional=bidirectional)
        self.fc = nn.Linear(num_hidden, vocab_size)
    
    def forward(self, x, h0, c0, seq_lengths):
        # x has shape (batch, seq_len), h0 and c0 have shape (num_layers * num_directions, batch, num_hidden)
        emb = self.emb(x)  # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        
        # Pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # See https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lengths, batch_first=True)
        
        out, _ = self.lstm(emb, h0, c0)
        
        # Undo the packing operation
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        logits = self.fc(out)
        return logits
                     
    def init_cell_state(self, batch_size, device):
        c = torch.zeros(self.num_layers * self.num_directions, batch_size, self.num_hidden)
        c = c.to(device)
        return c
     
def get_encoder(num_hidden):
    """ Returns a pretrained resnet34 image encoder. """
    encoder = models.resnet34(pretrained=True)
    in_features = encoder.fc.in_features
    encoder.fc = nn.Linear(in_features, num_hidden)  # ensure that the output of the encoder is of the right size 
    for p in encoder.parameters():  # freeze the weights
        p.requires_grad = False
    encoder.fc.weight.requires_grad = True  # only train the last linear layer of the encoder
    return encoder
                    