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
from src.utils import count_parameters

logger = logging.getLogger(__name__)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        logger.info(f"Number of trainable parameters: {count_parameters(self)}")
        
    def forward(self, img, caption, seq_lengths, targets=None):
        img_features = self.encoder(img)  # shape: (batch, img_features)
        img_features = img_features.repeat(self.decoder.num_layers, 1, 1)  # shape: (num_layers, batch, img_features)
        
        loss, num_correct = None, None
        if targets is not None:
            logits, _ = self.decoder(caption, img_features, seq_lengths) 
            import pdb; pdb.set_trace()
            loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
            num_correct = (torch.argmax(logits, dim=-1) == targets).sum().item()
        return logits, loss, num_correct, logits.size(0) * logits.size(1)

class LSTMDecoder(nn.Module):
    def __init__(self, num_hidden, embedding_dim, vocab_size, device, num_layers=2, bidirectional=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.device = device
        
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, num_hidden, num_layers=2, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(num_hidden, vocab_size)
    
    def forward(self, x, h0, seq_lengths):
        # x has shape (batch, seq_len), h0 has shape (num_layers * num_directions, batch, num_hidden)
        
        c0 = self.init_cell_state(h0.shape[1])
        
        emb = self.emb(x)  # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        
        # Pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # See https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        # We use seq_lengths - 1 because we do not want to count the last <END> token in each caption.
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lengths - 1, batch_first=True)
        
        out, (h, _) = self.lstm(emb, (h0, c0))

        # Undo the packing operation
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (batch, seq_len, num_directions * num_hidden)
        
        logits = self.fc(out)
        return logits, h
                     
    def init_cell_state(self, batch_size):
        c = torch.zeros(self.num_layers * self.num_directions, batch_size, self.num_hidden)
        c = c.to(self.device)
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
                    