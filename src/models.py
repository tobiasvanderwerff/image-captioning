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


class MLP(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.num_hidden = num_hidden
        
        self.lin1 = nn.Linear(num_hidden, num_hidden)
        self.relu1 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.relu2 = nn.ReLU(inplace=True)
       
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, max_seq_len=30):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_seq_len = max_seq_len

        logger.info(f"Number of trainable parameters: {count_parameters(self)}")  # TODO: this is not right for the encoder
        
    def forward(self, imgs, split, captions=None, seq_lengths=None, metrics_callback_fn=None):
        if type(split) == list or type(split) == tuple:
            split = split[0] # TODO: fix this?
        assert split in ['train', 'eval', 'test']
        
        loss, sampled_ids, scores = None, None, None
        # TODO: is split redundant and can the same information be obtained by looking at captions variable?
        max_seq_len = self.max_seq_len if split == 'test' else captions.size(1)
        
        feature_map, h0, c0 = self.encoder(imgs)
        h0, c0 = self.init_decoder_cell_state_and_hidden_state(h0, c0)

        if captions is not None: 
            targets = captions[:, 1:]  # the model skips the <START> token when making predictions
            
        if split == 'train':
            # all_logits = self.decoder(captions, feature_map, seq_lengths.cpu(), h0, c0)
            all_logits, sampled_ids = self.decoder(feature_map, h0, c0,
                                                   max_seq_len, captions=captions,
                                                   use_teacher_forcing=True)
            loss = F.cross_entropy(all_logits.transpose(2, 1), targets, ignore_index=0)
        elif split == 'eval' or split == 'test':
            all_logits, sampled_ids = self.decoder(feature_map, h0, c0, max_seq_len)
            if captions is not None:
                scores = {}
                for n in range(1, 5):
                    # calculate BLEU scores
                    bleu_score = metrics_callback_fn(sampled_ids, targets, max_n=n)
                    scores.update({f'BLEU-{n}': bleu_score})
        return all_logits, loss, scores, sampled_ids
    
    def sample(self, imgs):
        # TODO. Although, perhaps forward is sufficient for sampling...
        pass

    def init_decoder_cell_state_and_hidden_state(self, h0, c0):
        c0 = c0.repeat(self.decoder.num_layers * self.decoder.num_directions, 1, 1)
        h0 = h0.repeat(self.decoder.num_layers * self.decoder.num_directions, 1, 1)
        c0 = c0.to(self.device)
        h0 = h0.to(self.device)
        return h0, c0
    

class LSTMDecoder(nn.Module):
    def __init__(self, num_hidden, embedding_dim, vocab_size, annotation_dim, 
                 start_token_idx, device, num_layers=2, bidirectional=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.annotation_dim = annotation_dim
        self.start_token_idx = start_token_idx
        self.num_directions = 2 if bidirectional else 1
        self.device = device
        
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, num_hidden, num_layers, batch_first=True, bidirectional=bidirectional)
        self.attn = nn.Linear(num_hidden + annotation_dim, 1)  # TODO: make this an MLP?
        self.fc_h = nn.Linear(num_hidden, embedding_dim)
        self.fc_ctx = nn.Linear(annotation_dim, embedding_dim)
        self.fc_clsf = nn.Linear(embedding_dim, vocab_size)
        # self.drop = nn.Dropout(p=0.1)
    
    def forward(self, feature_map, h0, c0, max_seq_len, captions=None, use_teacher_forcing=False):
        assert ((captions is None and not use_teacher_forcing) or 
                (captions is not None and use_teacher_forcing))
        all_logits, sampled_ids = [], []
        batch_size, n_annotations, _ = feature_map.shape

        hc = (h0, c0)
        for t in range(max_seq_len - 1):
            if use_teacher_forcing:
                lstm_in = captions[:, t]
            else: 
                if sampled_ids == []:
                    lstm_in = torch.full([batch_size], self.start_token_idx)
                else:
                    lstm_in = sample
            word_emb = self.emb(lstm_in)
            hiddens, hc = self.lstm(word_emb.unsqueeze(1), hc)
            hiddens.squeeze_(1)  # hiddens: (batch_size, num_hidden)

            # Attention 
            attn_in = torch.cat([feature_map, hiddens.unsqueeze(1).repeat(1, n_annotations, 1)], 2)
            attn_out = self.attn(attn_in).squeeze(-1)  # (batch_size, n_annotations)
            attn_weights = F.softmax(attn_out, dim=-1)
         
            ctx = torch.sum(attn_weights.unsqueeze(-1) * feature_map, dim=1, keepdim=False)
            logits = self.fc_clsf(word_emb + self.fc_h(hiddens) + self.fc_ctx(ctx)) 

            _, sample = logits.max(-1)
            sampled_ids.append(sample)
            all_logits.append(logits)
        sampled_ids = torch.stack(sampled_ids, 1)
        all_logits = torch.stack(all_logits, 1)

        return all_logits, sampled_ids


class ResNetEncoder(nn.Module):
    """ Pretrained resnet50 image encoder """
    
    def __init__(self, num_hidden):
        super().__init__()
        resnet = models.resnet50(pretrained=True) 
        modules = list(resnet.children())
        self.annotation_dim = modules[-1].in_features
        
        # TODO: perhaps change fc_init_h and fc_init_c to multi-layer MLPs
        self.resnet = nn.Sequential(*modules[:-2])
        self.fc_init_h = nn.Linear(self.annotation_dim, num_hidden, bias=False)
        self.fc_init_c = nn.Linear(self.annotation_dim, num_hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.bn2 = nn.BatchNorm1d(num_hidden)
    
    def forward(self, imgs):
        with torch.no_grad():  # do not keep track of resnet gradients, because we want to freeze those weights
            feature_map = self.resnet(imgs).flatten(2, -1).transpose(1, 2)  # feature_map: (batch, 16, 2048)
        mlp_input = feature_map.mean(1)
        h0 = self.bn1(self.fc_init_h(mlp_input))  # h0: (batch, 512)
        c0 = self.bn2(self.fc_init_c(mlp_input))  # c0: (batch, 512)
        return feature_map, h0, c0 
