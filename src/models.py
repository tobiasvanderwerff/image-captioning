"""
Encoder-decoder classes/functions for image captioning.
The main idea is:
- calculate image features using CNN encoder
- feed calculated image features into the initial state of an LSTM language model, which makes 
  use of an attention mechanism in order to generate a caption for the image.
  
  TODO: REMOVE HOOK at the right spot
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

        self.mlp1 = MLP(decoder.num_hidden)
        self.mlp2 = MLP(decoder.num_hidden)
        
        logger.info(f"Number of trainable parameters: {count_parameters(self)}")  # TODO: this is not right for the encoder
        
    def forward(self, imgs, split, captions=None, seq_lengths=None, metrics_callback_fn=None):
        # This follows the implementation in the 2015 paper 'Show and Tell' by Vinyals et al. We feed the image
        # through the encoder, after which the LSTM is initialized by running the image features through the LSTM
        # and using the consequent hidden and cell state to run the LSTM for generating words.
        
        if type(split) == list or type(split) == tuple:
            split = split[0] # TODO: fix this?
        assert split in ['train', 'eval', 'test']
        
        batch, *_ = imgs.shape
        loss, sampled_ids, scores = None, None, None
        max_seq_len = self.max_seq_len if split == 'test' else captions.size(1)
        
        
        img_features = self.encoder(imgs)
        intermediate_layer1 = self.encoder.intermediate
        iml = intermediate_layer1.get('Convolution') 
        iml = torch.flatten(iml, 2, -1) 
        #[batch_size, num_hidden, featureMap_dir1, featureMap_dir2] -> [batch_size, num_hidden, featureMap_dir1 *featureMap_dir2]

        # Taking the average across the encoder hidden states
        # Pass this to 2 different MLPs used to initialise 
        # the hidden state h0 and cell state c0 of the LSTM decoder

        mlp_input = torch.mean(iml, dim=-1)
        h0 = self.mlp1(mlp_input)
        c0 = self.mlp2(mlp_input)
        h0, c0 = self.init_decoder_cell_state_and_hidden_state(h0, c0)

        if split == 'train':
            all_logits = self.decoder(captions, iml, seq_lengths.cpu(), h0, c0)
            loss = F.cross_entropy(all_logits.transpose(2, 1), captions, ignore_index=0)
        if split == 'eval' or split == 'test':
            all_logits, sampled_ids = [], []
            hiddens = [h0, c0]
            lstm_in = img_features.unsqueeze(1)
            for t in range(max_seq_len):
                lstm_out, hiddens = self.decoder.lstm(lstm_in, hiddens)

                # Attention ---------------------------------------------------------
                batch, seq_len, hidden_len = output.shape
                img_features = torch.transpose(img_features, 1, -1)
                n_annotations = img_features.size(1)
                attn_in = torch.zeros(batch, seq_len, n_annotations, self.num_hidden * 2).to(self.device)

                for b in range(batch):
                    for t in range(seq_len):
                        h = output[b, t]
                        # iterate over all annotation vectors
                        for k, ann in enumerate(img_features[b]):
                            #import pdb; pdb.set_trace()
                            attn_in[b, t, k, :] = torch.cat((ann, h))
                    
                attn_out = self.decoder.attn(attn_in).squeeze(-1)
                attn_weights = F.softmax(attn_out, dim=-1)
                img_features = img_features.unsqueeze(1)
                img_features = img_features.repeat(1, seq_len, 1, 1) 
     
                cs = torch.sum(attn_weights.unsqueeze(-1) * img_features, dim=2, keepdim=False)
                # -----------------------------------------------------------------------
                
                logits = self.decoder.fc(torch.cat((lstm_out, cs), dim=2))
                _, sample = logits.max(-1)
                lstm_in = self.decoder.emb(sample)
                sampled_ids.append(sample.squeeze(1))
                all_logits.append(logits)

            sampled_ids = torch.stack(sampled_ids, 1)
            all_logits = torch.cat(all_logits, 1)
            if captions is not None:
                scores = {}
                for n in range(1, 5):
                    # calculate BLEU scores
                    bleu_score = metrics_callback_fn(sampled_ids, captions, max_n=n)
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
    def __init__(self, num_hidden, embedding_dim, vocab_size, device, num_layers=2, bidirectional=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.device = device
        
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, num_hidden, num_layers, batch_first=True, bidirectional=bidirectional)
        self.attn = nn.Linear(self.num_hidden * 2, 1) 
        self.fc = nn.Linear(self.num_hidden * 2, vocab_size)
        #self.drop = nn.Dropout(p=0.5)
    
    def forward(self, captions, img_features, seq_lengths, h0, c0):      
        emb = self.emb(captions)  # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lengths - 1, batch_first=True)
        
        output, _ = self.lstm(emb, (h0, c0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Attention ----------------------------------------------------------
        
        batch, seq_len, hidden_len = output.shape
        img_features = torch.transpose(img_features, 1, -1)
        n_annotations = img_features.size(1)
        attn_in = torch.zeros(batch, seq_len, n_annotations, self.num_hidden * 2).to(self.device)
        
        for b in range(batch):
            for t in range(seq_len):
                h = output[b, t]
                # iterate over all annotation vectors
                for k, ann in enumerate(img_features[b]):
                    attn_in[b, t, k, :] = torch.cat((ann, h))
                    
        attn_out = self.attn(attn_in).squeeze(-1)
        attn_weights = F.softmax(attn_out, dim=-1)
        img_features = img_features.unsqueeze(1)
        img_features = img_features.repeat(1, seq_len, 1, 1) 
     
        cs = torch.sum(attn_weights.unsqueeze(-1) * img_features, dim=2, keepdim=False)
        logits = self.fc(torch.cat((output, cs), dim=2))  
        
        return logits


class ResNetEncoder(nn.Module):
    """ Pretrained resnet50 image encoder """
    
    def __init__(self, num_hidden):
        super().__init__()
        self.intermediate = {}  # Activation(s) of intermediate layers
        # Pre-trainer Resnet50
        self.pretrained = models.resnet50(pretrained=True)
        for p in self.pretrained.parameters():  # freeze the weights
            p.requires_grad = False
        # Attaching a hook to one of the last convoloutional layers
        self.pretrained.layer4[2].conv2.register_forward_hook(self.forward_hook('Convolution'))
        
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.intermediate.update({layer_name: output})
        return hook
    
    def forward(self, imgs):
        out = self.pretrained(imgs)
        return out
        
class MobileNetEncoder(nn.Module):
    """ Pretrained MobileNet v2 image encoder """

    def __init__(self, num_hidden):
        super().__init__()
        self.encoder = models.mobilenet_v2(pretrained=True)
        layer = self.encoder.classifier[1]
        in_features = layer.in_features
        
        self.encoder.classifier[1] = nn.Linear(in_features, num_hidden, bias=False)
        self.bn = nn.BatchNorm1d(num_hidden)
    
    def forward(self, imgs):
        with torch.no_grad(): 
            features = self.resnet(imgs)
        features = features.reshape(features.size(0), -1)
        features = self.classifier[1](features)
        features = self.bn(features)
        return features

class VGGNetEncoder(nn.Module):
    """ Pretrained VGG16 image encoder """

    def __init__(self, num_hidden):
        super().__init__()
        self.encoder = models.vgg16(pretrained=True)
        layer = self.encoder.classifier[6]
        in_features = layer.in_features
        
        self.encoder.classifier[6] = nn.Linear(in_features, num_hidden, bias=False)
        self.bn = nn.BatchNorm1d(num_hidden)
    
    def forward(self, imgs):
        with torch.no_grad(): 
            features = self.resnet(imgs)
        features = features.reshape(features.size(0), -1)
        features = self.classifier[6](features)
        features = self.bn(features)
        return features
    
class DenseNetEncoder(nn.Module):
    """ Pretrained DenseNet161 image encoder """

    def __init__(self, num_hidden):
        super().__init__()
        self.encoder = models.densenet161(pretrained=True)
        in_features = self.encoder.classifier.in_features
        
        self.encoder.classifier = nn.Linear(in_features, num_hidden, bias=False)
        self.bn = nn.BatchNorm1d(num_hidden)
    
    def forward(self, imgs):
        with torch.no_grad(): 
            features = self.resnet(imgs)
        features = features.reshape(features.size(0), -1)
        features = self.classifier(features)
        features = self.bn(features)
        return features  

class InceptionEncoder(nn.Module):
    """ Pretrained Inception v3 image encoder """

    def __init__(self, num_hidden):
        super().__init__()
        self.encoder = models.inception_v3(pretrained=True)
        in_features = self.encoder.fc.in_features
        
        self.encoder.fc = nn.Linear(in_features, num_hidden, bias=False)
        self.bn = nn.BatchNorm1d(num_hidden)
    
    def forward(self, imgs):
        with torch.no_grad(): 
            features = self.resnet(imgs)
        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        return features  	

