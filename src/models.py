"""
Encoder-decoder classes/functions for image captioning.
The main idea is:
- calculate image features using CNN encoder
- feed calculated image features into the initial state of an LSTM language model, which makes 
  use of an attention mechanism in order to generate a caption for the image.
  
The layer we want to extract the feature vector from (using a forward hook):
model.layer4[2].bn2
"""

# ToDo: 
# - Replace output layer of decoder with deep layer 
# (eg RBMs, Autoencoders, or neural autoregressive distribution estimators)
# - Let LSTM output word by word instead of whole caption

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.num_hidden = num_hidden

        self.lin1 = nn.Linear(num_hidden, num_hidden)
        self.relu1 = nn.ReLU(inplace=True)
       
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate = {}  # Activation(s) of intermediate layers
        # Pre-trainer Resnet34
        self.pretrained = models.resnet34(pretrained=True)
        # Attaching a hook to the last layer with informative feature vector
        self.pretrained.layer4[2].bn2.register_forward_hook(self.forward_hook('BatchNorm'))
        
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.intermediate.update({layer_name: output})
        return hook

    def forward(self, input):
        out = self.pretrained(input)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.mlp1 = MLP(decoder.num_hidden)
        self.mlp2 = MLP(decoder.num_hidden)
        self.decoder = decoder
        
    def forward(self, img, caption, seq_lengths, targets=None):
        img_features = self.encoder(img)
        intermediate_layer1 = self.encoder.intermediate
        #intermediate layer shape: [2, 512, 4, 4]
        #im_features shape (batch, img_features)
        iml = intermediate_layer1.get('BatchNorm')
        iml = torch.flatten(iml, 2, -1)

        # Compute mean of the feature map
        mlp_input = torch.mean(iml, dim=-1)

        # Feed feature vector into MLP
        h0 = self.mlp1(mlp_input)
        c0 = self.mlp2(mlp_input)

        # Feed LSTM decoder the mlp output, captions,  
        # and the feature vector of the intermediate layer
        logits = self.decoder(caption, h0, c0, seq_lengths, iml)
        loss = None
        if targets is not None:  # calculate the loss
            loss = F.cross_entropy(logits.permute(0, 2, 1), targets)   
        return logits, loss
    
      
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
        self.lstm = nn.LSTM(embedding_dim, num_hidden, num_layers=2, batch_first=True, dropout=0, bidirectional=bidirectional) 
        self.attn = nn.Linear(self.num_hidden * 2, 1) 
        self.fc = nn.Linear(self.num_hidden * 2, vocab_size)
    
    def forward(self, caption, h0, c0, seq_lengths, intermediate_layer):
"""  
          -------------- Inputs --------------------  
             caption = caption; shape = (batch, seq_len)   
             mlp = image feature vector averaged and passed through MLP   
             seq_lengths = ''   
             intermediate_layer = output of 'encoder.layer4[2].bn2'    
"""
        
        # Initialising the hidden state and cell state of the LSTM with 
        # the two different MLPs

        c0, h0 = self.init_cell_state_and_hidden_state(caption.shape[0], c0, h0)
        
        emb = self.emb(caption)  # (batch, seq_len) -> (batch, seq_len, embedding_dim)
     
        # Pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # See https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        # We use seq_lengths - 1 because we do not want to count the last <END> token in each caption.
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lengths - 1, batch_first=True)
        
        out, _ = self.lstm(emb, (h0, c0))

        # Undo the packing operation
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (batch, seq_len, num_directions * num_hidden)

batch, seq_len, directions_times_hidden = out.shape
_, _, n_annotations = intermediate_layer.shape
        # ---------------- Attention -----------------------

        # torch.Size([2, 512, 16]) -> torch.Size([2, 16, 512])
        intermediate_layer = torch.transpose(intermediate_layer, 1, -1)
        
        attn_in = torch.zeros(out.size(0), out.size(1), out.size(2) * 2).to(self.device)
        
        for i in range(out.size(0)):
            seq_len = seq_lengths[i].item() - 1

            stack_intermediate_layer = torch.zeros(out.size(1), out.size(2)).to(self.device)

            # As the two tensors (out, intermediate_layer) have different shapes,
            # I copy from intermediate_layer until I have reached shape [2, 38, 512] ('out' shape)
            for j in range(seq_len):
                idx = j % (intermediate_layer.size(1) - 1)
                stack_intermediate_layer[j] = intermediate_layer[i][idx].clone().detach()

            attn_in[i, :seq_len] = torch.cat((stack_intermediate_layer, out[i, :seq_len]), dim = 1)

        attn_out = self.attn(attn_in)
        attn_weights = F.softmax(attn_out, dim=1)
        
        cs = torch.sum(attn_weights.unsqueeze(1) * out, dim=1, keepdim=False)

        #import pdb; pdb.set_trace()
        logits = self.fc(torch.cat((out, cs), dim=2))  
        return logits
                     
    def init_cell_state_and_hidden_state(self, batch_size, c0, h0):
        c0 = c0.repeat(self.num_layers * self.num_directions, 1, 1)
        h0 = h0.repeat(self.num_layers * self.num_directions, 1, 1)
        c0 = c0.to(self.device)
        h0 = h0.to(self.device)
        return c0, h0
       
def get_encoder(num_hidden):
    """ Returns a pretrained resnet34 image encoder. """
    encoder = ResNet()
    in_features = encoder.pretrained.fc.in_features
    encoder.fc = nn.Linear(in_features, num_hidden)  # ensure that the output of the encoder is of the right size 
    for p in encoder.parameters():  # freeze the weights
        p.requires_grad = False
    encoder.fc.weight.requires_grad = True  # only train the last linear layer of the encoder
    return encoder
                    
