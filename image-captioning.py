#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import random
import re
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.core.debugger import set_trace


# In[3]:


# logging set-up
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)


# In[54]:


# Make deterministic
random.seed(47)
torch.manual_seed(47);


# In[5]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device


# In[6]:


# path = Path('/data/s4314719/image-captioning/')  # set this to the path where the dataset should be stored
# path = Path('/image_data/')  # set this to the path where the dataset should be stored
# path = Path('C:/Users/leann/Documents/Master AI IS/Deep Learning/image_data/')

path.mkdir(exist_ok=True)


# In[7]:


img_dir = path/'Flicker8k_Dataset'
ann_file = path/'Flickr8k.token.txt'


# In[8]:


URL = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/'
FILENAMES = ['Flickr8k_text.zip', 'Flickr8k_Dataset.zip']


# In[9]:


# Download dataset
import requests
from zipfile import ZipFile

for fn in FILENAMES:
    if not (path/fn).exists():  # do not download if the dataset has already been downloaded
        print(f'Downloading {fn}')
        r = requests.get(URL + fn)
        (path/fn).write_bytes(r.content)
        print(f'Extracting files...')
        with ZipFile(path/fn) as zipf:
            zipf.extractall(path)


# In[10]:


list(path.glob('*')) # list all files


# In[11]:


print((path/'Flickr8k.token.txt').read_text()[:696])


# In[47]:


from torch.utils.data import Dataset, DataLoader
from src.caption_utils import preprocess_tokens
from PIL import Image

class FlickrDataset(Dataset):
    unk_token = '<UNKNOWN>'
    ns_token = '<NOTSET>'
    start_token = '<START>'
    end_token = '<END>'
        
    def __init__(self, img_dir, img_captions_enc, known_words, ann_file, img_ids, trnsf=None):
        self.img_dir = Path(img_dir)
        self.known_words = known_words
        self.ann_file = Path(ann_file)
        self.trnsf = trnsf
        self.annotations = {}
        self.targets = {}
        
        #img_captions_enc, self.known_words, _ = preprocess_tokens(ann_file)
        
        end_idx = list(self.known_words.keys()).index(self.end_token)
        # the targets are the captions shifted one place to the right.
        target_lst = [[c[i] for i in range(1, len(c)) if c[i] != self.ns_token] 
                      for c in list(zip(*img_captions_enc))[1]]
        
        # iterate through the annotation file and create (image, caption) pairs
        img_ids = Path(img_ids).read_text().split('\n')
        for i, (img_id, annotation) in enumerate(img_captions_enc):
            # TODO: we only consider the first of 5 annotations for each image. Do something with the other four.
            if img_id in self.annotations or not (img_dir/img_id).exists() or not img_id in img_ids:
                continue
            self.annotations.update({img_id: annotation})
            self.targets.update({img_id: target_lst[i]})
        self.img_ids = list(self.annotations.keys())
        
    def __len__(self): 
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        annotation = self.annotations[img_id]
        annotation_len = len(annotation)  # TODO: caption lengths are for padded captions, so incorrect
        im = Image.open(self.img_dir/img_id)
        if self.trnsf is not None:
            im = self.trnsf(im)
        target = self.targets[img_id]
        return im, np.array(annotation), annotation_len, np.array(target)
    
    def decode_caption(self, caption):
        res = ''
        for i, word_idx in enumerate(caption):
            word = list(self.known_words.keys())[word_idx]
            res += word
            if word == self.end_token:
                break
            res += ' '
        return res
    
    def encode_caption(self, caption):
        res = []
        for i, word in enumerate(caption):
            if word in known_words:
                # this is not very efficient; can be improved by creating a dictionary of (word, encoding) pairs
                res.append(list(self.known_words.keys()).index(word))
            else:
                res.append(list(self.known_words.keys()).index(self.unk_token))
        return res


# ### Preprocessing the images

# In[13]:


# Calculate mean and standard deviation over all channels for normalization
# mean = torch.cat([ex[0].view(3, -1) for ex in data], 1).mean(1)
# std = torch.cat([ex[0].view(3, -1) for ex in data], 1).std(1)
mean = np.array([0.4629, 0.4468, 0.4050])
std = np.array([0.2661, 0.2587, 0.2727])


# In[14]:


from torchvision import transforms

# We resize the images and center crop them, as in the 'Show, attend and tell' paper. In the paper they
# resize the shortest side to 256 while preserving the aspect ratio, and then apply a center crop to 
# end up with an image of size 224x224. Note that for quick experimentation, we can use a smaller image 
# size (e.g. 128x128) to speed up training. 
trnsf = {
    'train': transforms.Compose([
        transforms.Resize(128),
#         transforms.Resize(256),
        transforms.CenterCrop(128),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'eval': transforms.Compose([
        transforms.Resize(128),
#         transforms.Resize(256),
        transforms.CenterCrop(128),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
}


# ### Preprocessing the captions

# In[15]:


# Calls the preprocessing of the captions which may take a while (progress is printed)
img_captions_enc, known_words, _ = preprocess_tokens(ann_file)
vocab_size = len(known_words)
vocab_size


# ### Split data up into train and evaluation set

# In[48]:


# We use the predefined train/eval splits of Flickr8k
ds_train = FlickrDataset(img_dir, img_captions_enc, known_words, ann_file,
                         path/'Flickr_8k.trainImages.txt', trnsf=trnsf['train'])
ds_eval = FlickrDataset(img_dir, img_captions_enc, known_words, ann_file,
                        path/'Flickr_8k.devImages.txt', trnsf=trnsf['eval'])
len(ds_train), len(ds_eval)


# In[49]:


# show an example from the dataset
img = '1000268201_693b08cb0e.jpg'
caption = ds_train.annotations[img]
print(img)
print(caption)
print(ds_train.decode_caption(caption))


# ### Visualizing some images

# In[51]:


def imshow(im):
    imnp = im.numpy()
    imnp = imnp * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]  # undo normalization
    imnp = np.clip(imnp, 0, 1)
    imnp = imnp.transpose([1, 2, 0])
    plt.imshow(imnp)


# In[55]:


# Plot some examples from the training set
from matplotlib import pyplot as plt

nrows = 2
ncols = 2
plt.figure(figsize=(16, 12))
for i in range(nrows * ncols):
    idx = random.randint(0, len(ds_train)-1)
    im, caption, _, _ = ds_train[idx]
    plt.subplot(nrows, ncols, i+1)
    plt.title(ds_train.decode_caption(caption))
    imshow(im)
    plt.axis('off')


# ### Model initialization and training

# In[60]:


# hyperparameters
num_hidden = 512
embedding_dim = 512
batch_size = 64
epochs = 10
num_workers = 0


# In[61]:


from src.models import EncoderDecoder, LSTMDecoder, get_encoder

encoder = get_encoder(num_hidden)
decoder = LSTMDecoder(num_hidden, embedding_dim, vocab_size, device)
encoder, decoder = encoder.to(device), decoder.to(device)

model = EncoderDecoder(encoder, decoder)
model.to(device);


# In[62]:


# model


# In[63]:


optimizer = optim.Adam(model.parameters())


# In[64]:


# overfit on a mini dataset
# ds_train_mini = torch.utils.data.Subset(ds_train, [0, 1, 2, 3, 4, 5])


# In[65]:


from src.trainer import Trainer, TrainerConfig

config = TrainerConfig(batch_size=batch_size, epochs=epochs, 
                       num_workers=num_workers, track_grad_norm=True)
# trainer = Trainer(config, model, optimizer, ds_train_mini)
trainer = Trainer(config, model, optimizer, ds_train, ds_eval)


# In[66]:


trainer.train()  # overfit on a mini dataset for a quick sanity check


# In[73]:


# Plot the gradients over time
# plt.plot(trainer.grad_norms)
# plt.show()


# ### Sample some predictions

# In[29]:


import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# TODO: make it easier to get indices for individual tokens
@torch.no_grad()
def sample(img, top_k=1):
    """ Perform inference on the model by feeding it an image and generating a caption. """
    start_token, end_token = FlickrDataset.start_token, FlickrDataset.end_token
    next_token = start_token
    res = start_token
    tokens = ds_train.encode_caption([start_token])
    
    model.eval()
    img = img.to(device)
    img_features = model.encoder(img)  # shape: (batch, img_features)
    h = img_features.repeat(model.decoder.num_layers, 1, 1)  # shape: (num_layers, batch, img_features)
    while next_token != end_token:
        x = torch.tensor([tokens[-1]]).unsqueeze(0).to(device)

        logits, h = model.decoder(x, h, np.array([2]))

        klogits, kidxs = torch.topk(logits, top_k)
        probs = F.softmax(klogits, dim=-1).data.cpu()
        
        token_idx = kidxs[Categorical(probs).sample()]  # sample a token
        tokens.append(token_idx)
        
        next_token = ds_train.decode_caption([token_idx])
    caption = ds_train.decode_caption(tokens)
    return caption


# In[30]:


sample(ds_train_mini[1][0].unsqueeze(0))


# In[ ]:




