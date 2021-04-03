from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.caption_utils import preprocess_tokens
from PIL import Image

class Lang:
    unk_token = '<UNKNOWN>'
    ns_token = '<NOTSET>'
    start_token = '<START>'
    end_token = '<END>'
    
    def __init__(self, tokens, word2idx, idx2word):
        self.tokens = tokens
        self.word2idx = word2idx
        self.idx2word = idx2word
        
    def decode_caption(self, caption):
        res = ''
        for i, word_idx in enumerate(caption):
            word = self.idx2word[word_idx]
            res += word
            if word == self.end_token:
                break
            res += ' '
        return res
    
    def encode_caption(self, caption):
        res = []
        for i, word in enumerate(caption):
            idx = self.word2idx.get(word, self.unk_token)
            res.append(idx)
        return res

class FlickrDataset(Dataset):
    unk_token = '<UNKNOWN>'
    ns_token = '<NOTSET>'
    start_token = '<START>'
    end_token = '<END>'
        
    def __init__(self, img_dir, img_captions_enc, lang, ann_file, img_ids, split, trnsf=None):
        self.img_dir = Path(img_dir)
        self.lang = lang
        self.ann_file = Path(ann_file)
        self.trnsf = trnsf
        self.split = split
        self.annotations = {}
        
        assert self.split in ['train', 'eval', 'test']
        
        # iterate through the annotation file and create (image, caption) pairs
        img_ids = Path(img_ids).read_text().split('\n')
        for i, (img_id, annotation) in enumerate(img_captions_enc):
            if not (img_dir/img_id).exists() or not img_id in img_ids:
                continue
            if img_id in self.annotations:
                img_annotations = self.annotations[img_id]
                img_annotations.append(annotation)
            else:
                img_annotations = [annotation]
            self.annotations.update({img_id: img_annotations})
        self.img_ids = list(self.annotations.keys())
        
    def __len__(self): 
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_annotations = self.annotations[img_id]
        if self.split == 'train':
            ann_idx = np.random.randint(0, len(img_annotations) - 1)  # select one of the five annotations at random
            annotation = img_annotations[ann_idx]
        else:
            # select all 5 captions for evaluation, for the purposes of calculating BLEU score
            annotation = img_annotations
        im = Image.open(self.img_dir/img_id)
        if self.trnsf is not None:
            im = self.trnsf(im)
        return im, annotation, self.split