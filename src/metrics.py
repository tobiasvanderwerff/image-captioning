import logging

import numpy as np
import torch
from torchtext.data.metrics import bleu_score

logger = logging.getLogger(__name__)

@torch.no_grad()
def calculate_bleu_score(candidate, reference, lang, max_n=2):
    """
    Input:
    - candidate: numpy array or pytorch tensor of shape (batch, max_seq_len1)
    - reference : numpy array or pytorch tensor of shape (batch, 5, max_seq_len2)
    - lang: Lang class instance that can be used to decode numerical captions
    
    Output:
    - scores: numpy array containing BLEU scores for each sample, of shape (batch,)
    """
    if isinstance(candidate, torch.Tensor):
        candidate = candidate.cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    
    scores = []
    candidate = [lang.decode_caption(ann).split() for ann in candidate]
    reference = [[lang.decode_caption(ann).split() for ann in ann_list] for ann_list in reference]
    
    score = bleu_score(candidate, reference, max_n=max_n, weights=[1 / max_n for _ in range(max_n)])
    scores.append(score * 100)
    
    # logging
#     blue_mean = np.mean(scores)
#     logger.info(f"Mean BLEU score: {blue_mean:.2f}")
    
    return np.array(scores)