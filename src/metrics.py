import logging

import numpy as np
import torch
from torchtext.data.metrics import bleu_score


@torch.no_grad()
def calculate_bleu_score(candidate, reference, lang, max_n=2):
    """
    Input:
    - candidate: numpy array or pytorch tensor of shape (batch, max_seq_len1)
    - reference : numpy array or pytorch tensor of shape (batch, 5, max_seq_len2)
    - lang: Lang class instance that can be used to decode numerical captions
    
    Output:
    - scores: list containing BLEU scores for each sample, where len(scores) == batch
    """
    if isinstance(candidate, torch.Tensor):
        candidate = candidate.cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    
    scores = []
    for cand, ref_list in zip(candidate, reference):  # calculate the BLEU score for all items in the batch
        cand = [lang.decode_caption(cand).split()]
        ref_list = [[lang.decode_caption(ann).split() for ann in ref_list]]
        score = bleu_score(cand, ref_list, max_n=max_n, weights=[1 / max_n for _ in range(max_n)])
        scores.append(score * 100)
    
    return scores