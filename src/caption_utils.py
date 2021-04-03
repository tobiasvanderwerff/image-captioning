# Building a dictionary of each word in the training set
# Note: this takes some time (a few mins)

# NOTES
# Does not yet do stemming
# e.g. should be going -> go and doors -> door. This would shrink the vocabulary and save memory + improve generalization

import string
import re
import logging
from pathlib import Path
from collections import OrderedDict
from operator import itemgetter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

def preprocess_tokens(annotation_file):
    """ 
    Preprocessing of image captions.
    annotation_file should be a path to a file containing one image/annotation pair per line, 
    where the image and annotation are separated by a \t symbol (tab). 
    """    
    captions = []
    known_word = {}
    
    img_annotations = Path(annotation_file).read_text().split('\n')
    for i, line in enumerate(img_annotations):
        if len(line) == 0:
            continue
        _, sentence = line.split('\t')  # we are only interested in the caption, not the image
        sentence = word_tokenize(sentence)
        for word in sentence:
            # Remove punctuation and capital letters
            word = word.translate(str.maketrans('','',string.punctuation)).lower()
            # To prevent adding the empty string
            if word != "":
                # Add the word to the dictionary
                if word in known_word.keys():
                    known_word[word] = known_word[word] + 1
                else:
                    known_word[word] = 1
 
        captions.append(sentence)
        if i % 10000 == 0:
            logger.info(f"Creating word dictionary: {i}/{len(img_annotations)}")


    # Remove the single occurences from the dictionary to save memory
    tokens = [word for word, cnt in known_word.items() if cnt > 1]
    tokens = ['<NOTSET>', '<UNKNOWN>', '<START>', '<END>'] + tokens
    idx2word = dict(enumerate(tokens))
    word2idx = {v: k for k, v in idx2word.items()}

    # The words in the original captions are replaced by the index of the word in the dictionary
    # If the word is not in the dictionary, it is replaced with the index of '<UNKNOWN>' (1)
    train_sequences = []
    for i, line in enumerate(img_annotations):
        if len(line) == 0:
            continue
        img_id, sentence = line.split('\t')
        img_id = re.match(r'.*\.jpg', img_id).group(0)
        sentence = word_tokenize(sentence)
        sentence_enc = []  # create a new list for the encoded sentence
        for j, word in enumerate(sentence):
            word = word.translate(str.maketrans('','',string.punctuation)).lower()
            if j == 0:  # start the beginning of the caption with a <START> token
                idx = word2idx['<START>']
                sentence_enc.append(idx)
            if word == '':
                continue
            idx = word2idx.get(word, word2idx['<UNKNOWN>'])
            sentence_enc.append(idx)
        sentence_enc.append(word2idx['<END>'])  # end with a <END> token

        train_sequences.append((img_id, sentence_enc))
        if i % 10000 == 0:
            logger.info(f"Replacing tokens with numerical values: {i}/{len(img_annotations)}")
            
    return train_sequences, tokens, word2idx, idx2word