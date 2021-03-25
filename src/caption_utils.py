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
    """ Preprocessing of image captions.
        annotation_file should be a path to a file containing one image/annotation pair per line, 
        where the image and annotation are separated by a \t symbol (tab). """    
   
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
                if word in known_word:
                    known_word[word] = known_word[word] + 1
                else:
                    known_word[word] = 1
 
        captions.append(sentence)
        if i % 10000 == 0:
            logger.info(f"Creating word dictionary: {i}/{len(img_annotations)}")


    # Remove the single occurences from the dictionary to save memory
    # With single occurences: vocab_size = 3858
    # Without single occurences: vocab_size = 2006
    known_word2 = {}

    for word in known_word:
        if(known_word[word] > 1):
            known_word2[word] = known_word[word]

#     print(known_word2)
#     print(len(known_word2))

    # Sorting the dictionary by frequency (descending)
    known_words_final = sorted(known_word2.items(), key=lambda x: x[1], reverse=True)
    known_words_final = [('<NOTSET>', 0), ('<UNKNOWN>', 0), ('<START>', 0), ('<END>', 0)] + known_words_final

    known_words_final = dict(known_words_final)

#     print(known_words_final)
#     print(len(known_words_final))

    # The words in the original captions are replaced by the index of the word in the dictionary
    # If the word is not in the dictionary, it is replaced with the index of '<UNKNOWN>' (1)
    train_sequences = []
    max_length = 0

    for i, line in enumerate(img_annotations):
        if len(line) == 0:
            continue
        img_id, sentence = line.split('\t')
        img_id = re.match(r'.*\.jpg', img_id).group(0)
        sentence = word_tokenize(sentence)
        sentence_enc = []  # create a new list for the encoded sentence
        #print(sentence)
        for j, word in enumerate(sentence):
            word = word.translate(str.maketrans('','',string.punctuation)).lower()
            if j == 0:  # start the beginning of the caption with a <START> token
                sentence_enc.append(list(known_words_final.keys()).index('<START>'))
            # If the word is in our dictionary, it is replaced by the index in the sorted dictionary
            if word == '':
                continue
            if word in known_words_final:
                sentence_enc.append(list(known_words_final.keys()).index(word))
            else:
                # If not, it is unknown and we give index 1
                sentence_enc.append(1)
        sentence_enc.append(list(known_words_final.keys()).index('<END>'))  # end with a <END> token

        #print(res, '\n\n')
        train_sequences.append((img_id, sentence_enc))
        if(len(sentence_enc) > max_length):
            max_length = len(sentence_enc)
        if i % 10000 == 0:
            logger.info(f"Replacing tokens with numerical values: {i}/{len(img_annotations)}")
            
#     print('\nExample:')
#     print(ds_train[0][1])
#     print(train_sequences[0])
#     print('\n')
#     print(len(train_sequences))
#     print(max_length)

    return train_sequences, known_words_final, max_length