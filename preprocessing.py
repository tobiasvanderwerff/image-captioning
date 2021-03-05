# Building a dictionary of each word in the training set
# Note: this takes some time (a few mins)

# NOTES
# Still have to remove the empty string 
# (for instance in the example above the '.' becomes the empty string which has position 3 in the sorted dictionary)
# Does not yet do stemming
# e.g. should be going -> go and doors -> door. This would shrink the vocabulary and save memory + improve generalization

import string
from collections import OrderedDict
from operator import itemgetter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess(dataset):
    captions = []
    known_word = {}

    for i in range(len(dataset)):
        sentence = word_tokenize(dataset[i][1])
        for word in sentence:
            # Remove punctuation and capital letters
            word = word.translate(str.maketrans('','',string.punctuation)).lower()
            # Add the word to the dictionary
            if word in known_word:
                known_word[word] = known_word[word] + 1
            else:
                known_word[word] = 1

        captions.append(sentence)
        if i % 500 == 0:
            print(i, "/", len(dataset))
            
#     print(len(captions))
#     print('\n\n')
#     print(known_word)
#     print(len(known_word))

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
    known_words_final = [('<NOTSET>', 0), ('<UNKNOWN>', 0)] + known_words_final

    known_words_final = dict(known_words_final)
    #known_words_final = OrderedDict(sorted(known_word2.items(), key=itemgetter(1), reverse = True))

#     print(known_words_final)
#     print(len(known_words_final))

    # The words in the original captions are replaced by the index of the word in the dictionary
    # If the word is not in the dictionary, it is replaced with the index of '<UNKNOWN>' (1)
    train_sequences = []
    max_length = 0

    for i in range(len(dataset)):
        sentence = word_tokenize(dataset[i][1])
        #print(sentence)
        j = 0
        for word in sentence:
            word = word.translate(str.maketrans('','',string.punctuation)).lower()
            # If the word is in our dictionary, it is replaced by the index in the sorted dictionary
            if word in known_words_final:
                sentence[j] = list(known_words_final.keys()).index(word)
            else:
                # If not, it is unknown and we give index 1
                sentence[j] = 1
            j = j + 1

        #print(sentence, '\n\n')
        train_sequences.append(sentence)
        if(len(sentence) > max_length):
            max_length = len(sentence)
        if i % 500 == 0:
            print(i, "/", len(dataset))
            

#     print('\nExample:')
#     print(ds_train[0][1])
#     print(train_sequences[0])
#     print('\n')
#     print(len(train_sequences))
#     print(max_length)

    # Pad each vector to the maximum length of the captions (max_length) with the index of '<NOTSET>' (0)
    for k in range(len(train_sequences)):
        for l in range(max_length - len(train_sequences[k])):
            train_sequences[k].append(0)
            
    # Print some examples
#     for i in range(10):
#         print(train_sequences[i])

    return train_sequences, known_words_final, max_length