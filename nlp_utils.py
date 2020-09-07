import torch
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
  # pass
  return nltk.word_tokenize(sentence)
  
def stem(word):
  # pass
  return stemmer.stem(word.lower())
  
def bag_of_words(tokenized_sentence, all_words):
  # pass
  tokenized_sentence = [stem(word) for word in tokenized_sentence]
  bag = np.zeros(len(all_words), dtype=np.float32)

  for ind, word in enumerate(all_words):
    if(word in tokenized_sentence):
      bag[ind] = 1.0

  return bag