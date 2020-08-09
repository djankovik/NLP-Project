import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import re
import zipfile
import itertools
import urllib
from nltk.tokenize import word_tokenize
import lxml.etree
from functools import reduce
import operator
import os,glob
from numpy import array
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize
import librosa
from statistics import mean
import keras
import keras.backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

TRAIN_DATA_PATH = r'data\semeval2016-task6-trainingdata.txt'
TEST_DATA_PATH_A = r'data\SemEval2016-Task6-subtaskA-testdata-gold.txt'

def print_dataset_stats(targets,stances):
    target_stances_dict = {}
    for target in list(set(targets)):
        target_stances_dict[target] = {'AGAINST':0,'FAVOR':0,'NONE':0,'total':0}

    for (trg,stnc) in zip(targets,stances):
        target_stances_dict[trg][stnc]+=1
        target_stances_dict[trg]['total']+=1
    
    for target in target_stances_dict.keys():
        print(str(target)+" -> "+str(target_stances_dict[target]))

def get_vocabulary(sentences):
    vocab = []
    for sentence in sentences:
        vocab.extend(tokenize_sentence(sentence))
    return list(set(vocab))

def tokenize_sentence(sentence):
    return word_tokenize(sentence.lower())

def tokenize(text):
    # obtains tokens with a least 1 alphabet
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def get_target_or_stance_onehot(data,vocab):
    unique = list(set(vocab))
    uniq_dict = {unique[i]: i for i in range(len(unique))}
    ohots = []
    for thing in data:
        oh = [0]*len(unique)
        oh[uniq_dict[thing]] = 1
        ohots.append(oh)
    return ohots

def get_tweets_as_lists_of_words(tweets):
    tweetswords = []
    for tweet in tweets:
        tweetswords.append(tokenize(tweet))
    return tweetswords

def read_data(filesrc):
    targets = []
    tweets = []
    stances = []
    #print(os.getcwd()+filesrc)
    with open(filesrc) as file:
        line = file.readline() # to pass through the first line that has column headers
        line = file.readline()
        while line:
            parts = re.split(r'\t+', line)
            targets.append(parts[1].strip().replace("#SemST","").lower())
            tweets.append(parts[2].strip().replace("#SemST",""))
            stances.append(parts[3].strip().replace("#SemST",""))
            line = file.readline()
    # print(set(targets))
    # print(set(stances))
    return [targets,tweets,stances]



targets_train,tweets_train,stances_train = read_data(TRAIN_DATA_PATH)
#print("tweets train len: "+str(len(tweets_train))+", stances train len: "+str(len(stances_train)))
targets_test,tweets_test,stances_test = read_data(TEST_DATA_PATH_A)
#print("tweets test len: "+str(len(tweets_test))+", stances test len: "+str(len(stances_test)))

target_vocabulary = list(set(targets_train))
stance_vocabulary = list(set(stances_train))

stances_onehot_train = get_target_or_stance_onehot(stances_train,stance_vocabulary)
stances_onehot_test = get_target_or_stance_onehot(stances_test,stance_vocabulary)

tweets_wordlists_train = get_tweets_as_lists_of_words(tweets_train)
tweets_wordlists_test = get_tweets_as_lists_of_words(tweets_test)

# print("train data stats")
# print_dataset_stats(targets_train,stances_train)
# print("test data stats")
# print_dataset_stats(targets_test,stances_test)
