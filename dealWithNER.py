import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import numpy as np
import os
import pickle
from readDataFromFile import tweets_train, tweets_test, stances_onehot_train, stances_onehot_test
from utils import *
from sklearn.metrics import classification_report
import en_core_web_sm
nlp = en_core_web_sm.load()

def get_ner_spacy_for_tweet(tweet_text):
    doc = nlp(tweet_text)
    word_labels = [(X.text, X.label_) for X in doc.ents]
    return word_labels

def get_tweets_as_spacy_NER_lists(tweets):
    tweets_spacy = []
    for tweet in tweets:
        ners = get_ner_spacy_for_tweet(tweet)
        tweet_ners = []
        for (word,tag) in ners:
            if tag in ["PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE","MONEY","ORDINAL"]:
                tweet_ners.append(word)
        tweets_spacy.append(tweet_ners)
    return tweets_spacy
        

def load_embeddings(file_name, vocabulary):
    print('loading embeddings')
    embeddings = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            parts = line.split(' ')
            vals = np.array(parts[1:], dtype=np.float)
            if parts[0] in vocabulary:
                embeddings[parts[0]] = vals
            line = doc.readline()
    return embeddings

def load_embedding_weights(vocabulary, glovepath="",savepath="",embedding_size=50):
    if os.path.exists(savepath):
        with open(savepath, 'rb') as f:
            embedding_matrix = pickle.load(f)
    else:
        #print('Creating embedding weights...')
        embeddings = load_embeddings(glovepath,vocabulary)
        embedding_matrix = np.zeros((len(vocabulary), embedding_size))
        for i in range(len(vocabulary)):
            if vocabulary[i] in embeddings.keys():
                embedding_matrix[i] = embeddings[vocabulary[i]]
            else:
                embedding_matrix[i] = np.random.standard_normal(embedding_size)
        with open(savepath, 'wb') as f:
            pickle.dump(embedding_matrix, f)
    return embedding_matrix

pattern = 'NP: {<DT>?<JJ>*<NN>}'

def get_postagged_text(text):
    sent = nltk.word_tokenize(text)
    sent = nltk.pos_tag(sent)
    return sent

# Spacy tags: https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
#   PERSON	People, including fictional.
#   NORP	Nationalities or religious or political groups.
#    FAC	Buildings, airports, highways, bridges, etc.
#    ORG	Companies, agencies, institutions, etc.
#    GPE	Countries, cities, states.
#   LOC	Non-GPE locations, mountain ranges, bodies of water.
#   PRODUCT	Objects, vehicles, foods, etc. (Not services.)
#   EVENT	Named hurricanes, battles, wars, sports events, etc.
#   WORK_OF_ART	Titles of books, songs, etc.
#    LAW	Named documents made into laws.
#    LANGUAGE	Any named language.
#    DATE	Absolute or relative dates or periods.
#    TIME	Times smaller than a day.
#    PERCENT	Percentage, including ”%“.
#    MONEY	Monetary values, including unit.
#   QUANTITY	Measurements, as of weight or distance.
#    ORDINAL	“first”, “second”, etc.
#    CARDINAL	Numerals that do not fall under another type.

def get_tweets_as_wordOhs(tweets,vocabulary,size=100,padto=10):
    wordOhs = []
    for tweet in tweets:
        twt_words = []
        for word in tweet:
            if word in vocabulary and vocabulary.index(word) < size and len(twt_words) < padto:
                word_oh = [0]*size
                word_oh[vocabulary.index(word)] = 1
                twt_words.append(word_oh)
        while len(twt_words) < padto:
            twt_words.append([0]*size)
        wordOhs.append(twt_words)
    return wordOhs

def get_spacy_tags_for_text(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    tagged = [(X.text, X.label_) for X in doc.ents]
    return tagged

def get_tweets_as_spacy_tag_lists(tweets):
    spacytagged_tweeets = []
    for tweet in tweets:
        stags = get_spacy_tags_for_text(tweet)
        relevant_words = [word for (word,tag) in stags]
        print(relevant_words)
        spacytagged_tweeets.append(relevant_words)
    return spacytagged_tweeets

def get_sorted_dictionary_vocab(tweets):
    dictionary = {}
    for tweet in tweets:
        for nn in tweet:
            if nn in dictionary:
                dictionary[nn] += 1
            else:
                dictionary[nn] = 1
    sorteddict_asc = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    return (sorteddict_asc, list(sorteddict_asc))

def get_tweets_as_ner_lists(tweets):
    tweets_nouns = []
    for tweet in tweets:
        postags = get_postagged_text(tweet)
        twt_pstg = []
        for (word,tag) in postags:
            if 'NN' in tag:
                twt_pstg.append(word.lower())
        tweets_nouns.append(twt_pstg)
    return tweets_nouns

def get_tweetnerlists_vocab_vectors(tweets_ner_lists,vocabulary,size=500):
    tweet_vectors = []
    for tweet in tweets_ner_lists:
        tweet_vector = []
        for nn in vocabulary[0:size]:
            if nn in tweet:
                tweet_vector.append(1)
            else:
                tweet_vector.append(0)
        tweet_vectors.append(tweet_vector)
    return tweet_vectors

def get_tweetnerlists_onehots_intarray(tweets_ner_lists,vocabulary, padtosize=12,padwith=0):
    tweets_onehots = []
    for tweet in tweets_ner_lists:
        tweet_onehot = []
        for nn in tweet:
            if len(tweet_onehot) < padtosize:
                if nn in vocabulary:
                    tweet_onehot.append(vocabulary.index(nn))
                else:
                    tweet_onehot.append(0)
        while len(tweet_onehot) < padtosize:
            tweet_onehot.append(0)
        tweets_onehots.append(tweet_onehot)
    return tweets_onehots

tweets_ners_train = get_tweets_as_ner_lists(tweets_train)
tweets_ners_test = get_tweets_as_ner_lists(tweets_test)
dict_vocab_nns, vocab_nns = get_sorted_dictionary_vocab(tweets_ners_train)

tweets_ner_vocabohvec_train = get_tweetnerlists_vocab_vectors(tweets_ners_train,vocab_nns,size = 500)
tweets_ner_vocabohvec_test = get_tweetnerlists_vocab_vectors(tweets_ners_test,vocab_nns,size = 500)
tweets_ner_ohvec_train = get_tweetnerlists_onehots_intarray(tweets_ners_train,vocab_nns)
tweets_ner_ohvec_test = get_tweetnerlists_onehots_intarray(tweets_ners_test,vocab_nns)

tweets_ner_wordsohvecs_train = get_tweets_as_wordOhs(tweets_ners_train,vocab_nns,size=300)
tweets_ner_wordsohvecs_test = get_tweets_as_wordOhs(tweets_ners_test,vocab_nns,size=300)

# embedding_matrix_ner = load_embedding_weights(vocab_nns,glovepath='data\glove.6B.50d.txt',savepath="data\embedding_matrix_ner.pkl")
# embedding_matrix_ner_twitter = load_embedding_weights(vocab_nns,glovepath='data\glove.twitter.27B.50d.txt',savepath="data\embedding_matrix_ner_twt.pkl")


#SPACY
tweets_spacy_train = get_tweets_as_spacy_NER_lists(tweets_train)
tweets_spacy_test = get_tweets_as_spacy_NER_lists(tweets_test)
spacy_NER_vocabulary_dict, spacy_vocab = get_sorted_dictionary_vocab(tweets_spacy_train)

tweets_spacy_ohvec_train = get_tweetnerlists_onehots_intarray(tweets_spacy_train,spacy_vocab)
tweets_spacy_vocabvec_train = get_tweetnerlists_vocab_vectors(tweets_spacy_train,spacy_vocab,size=300)
tweets_spacy_wordohsvecs_train = get_tweets_as_wordOhs(tweets_spacy_train,spacy_vocab,size=300)

tweets_spacy_ohvec_test = get_tweetnerlists_onehots_intarray(tweets_spacy_test,spacy_vocab)
tweets_spacy_vocabvec_test = get_tweetnerlists_vocab_vectors(tweets_spacy_test,spacy_vocab,size=300)
tweets_spacy_wordohsvecs_test = get_tweets_as_wordOhs(tweets_spacy_test,spacy_vocab,size=300)

# embedding_matrix_NER_spacy = load_embedding_weights(spacy_vocab,glovepath='data\glove.6B.50d.txt',savepath="data\embedding_matrix_NER_spacey.pkl")
# embedding_matrix_NER_spacy_twitter = load_embedding_weights(spacy_vocab,glovepath='data\glove.6B.50d.txt',savepath="data\embedding_matrix_NER_spacey_twt.pkl")

# from model_builder import *
# print("_____________________NER NN SPACEY_____________________________")
# print("--------------------NER-------------------------------------")
# print("--------------------NER targets not considered-------------------------------------")
# build_train_test_simpleDense_models(tweets_spacy_vocabvec_train,tweets_spacy_vocabvec_test,"ner_vocabvecSPACEY")
# build_train_test_simpleDense_models(tweets_spacy_ohvec_train,tweets_spacy_ohvec_test,"ner_ohvecSPACEY")
# build_train_test_models(tweets_spacy_wordohsvecs_train,tweets_spacy_wordohsvecs_test,"ner_wordOhvecsSPACEY")
# build_train_test_embeddings_models(tweets_spacy_ohvec_train,tweets_spacy_ohvec_test,embedding_matrix_NER_spacy,len(spacy_vocab),"ner_embeddingsSPACEY")
# build_train_test_embeddings_models(tweets_spacy_ohvec_train,tweets_spacy_ohvec_test,embedding_matrix_NER_spacy_twitter,len(spacy_vocab),"ner_twitterembeddingsSPACEY")

# print("--------------------NER targets considered-------------------------------------")
# build_train_test_simpleDense_models_targets(tweets_spacy_vocabvec_train,tweets_spacy_vocabvec_test,"ner_vocabvecSPACEY TARGETS")
# build_train_test_simpleDense_models_targets(tweets_spacy_ohvec_train,tweets_spacy_ohvec_test,"ner_ohvecSPACEY TARGETS")
# build_train_test_models_targets(tweets_spacy_wordohsvecs_train,tweets_spacy_wordohsvecs_test,"ner_wordOhvecsSPACEY TARGETS")
# build_train_test_models_embeddings_targets(tweets_spacy_ohvec_train,tweets_spacy_ohvec_test,embedding_matrix_NER_spacy,len(spacy_vocab),"ner_embeddingsSPACEY  TARGETS")
# build_train_test_models_embeddings_targets(tweets_spacy_ohvec_train,tweets_spacy_ohvec_test,embedding_matrix_NER_spacy_twitter,len(spacy_vocab),"ner_twitterembeddingsSPACEY  TARGETS")

# from classifier_builder import *
# print("_____________________NER CLASSIFIERS SPACEY_____________________________")
# print("--------------------NER-------------------------------------") 
# print("--------------------NER targets not considered-------------------------------------")
# build_and_evaluate_classifier(tweets_spacy_vocabvec_train,tweets_spacy_vocabvec_test,name="ner_vocabvecSPACEY")
# build_and_evaluate_classifier(tweets_spacy_ohvec_train,tweets_spacy_ohvec_test,name="ner_ohvecSPACEY")

# print("--------------------NER targets considered-------------------------------------")
# build_and_evaluate_classifiers_targets(tweets_spacy_vocabvec_train,tweets_spacy_vocabvec_test,name="ner_vocabvecSPACEY TARGETS")
# build_and_evaluate_classifiers_targets(tweets_spacy_ohvec_train,tweets_spacy_ohvec_test,name="ner_ohvecSPACEY TARGETS")


# from model_builder import *
# print("_____________________NER NN_____________________________")
# print("--------------------NER-------------------------------------")
# print("--------------------NER targets not considered-------------------------------------")
# build_train_test_simpleDense_models(tweets_ner_vocabohvec_train,tweets_ner_vocabohvec_test,"ner_vocabvec")
# build_train_test_simpleDense_models(tweets_ner_ohvec_train,tweets_ner_ohvec_test,"ner_ohvec")
# build_train_test_models(tweets_ner_wordsohvecs_train,tweets_ner_wordsohvecs_test,"ner_wordOhvecs")
# build_train_test_embeddings_models(tweets_ner_ohvec_train,tweets_ner_ohvec_test,embedding_matrix_ner,len(vocab_nns),"ner_embeddings")
# build_train_test_embeddings_models(tweets_ner_ohvec_train,tweets_ner_ohvec_test,embedding_matrix_ner_twitter,len(vocab_nns),"ner_twitterembeddings")

# print("--------------------NER targets considered-------------------------------------")
# build_train_test_simpleDense_models_targets(tweets_ner_vocabohvec_train,tweets_ner_vocabohvec_test,"ner_vocabvec TARGETS")
# build_train_test_simpleDense_models_targets(tweets_ner_ohvec_train,tweets_ner_ohvec_test,"ner_ohvec TARGETS")
# build_train_test_models_targets(tweets_ner_wordsohvecs_train,tweets_ner_wordsohvecs_test,"ner_wordOhvecs TARGETS")
# build_train_test_models_embeddings_targets(tweets_ner_ohvec_train,tweets_ner_ohvec_test,embedding_matrix_ner,len(vocab_nns),"ner_embeddings  TARGETS")
# build_train_test_models_embeddings_targets(tweets_ner_ohvec_train,tweets_ner_ohvec_test,embedding_matrix_ner_twitter,len(vocab_nns),"ner_twitterembeddings  TARGETS")

# from classifier_builder import *
# print("_____________________NER CLASSIFIERS_____________________________")
# print("--------------------NER-------------------------------------")
# print("--------------------NER targets not considered-------------------------------------")
# build_and_evaluate_classifier(tweets_ner_vocabohvec_train,tweets_ner_vocabohvec_test,name="ner_vocabvec")
# build_and_evaluate_classifier(tweets_ner_ohvec_train,tweets_ner_ohvec_test,name="ner_ohvec")

# print("--------------------NER targets considered-------------------------------------")
# build_and_evaluate_classifiers_targets(tweets_ner_vocabohvec_train,tweets_ner_vocabohvec_test,name="ner_vocabvec TARGETS")
# build_and_evaluate_classifiers_targets(tweets_ner_ohvec_train,tweets_ner_ohvec_test,name="ner_ohvec TARGETS")
