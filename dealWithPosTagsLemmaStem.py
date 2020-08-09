from readDataFromFile import tweets_train, tweets_test, tokenize, targets_test, targets_train
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer() # lemmatizer.lemmatize(word) # default POS TAG: NOUN / lemmatizer.lemmatize(word, pos=POS_TAG)
stemmer = PorterStemmer() #stemmer.stem(word)

def get_tweets_as_lemma_lists(tweets,postagged = False):
    tweets_lemmas = []
    for tweet in tweets:
        words = tokenize(tweet)
        tags = pos_tag(words)
        tweet_lemmas = []
        for (word,tag) in zip(words,tags):
            if postagged:
                tweet_lemmas.append(lemmatizer.lemmatize(word,pos=get_wordnet_pos(tag[1])))
            else:
                tweet_lemmas.append(lemmatizer.lemmatize(word))
        tweets_lemmas.append(tweet_lemmas)
    return tweets_lemmas

def get_tweets_as_stem_lists(tweets):
    tweets_stems = []
    for tweet in tweets:
        words = tokenize(tweet)
        tweet_stems = []
        for word in words:
            tweet_stems.append(stemmer.stem(word))
        tweets_stems.append(tweet_stems)
    return tweets_stems

def get_lemma_vocabulary(lematized_tweets):
    dictionary = {}
    for tweet in lematized_tweets:
        for lemma in tweet:
            if lemma in dictionary:
                dictionary[lemma] = dictionary[lemma] + 1
            else:
                dictionary[lemma] = 1
    sorteddict = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    return list(sorteddict.keys())

def get_stem_vocabulary(stemmed_tweets):
    dictionary = {}
    for tweet in stemmed_tweets:
        for stem in tweet:
            if stem in dictionary:
                dictionary[stem] = dictionary[stem] + 1
            else:
                dictionary[stem] = 1
    sorteddict = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    return list(sorteddict.keys())

def get_tweets_as_intarrays_given_vocab(tweets, vocab, padtosize = 100):
    tweets_intarrays = []
    for tweet in tweets:
        tweet_intarray = []
        for word in tweet:
            if word in vocab and len(tweet_intarray) < padtosize:
                tweet_intarray.append(vocab.index(word)+1)
            elif word not in vocab and len(tweet_intarray) < padtosize:
                tweet_intarray.append(0)
        while len(tweet_intarray) < padtosize:
            tweet_intarray.append(0)
        tweets_intarrays.append(tweet_intarray)
    return tweets_intarrays
        
def get_tweets_as_intarrays_given_vector(tweets, vector, vectorlen=100):
    tweets_intarrays = []
    for tweet in tweets:
        tweet_intarray = []
        for word in vector[0:vectorlen]:
            if word in tweet:
                tweet_intarray.append(tweet.count(word))
            else:
                tweet_intarray.append(0)
        tweets_intarrays.append(tweet_intarray)
    return tweets_intarrays
  
tweets_lemmas_train = get_tweets_as_lemma_lists(tweets_train,postagged=True)
tweets_stems_train = get_tweets_as_stem_lists(tweets_train)
tweets_lemmas_test = get_tweets_as_lemma_lists(tweets_test,postagged=True)
tweets_stems_test = get_tweets_as_stem_lists(tweets_test)

lemma_vocabulary = get_lemma_vocabulary(tweets_lemmas_train)
stem_vocabulary = get_stem_vocabulary(tweets_stems_train)

# print("lemma_vocabulary size:"+str(len(lemma_vocabulary)))
# print("stem_vocabulary size:"+str(len(stem_vocabulary)))

tweets_lemm_onehot_vocab_train = get_tweets_as_intarrays_given_vocab(tweets_lemmas_train, lemma_vocabulary,padtosize=30)
tweets_lemm_onehot_vector_train = get_tweets_as_intarrays_given_vector(tweets_lemmas_train, lemma_vocabulary,vectorlen = 100)
tweets_lemm_onehot_vocab_test = get_tweets_as_intarrays_given_vocab(tweets_lemmas_test, lemma_vocabulary,padtosize=30)
tweets_lemm_onehot_vector_test = get_tweets_as_intarrays_given_vector(tweets_lemmas_test, lemma_vocabulary,vectorlen = 100)

tweets_stem_onehot_vocab_train = get_tweets_as_intarrays_given_vocab(tweets_stems_train, stem_vocabulary,padtosize=30)
tweets_stem_onehot_vector_train = get_tweets_as_intarrays_given_vector(tweets_stems_train, stem_vocabulary,vectorlen = 100)
tweets_stem_onehot_vocab_test = get_tweets_as_intarrays_given_vocab(tweets_stems_test, stem_vocabulary,padtosize=30)
tweets_stem_onehot_vector_test = get_tweets_as_intarrays_given_vector(tweets_stems_test, stem_vocabulary,vectorlen = 100)


# from utils import get_tweets_as_wordOhs

# tweets_lemma_wordOhs_train = get_tweets_as_wordOhs(tweets_lemmas_train,lemma_vocabulary,size=300,padto=20)
# tweets_lemma_wordOhs_test = get_tweets_as_wordOhs(tweets_lemmas_test,lemma_vocabulary,size=300,padto=20)

# tweets_stem_wordOhs_train = get_tweets_as_wordOhs(tweets_stems_train,stem_vocabulary,size=300,padto=20)
# tweets_stem_wordOhs_test = get_tweets_as_wordOhs(tweets_stems_test,stem_vocabulary,size=300,padto=20)

# from model_builder import *
# print("________________________LEMMA/STEM NN________________________")
# print("----------------LEMMA STEM-----------------------")
# print("-----------LEMMA/STEM targets not considered-----------------")
# build_train_test_simpleDense_models(tweets_lemm_onehot_vocab_train,tweets_lemm_onehot_vocab_test,"lemma_ohvocab")
# build_train_test_simpleDense_models(tweets_lemm_onehot_vector_train,tweets_lemm_onehot_vector_test,"lemma_vec")
# build_train_test_simpleDense_models(tweets_stem_onehot_vocab_train,tweets_stem_onehot_vocab_test,"stem_ohvocab")
# build_train_test_simpleDense_models(tweets_stem_onehot_vector_train,tweets_stem_onehot_vector_test,"stem_vec")
# build_train_test_models(tweets_lemma_wordOhs_train,tweets_lemma_wordOhs_test,"lemma_wordOhs")
# build_train_test_models(tweets_stem_wordOhs_train,tweets_stem_wordOhs_test,"stem_wordOhs")

# print("-----------LEMMA/STEM targets considered-----------------")
# build_train_test_simpleDense_models_targets(tweets_lemm_onehot_vocab_train,tweets_lemm_onehot_vocab_test,"lemma_ohvocab TARGETS")
# build_train_test_simpleDense_models_targets(tweets_lemm_onehot_vector_train,tweets_lemm_onehot_vector_test,"lemma_vec TARGETS")
# build_train_test_simpleDense_models_targets(tweets_stem_onehot_vocab_train,tweets_stem_onehot_vocab_test,"stem_ohvocab TARGETS")
# build_train_test_simpleDense_models_targets(tweets_stem_onehot_vector_train,tweets_stem_onehot_vector_test,"stem_vec TARGETS")
# build_train_test_models_targets(tweets_lemma_wordOhs_train,tweets_lemma_wordOhs_test,"lemma_wordOhs TARGETS")
# build_train_test_models_targets(tweets_stem_wordOhs_train,tweets_stem_wordOhs_test,"stem_wordOhs TARGETS")

# from classifier_builder import *

# print("________________________LEMMA/STEM CLASSIFIERS________________________")
# print("----------------LEMMA STEM targets not considered-----------------------")
# build_and_evaluate_classifier(tweets_lemm_onehot_vocab_train,tweets_lemm_onehot_vocab_test,name="lemma_ohvocab")
# build_and_evaluate_classifier(tweets_lemm_onehot_vector_train,tweets_lemm_onehot_vector_test,name="lemma_vec")
# build_and_evaluate_classifier(tweets_stem_onehot_vocab_train,tweets_stem_onehot_vocab_test,name="stem_ohvocab")
# build_and_evaluate_classifier(tweets_stem_onehot_vector_train,tweets_stem_onehot_vector_test,name="stem_vec")

# print("----------------LEMMA STEM targets considered-----------------------")
# build_and_evaluate_classifiers_targets(tweets_lemm_onehot_vocab_train,tweets_lemm_onehot_vocab_test,name="lemma_ohvocab TARGETS")
# build_and_evaluate_classifiers_targets(tweets_lemm_onehot_vector_train,tweets_lemm_onehot_vector_test,name="lemma_vec TARGETS")
# build_and_evaluate_classifiers_targets(tweets_stem_onehot_vocab_train,tweets_stem_onehot_vocab_test,name="stem_ohvocab TARGETS")
# build_and_evaluate_classifiers_targets(tweets_stem_onehot_vector_train,tweets_stem_onehot_vector_test,name="stem_vec TARGETS")