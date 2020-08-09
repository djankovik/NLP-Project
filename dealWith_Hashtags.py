from readDataFromFile import tweets_train,tweets_test, stances_onehot_train, stances_onehot_test
import re
import numpy as np
import pickle
import os

def get_tweethashtags_oh_given_vocab_vec(tweets_hashtags_lists,vocabulary,size=1900):
    tweet_vectors = []
    for tweet in tweets_hashtags_lists:
        tweet_vector = []
        for hashtag in vocabulary[0:size]:
            if hashtag in tweet:
                tweet_vector.append(1)
            else:
                tweet_vector.append(0)
        tweet_vectors.append(tweet_vector)
    return tweet_vectors

def get_tweethashtags_onehot_intarrays(tweets_hashtags_lists,vocabulary, padtosize=10,padwith=0):
    tweets_onehots = []
    for tweet in tweets_hashtags_lists:
        tweet_onehot = []
        for hashtag in tweet:
            if len(tweet_onehot) < padtosize:
                if hashtag in vocabulary:
                    tweet_onehot.append(vocabulary.index(hashtag))
                else:
                    tweet_onehot.append(0)
        while len(tweet_onehot) < padtosize:
            tweet_onehot.append(0)
        tweets_onehots.append(tweet_onehot)
    return tweets_onehots

def get_stats_hashtags_per_tweet(tweets):
    cnt = 0.0
    max = -1
    for tweet in tweets:
        twtcnt = tweet.count('#')
        cnt += twtcnt
        if max < twtcnt:
            max = twtcnt
    print('average hashtags per tweet: '+str(cnt/len(tweets))+" | max hashtags per tweet: "+str(max))

def get_hashtags_from_tweets(tweets):
    tweets_hashtags = []
    hashtags_vocab = {}
    for tweet in tweets:
        parts = re.split(r'[.;,?!\[\]\(\)\s]+', tweet)
        tweet_hashtags = []
        for part in parts:
            if part.find('#') != -1:
                part_pure = part.lower().replace('#','')
                tweet_hashtags.append(part_pure)
                if part_pure in hashtags_vocab:
                    hashtags_vocab[part_pure] = hashtags_vocab[part_pure]
                else:
                    hashtags_vocab[part_pure] = 1
        tweets_hashtags.append(tweet_hashtags)
    hashtagssortedvocab = {k: v for k, v in sorted(hashtags_vocab.items(), key=lambda item: item[1], reverse=True)}
    return (tweets_hashtags,list(hashtagssortedvocab.keys()))

def get_tweets_as_wordOhs(tweets,vocabulary,size=500,padto=10):
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

tweets_hashtags_train, hashtag_vocabulary = get_hashtags_from_tweets(tweets_train)
tweets_hashtags_test, h_v = get_hashtags_from_tweets(tweets_test)

tweets_hashtags_vectors_train = get_tweethashtags_oh_given_vocab_vec(tweets_hashtags_train,hashtag_vocabulary)
tweets_hashtags_vectors_test = get_tweethashtags_oh_given_vocab_vec(tweets_hashtags_test,hashtag_vocabulary)

tweets_hashtags_onehots_train = get_tweethashtags_onehot_intarrays(tweets_hashtags_train,hashtag_vocabulary)
tweets_hashtags_onehots_test = get_tweethashtags_onehot_intarrays(tweets_hashtags_test,hashtag_vocabulary)


# print("hashtag vocabulary size:"+str(len(hashtag_vocabulary)))

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

embedding_matrix_hashtags = load_embedding_weights(hashtag_vocabulary,glovepath='data\glove.6B.50d.txt',savepath="data\embedding_matrix_hash.pkl")
embedding_matrix_hashtags_twitter = load_embedding_weights(hashtag_vocabulary,glovepath='data\glove.twitter.27B.50d.txt',savepath="data\embedding_matrix_hash_twt.pkl")

tweets_hashtags_wordsonehots_train = get_tweets_as_wordOhs(tweets_hashtags_train,hashtag_vocabulary) #each word encoded as OH array
tweets_hashtags_wordsonehots_test = get_tweets_as_wordOhs(tweets_hashtags_test,hashtag_vocabulary) #each word encoded as OH array

# from utils import *
# from model_builder import *

# print("_____________________________HASHTAGS NN_________________________")
# print("--------------------HASHTAGS---------------------------------------")
# print("--------------------HASHTAGS Targets not considered---------------------------------------")

# build_train_test_models(tweets_hashtags_wordsonehots_train,tweets_hashtags_wordsonehots_test,"hashtags_wordsOhs")
# build_train_test_simpleDense_models(tweets_hashtags_vectors_train,tweets_hashtags_vectors_test,"hashtags_vectors")
# build_train_test_simpleDense_models(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,"hashtags_ohs")
# build_train_test_embeddings_models(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,embedding_matrix_hashtags,len(hashtag_vocabulary),"hashtags_ohs+embeddings")
# build_train_test_embeddings_models(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,embedding_matrix_hashtags_twitter,len(hashtag_vocabulary),"hashtags_ohs+tweeter_embeddings")


# print("--------------------HASHTAGS Targets considered---------------------------------------")
# build_train_test_models_targets(tweets_hashtags_wordsonehots_train,tweets_hashtags_wordsonehots_test,"hashtags_wordsOhs TARGETS")
build_train_test_simpleDense_models_targets(tweets_hashtags_vectors_train,tweets_hashtags_vectors_test,"hashtags_vectors TARGETS")
# build_train_test_simpleDense_models_targets(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,"hashtags_ohs TARGETS")
# build_train_test_models_embeddings_targets(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,embedding_matrix_hashtags,len(hashtag_vocabulary),"hashtags_ohs+embeddings TARGETS")
# build_train_test_models_embeddings_targets(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,embedding_matrix_hashtags_twitter,len(hashtag_vocabulary),"hashtags_ohs+twitter_embeddings TARGETS")



# from classifier_builder import *
# print("_____________________________HASHTAGS CLASSIFIERS_________________________")
# print("--------------------HASHTAGS---------------------------------------")
# print("--------------------HASHTAGS Targets not considered---------------------------------------")
# build_and_evaluate_classifier(tweets_hashtags_vectors_train,tweets_hashtags_vectors_test,name="hashtags_vectors")
# build_and_evaluate_classifier(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,name="hashtags_ohs")

# print("--------------------HASHTAGS Targets considered---------------------------------------")
# build_and_evaluate_classifiers_targets(tweets_hashtags_vectors_train,tweets_hashtags_vectors_test,name="hashtags_vectors TARGETS")
# build_and_evaluate_classifiers_targets(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,name="hashtags_ohs TARGETS")
