from preprocess import *
from representation import *
from representation import embeddings_GLOBAL, embeddings_twitter_GLOBAL
import numpy as np

def get_ngram_sorted_dictionary_vocab(tweets_ngr):
    dictionary = {}
    for tweet in tweets_ngr:
        for ngram in tweet:
            if ngram in dictionary:
                dictionary[ngram] += 1
            else:
                dictionary[ngram] = 1
    sorteddict_asc = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    return (sorteddict_asc, list(sorteddict_asc))

def get_tweets_as_ngrams(tweets_train,tweets_test,n):
    tweets_ngrams_train = []
    for tweet in tweets_train:
        tweet_ngrams = []
        for i in range(len(tweet)-n+1):
            ngram = []
            for j in range(0,n):
                ngram.append(tweet[i+j])             
            tweet_ngrams.append(tuple(ngram))
        tweets_ngrams_train.append(tweet_ngrams)
    tweets_ngrams_test = []
    for tweet in tweets_test:
        tweet_ngrams = []
        for i in range(len(tweet)-n+1):
            ngram = []
            for j in range(0,n):
                ngram.append(tweet[i+j])             
            tweet_ngrams.append(tuple(ngram))
        tweets_ngrams_test.append(tweet_ngrams)
    return (tweets_ngrams_train,tweets_ngrams_test)


def get_ngrams_embedded(ngrams_train,ngrams_test,embeddings):
    ngrams_embedded_train = []
    ngrams_embedded_test = []
    ngrams_embedded_summary_train = []
    ngrams_embedded_summary_test = []

    for tweet_ngrams in ngrams_train:
        ngram_embed = []
        for ngram in tweet_ngrams:
            for part in ngram:
                if part in embeddings:
                    ngram_embed.append(embeddings[part])
                else:
                    ngram_embed.append(np.random.standard_normal(50))
        ngrams_embedded_train.append(ngram_embed)
        ngrams_embedded_summary_train.append(np.average(ngram_embed,axis=0))

    for tweet_ngrams in ngrams_test:
        ngram_embed = []
        for ngram in tweet_ngrams:
            for part in ngram:
                if part in embeddings:
                    ngram_embed.append(embeddings[part])
                else:
                    ngram_embed.append(np.random.standard_normal(50))
        ngrams_embedded_test.append(ngram_embed)
        ngrams_embedded_summary_test.append(np.average(ngram_embed,axis=0))
    return (ngrams_embedded_train,ngrams_embedded_test,ngrams_embedded_summary_train,ngrams_embedded_summary_test)


words_2grams_train, words_2grams_test = get_tweets_as_ngrams(words_train,words_test,2)
words_3grams_train, words_3grams_test = get_tweets_as_ngrams(words_train,words_test,3)
words_4grams_train, words_4grams_test = get_tweets_as_ngrams(words_train,words_test,4)

words_2grams_vocabulary,avg_w2 = get_sorted_dictionary_vocab(words_2grams_train)
words_3grams_vocabulary,avg_w3 = get_sorted_dictionary_vocab(words_3grams_train)
words_4grams_vocabulary,avg_w4 = get_sorted_dictionary_vocab(words_4grams_train)

words_2grams_embedded_train,words_2grams_embedded_test,words_2grams_embedded_summary_train,words_2grams_embedded_summary_test = get_ngrams_embedded(words_2grams_train, words_2grams_test,embeddings_GLOBAL)
words_2grams_embedded_TWT_train,words_2grams_embedded_TWT_test,words_2grams_embedded_TWT_summary_train,words_2grams_embedded_TWT_summary_test = get_ngrams_embedded(words_2grams_train, words_2grams_test,embeddings_twitter_GLOBAL)
words_3grams_embedded_train,words_3grams_embedded_test,words_3grams_embedded_summary_train,words_3grams_embedded_summary_test = get_ngrams_embedded(words_3grams_train, words_3grams_test,embeddings_GLOBAL)
words_3grams_embedded_TWT_train,words_3grams_embedded_TWT_test,words_3grams_embedded_TWT_summary_train,words_3grams_embedded_TWT_summary_test = get_ngrams_embedded(words_3grams_train, words_3grams_test,embeddings_twitter_GLOBAL)
words_4grams_embedded_train,words_4grams_embedded_test,words_4grams_embedded_summary_train,words_4grams_embedded_summary_test = get_ngrams_embedded(words_4grams_train, words_4grams_test,embeddings_GLOBAL)
words_4grams_embedded_TWT_train,words_4grams_embedded_TWT_test,words_4grams_embedded_TWT_summary_train,words_4grams_embedded_TWT_summary_test = get_ngrams_embedded(words_4grams_train, words_4grams_test,embeddings_twitter_GLOBAL)

##########################################################################################################################################################
lemmas_2grams_train, lemmas_2grams_test = get_tweets_as_ngrams(lemmas_train,lemmas_test,2)
lemmas_3grams_train, lemmas_3grams_test = get_tweets_as_ngrams(lemmas_train,lemmas_test,3)
lemmas_4grams_train, lemmas_4grams_test = get_tweets_as_ngrams(lemmas_train,lemmas_test,4)

lemmas_2grams_vocabulary,avg_l2 = get_sorted_dictionary_vocab(lemmas_2grams_train)
lemmas_3grams_vocabulary,avg_l3 = get_sorted_dictionary_vocab(lemmas_3grams_train)
lemmas_4grams_vocabulary,avg_l4 = get_sorted_dictionary_vocab(lemmas_4grams_train)

lemmas_2grams_embedded_train,lemmas_2grams_embedded_test,lemmas_2grams_embedded_summary_train,lemmas_2grams_embedded_summary_test = get_ngrams_embedded(lemmas_2grams_train, lemmas_2grams_test,embeddings_GLOBAL)
lemmas_2grams_embedded_TWT_train,lemmas_2grams_embedded_TWT_test,lemmas_2grams_embedded_TWT_summary_train,lemmas_2grams_embedded_TWT_summary_test = get_ngrams_embedded(lemmas_2grams_train, lemmas_2grams_test,embeddings_twitter_GLOBAL)
lemmas_3grams_embedded_train,lemmas_3grams_embedded_test,lemmas_3grams_embedded_summary_train,lemmas_3grams_embedded_summary_test = get_ngrams_embedded(lemmas_3grams_train, lemmas_3grams_test,embeddings_GLOBAL)
lemmas_3grams_embedded_TWT_train,lemmas_3grams_embedded_TWT_test,lemmas_3grams_embedded_TWT_summary_train,lemmas_3grams_embedded_TWT_summary_test = get_ngrams_embedded(lemmas_3grams_train, lemmas_3grams_test,embeddings_twitter_GLOBAL)
lemmas_4grams_embedded_train,lemmas_4grams_embedded_test,lemmas_4grams_embedded_summary_train,lemmas_4grams_embedded_summary_test = get_ngrams_embedded(lemmas_4grams_train, lemmas_4grams_test,embeddings_GLOBAL)
lemmas_4grams_embedded_TWT_train,lemmas_4grams_embedded_TWT_test,lemmas_4grams_embedded_TWT_summary_train,lemmas_4grams_embedded_TWT_summary_test = get_ngrams_embedded(lemmas_4grams_train, lemmas_4grams_test,embeddings_twitter_GLOBAL)

##########################################################################################################################################################

stems_2grams_train, stems_2grams_test = get_tweets_as_ngrams(stems_train,stems_test,2)
stems_3grams_train, stems_3grams_test = get_tweets_as_ngrams(stems_train,stems_test,3)
stems_4grams_train, stems_4grams_test = get_tweets_as_ngrams(stems_train,stems_test,4)

stems_2grams_vocabulary,avg_s2 = get_sorted_dictionary_vocab(stems_2grams_train)
stems_3grams_vocabulary,avg_s3 = get_sorted_dictionary_vocab(stems_3grams_train)
stems_4grams_vocabulary,avg_s4 = get_sorted_dictionary_vocab(stems_4grams_train)

stems_2grams_embedded_train,stems_2grams_embedded_test,stems_2grams_embedded_summary_train,stems_2grams_embedded_summary_test = get_ngrams_embedded(stems_2grams_train, stems_2grams_test,embeddings_GLOBAL)
stems_2grams_embedded_TWT_train,stems_2grams_embedded_TWT_test,stems_2grams_embedded_TWT_summary_train,stems_2grams_embedded_TWT_summary_test = get_ngrams_embedded(stems_2grams_train, stems_2grams_test,embeddings_twitter_GLOBAL)
stems_3grams_embedded_train,stems_3grams_embedded_test,stems_3grams_embedded_summary_train,stems_3grams_embedded_summary_test = get_ngrams_embedded(stems_3grams_train, stems_3grams_test,embeddings_GLOBAL)
stems_3grams_embedded_TWT_train,stems_3grams_embedded_TWT_test,stems_3grams_embedded_TWT_summary_train,stems_3grams_embedded_TWT_summary_test = get_ngrams_embedded(stems_3grams_train, stems_3grams_test,embeddings_twitter_GLOBAL)
stems_4grams_embedded_train,stems_4grams_embedded_test,stems_4grams_embedded_summary_train,stems_4grams_embedded_summary_test = get_ngrams_embedded(stems_4grams_train, stems_4grams_test,embeddings_GLOBAL)
stems_4grams_embedded_TWT_train,stems_4grams_embedded_TWT_test,stems_4grams_embedded_TWT_summary_train,stems_4grams_embedded_TWT_summary_test = get_ngrams_embedded(stems_4grams_train, stems_4grams_test,embeddings_twitter_GLOBAL)


##########################################################################################################################################################

nouns_2grams_train, nouns_2grams_test = get_tweets_as_ngrams(nouns_train,nouns_test,2)
nouns_3grams_train, nouns_3grams_test = get_tweets_as_ngrams(nouns_train,nouns_test,3)
nouns_4grams_train, nouns_4grams_test = get_tweets_as_ngrams(nouns_train,nouns_test,4)

nouns_2grams_vocabulary,avg_no2 = get_sorted_dictionary_vocab(nouns_2grams_train)
nouns_3grams_vocabulary,avg_no3 = get_sorted_dictionary_vocab(nouns_3grams_train)
nouns_4grams_vocabulary,avg_no4 = get_sorted_dictionary_vocab(nouns_4grams_train)

nouns_2grams_embedded_train,nouns_2grams_embedded_test,nouns_2grams_embedded_summary_train,nouns_2grams_embedded_summary_test = get_ngrams_embedded(nouns_2grams_train, nouns_2grams_test,embeddings_GLOBAL)
nouns_2grams_embedded_TWT_train,nouns_2grams_embedded_TWT_test,nouns_2grams_embedded_TWT_summary_train,nouns_2grams_embedded_TWT_summary_test = get_ngrams_embedded(nouns_2grams_train, nouns_2grams_test,embeddings_twitter_GLOBAL)
nouns_3grams_embedded_train,nouns_3grams_embedded_test,nouns_3grams_embedded_summary_train,nouns_3grams_embedded_summary_test = get_ngrams_embedded(nouns_3grams_train, nouns_3grams_test,embeddings_GLOBAL)
nouns_3grams_embedded_TWT_train,nouns_3grams_embedded_TWT_test,nouns_3grams_embedded_TWT_summary_train,nouns_3grams_embedded_TWT_summary_test = get_ngrams_embedded(nouns_3grams_train, nouns_3grams_test,embeddings_twitter_GLOBAL)
nouns_4grams_embedded_train,nouns_4grams_embedded_test,nouns_4grams_embedded_summary_train,nouns_4grams_embedded_summary_test = get_ngrams_embedded(nouns_4grams_train, nouns_4grams_test,embeddings_GLOBAL)
nouns_4grams_embedded_TWT_train,nouns_4grams_embedded_TWT_test,nouns_4grams_embedded_TWT_summary_train,nouns_4grams_embedded_TWT_summary_test = get_ngrams_embedded(nouns_4grams_train, nouns_4grams_test,embeddings_twitter_GLOBAL)


##########################################################################################################################################################

ners_2grams_train, ners_2grams_test = get_tweets_as_ngrams(ners_train,ners_test,2)
ners_3grams_train, ners_3grams_test = get_tweets_as_ngrams(ners_train,ners_test,3)
ners_4grams_train, ners_4grams_test = get_tweets_as_ngrams(ners_train,ners_test,4)

ners_2grams_vocabulary,avg_ner2 = get_sorted_dictionary_vocab(ners_2grams_train)
ners_3grams_vocabulary,avg_ner3 = get_sorted_dictionary_vocab(ners_3grams_train)
ners_4grams_vocabulary,avg_ner4 = get_sorted_dictionary_vocab(ners_4grams_train)

ners_2grams_embedded_train,ners_2grams_embedded_test,ners_2grams_embedded_summary_train,ners_2grams_embedded_summary_test = get_ngrams_embedded(ners_2grams_train, ners_2grams_test,embeddings_GLOBAL)
ners_2grams_embedded_TWT_train,ners_2grams_embedded_TWT_test,ners_2grams_embedded_TWT_summary_train,ners_2grams_embedded_TWT_summary_test = get_ngrams_embedded(ners_2grams_train, ners_2grams_test,embeddings_twitter_GLOBAL)
ners_3grams_embedded_train,ners_3grams_embedded_test,ners_3grams_embedded_summary_train,ners_3grams_embedded_summary_test = get_ngrams_embedded(ners_3grams_train, ners_3grams_test,embeddings_GLOBAL)
ners_3grams_embedded_TWT_train,ners_3grams_embedded_TWT_test,ners_3grams_embedded_TWT_summary_train,ners_3grams_embedded_TWT_summary_test = get_ngrams_embedded(ners_3grams_train, ners_3grams_test,embeddings_twitter_GLOBAL)
ners_4grams_embedded_train,ners_4grams_embedded_test,ners_4grams_embedded_summary_train,ners_4grams_embedded_summary_test = get_ngrams_embedded(ners_4grams_train, ners_4grams_test,embeddings_GLOBAL)
ners_4grams_embedded_TWT_train,ners_4grams_embedded_TWT_test,ners_4grams_embedded_TWT_summary_train,ners_4grams_embedded_TWT_summary_test = get_ngrams_embedded(ners_4grams_train, ners_4grams_test,embeddings_twitter_GLOBAL)

##########################################################################################################################################################
words_2grams_freq_vec_train,words_2grams_freq_vec_test = get_frequency_vectors(words_2grams_train,words_2grams_test, words_2grams_vocabulary,avg_w2)
words_2grams_vocab_vec_train,words_2grams_vocab_vec_test = get_vocabulary_vectors(words_2grams_train,words_2grams_test, words_2grams_vocabulary)
words_2grams_tokenohs_train,words_2grams_tokenohs_test = get_token_onehot_vectors(words_2grams_train,words_2grams_test, words_2grams_vocabulary)
words_3grams_freq_vec_train,words_3grams_freq_vec_test = get_frequency_vectors(words_3grams_train,words_3grams_test, words_3grams_vocabulary,avg_w3)
words_3grams_vocab_vec_train,words_3grams_vocab_vec_test = get_vocabulary_vectors(words_3grams_train,words_3grams_test, words_3grams_vocabulary)
words_3grams_tokenohs_train,words_3grams_tokenohs_test = get_token_onehot_vectors(words_3grams_train,words_3grams_test, words_3grams_vocabulary)
words_4grams_freq_vec_train,words_4grams_freq_vec_test = get_frequency_vectors(words_4grams_train,words_4grams_test, words_4grams_vocabulary,avg_w4)
words_4grams_vocab_vec_train,words_4grams_vocab_vec_test = get_vocabulary_vectors(words_4grams_train,words_4grams_test, words_4grams_vocabulary)
words_4grams_tokenohs_train,words_4grams_tokenohs_test = get_token_onehot_vectors(words_4grams_train,words_4grams_test, words_4grams_vocabulary)

lemmas_2grams_freq_vec_train,lemmas_2grams_freq_vec_test = get_frequency_vectors(lemmas_2grams_train,lemmas_2grams_test, lemmas_2grams_vocabulary,avg_l2)
lemmas_2grams_vocab_vec_train,lemmas_2grams_vocab_vec_test = get_vocabulary_vectors(lemmas_2grams_train,lemmas_2grams_test, lemmas_2grams_vocabulary)
lemmas_2grams_tokenohs_train,lemmas_2grams_tokenohs_test = get_token_onehot_vectors(lemmas_2grams_train,lemmas_2grams_test, lemmas_2grams_vocabulary)
lemmas_3grams_freq_vec_train,lemmas_3grams_freq_vec_test = get_frequency_vectors(lemmas_3grams_train,lemmas_3grams_test, lemmas_3grams_vocabulary,avg_l3)
lemmas_3grams_vocab_vec_train,lemmas_3grams_vocab_vec_test = get_vocabulary_vectors(lemmas_3grams_train,lemmas_3grams_test, lemmas_3grams_vocabulary)
lemmas_3grams_tokenohs_train,lemmas_3grams_tokenohs_test = get_token_onehot_vectors(lemmas_3grams_train,lemmas_3grams_test, lemmas_3grams_vocabulary)
lemmas_4grams_freq_vec_train,lemmas_4grams_freq_vec_test = get_frequency_vectors(lemmas_4grams_train,lemmas_4grams_test, lemmas_4grams_vocabulary,avg_l4)
lemmas_4grams_vocab_vec_train,lemmas_4grams_vocab_vec_test = get_vocabulary_vectors(lemmas_4grams_train,lemmas_4grams_test, lemmas_4grams_vocabulary)
lemmas_4grams_tokenohs_train,lemmas_4grams_tokenohs_test = get_token_onehot_vectors(lemmas_4grams_train,lemmas_4grams_test, lemmas_4grams_vocabulary)

stems_2grams_freq_vec_train,stems_2grams_freq_vec_test = get_frequency_vectors(stems_2grams_train,stems_2grams_test, stems_2grams_vocabulary,avg_s2)
stems_2grams_vocab_vec_train,stems_2grams_vocab_vec_test = get_vocabulary_vectors(stems_2grams_train,stems_2grams_test, stems_2grams_vocabulary)
stems_2grams_tokenohs_train,stems_2grams_tokenohs_test = get_token_onehot_vectors(stems_2grams_train,stems_2grams_test, stems_2grams_vocabulary)
stems_3grams_freq_vec_train,stems_3grams_freq_vec_test = get_frequency_vectors(stems_3grams_train,stems_3grams_test, stems_3grams_vocabulary,avg_s3)
stems_3grams_vocab_vec_train,stems_3grams_vocab_vec_test = get_vocabulary_vectors(stems_3grams_train,stems_3grams_test, stems_3grams_vocabulary)
stems_3grams_tokenohs_train,stems_3grams_tokenohs_test = get_token_onehot_vectors(stems_3grams_train,stems_3grams_test, stems_3grams_vocabulary)
stems_4grams_freq_vec_train,stems_4grams_freq_vec_test = get_frequency_vectors(stems_4grams_train,stems_4grams_test, stems_4grams_vocabulary,avg_s4)
stems_4grams_vocab_vec_train,stems_4grams_vocab_vec_test = get_vocabulary_vectors(stems_4grams_train,stems_4grams_test, stems_4grams_vocabulary)
stems_4grams_tokenohs_train,stems_4grams_tokenohs_test = get_token_onehot_vectors(stems_4grams_train,stems_4grams_test, stems_4grams_vocabulary)

nouns_2grams_freq_vec_train,nouns_2grams_freq_vec_test = get_frequency_vectors(nouns_2grams_train,nouns_2grams_test, nouns_2grams_vocabulary,avg_no2)
nouns_2grams_vocab_vec_train,nouns_2grams_vocab_vec_test = get_vocabulary_vectors(nouns_2grams_train,nouns_2grams_test, nouns_2grams_vocabulary)
nouns_2grams_tokenohs_train,nouns_2grams_tokenohs_test = get_token_onehot_vectors(nouns_2grams_train,nouns_2grams_test, nouns_2grams_vocabulary)
nouns_3grams_freq_vec_train,nouns_3grams_freq_vec_test = get_frequency_vectors(nouns_3grams_train,nouns_3grams_test, nouns_3grams_vocabulary,avg_no3)
nouns_3grams_vocab_vec_train,nouns_3grams_vocab_vec_test = get_vocabulary_vectors(nouns_3grams_train,nouns_3grams_test, nouns_3grams_vocabulary)
nouns_3grams_tokenohs_train,nouns_3grams_tokenohs_test = get_token_onehot_vectors(nouns_3grams_train,nouns_3grams_test, nouns_3grams_vocabulary)
nouns_4grams_freq_vec_train,nouns_4grams_freq_vec_test = get_frequency_vectors(nouns_4grams_train,nouns_4grams_test, nouns_4grams_vocabulary,avg_no4)
nouns_4grams_vocab_vec_train,nouns_4grams_vocab_vec_test = get_vocabulary_vectors(nouns_4grams_train,nouns_4grams_test, nouns_4grams_vocabulary)
nouns_4grams_tokenohs_train,nouns_4grams_tokenohs_test = get_token_onehot_vectors(nouns_4grams_train,nouns_4grams_test, nouns_4grams_vocabulary)

ners_2grams_freq_vec_train,ners_2grams_freq_vec_test = get_frequency_vectors(ners_2grams_train,ners_2grams_test, ners_2grams_vocabulary,avg_ner2)
ners_2grams_vocab_vec_train,ners_2grams_vocab_vec_test = get_vocabulary_vectors(ners_2grams_train,ners_2grams_test, ners_2grams_vocabulary)
ners_2grams_tokenohs_train,ners_2grams_tokenohs_test = get_token_onehot_vectors(ners_2grams_train,ners_2grams_test, ners_2grams_vocabulary)
ners_3grams_freq_vec_train,ners_3grams_freq_vec_test = get_frequency_vectors(ners_3grams_train,ners_3grams_test, ners_3grams_vocabulary,avg_ner3)
ners_3grams_vocab_vec_train,ners_3grams_vocab_vec_test = get_vocabulary_vectors(ners_3grams_train,ners_3grams_test, ners_3grams_vocabulary)
ners_3grams_tokenohs_train,ners_3grams_tokenohs_test = get_token_onehot_vectors(ners_3grams_train,ners_3grams_test, ners_3grams_vocabulary)
ners_4grams_freq_vec_train,ners_4grams_freq_vec_test = get_frequency_vectors(ners_4grams_train,ners_4grams_test, ners_4grams_vocabulary,avg_ner4)
ners_4grams_vocab_vec_train,ners_4grams_vocab_vec_test = get_vocabulary_vectors(ners_4grams_train,ners_4grams_test, ners_4grams_vocabulary)
ners_4grams_tokenohs_train,ners_4grams_tokenohs_test = get_token_onehot_vectors(ners_4grams_train,ners_4grams_test, ners_4grams_vocabulary)