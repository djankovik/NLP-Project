import numpy as np
import os
import pickle
from preprocess import *

def get_vocabulary_vectors(token_lists_train,token_lists_test,vocabulary,vectorsize=500):
    tweet_vectors_train = []
    for token_list in token_lists_train:
        tweet_vector = []
        for token in vocabulary[0:vectorsize]:
            if token in token_list:
                tweet_vector.append(1)
            else:
                tweet_vector.append(0)
        tweet_vectors_train.append(tweet_vector)

    tweet_vectors_test = []
    for token_list in token_lists_test:
        tweet_vector = []
        for token in vocabulary[0:vectorsize]:
            if token in token_list:
                tweet_vector.append(1)
            else:
                tweet_vector.append(0)
        tweet_vectors_test.append(tweet_vector)
    return tweet_vectors_train,tweet_vectors_test

def get_frequency_vectors(token_lists_train,token_lists_test,vocabulary, padtosize=10,padwith=0):
    tweets_onehots_train = []
    for token_list in token_lists_train:
        tweet_onehot = []
        for token in token_list:
            if len(tweet_onehot) < padtosize:
                if token in vocabulary:
                    tweet_onehot.append(vocabulary.index(token))
                else:
                    tweet_onehot.append(0)
        while len(tweet_onehot) < padtosize:
            tweet_onehot.append(0)
        tweets_onehots_train.append(tweet_onehot)
    tweets_onehots_test = []
    for token_list in token_lists_test:
        tweet_onehot = []
        for token in token_list:
            if len(tweet_onehot) < padtosize:
                if token in vocabulary:
                    tweet_onehot.append(vocabulary.index(token))
                else:
                    tweet_onehot.append(0)
        while len(tweet_onehot) < padtosize:
            tweet_onehot.append(0)
        tweets_onehots_test.append(tweet_onehot)
    return (tweets_onehots_train,tweets_onehots_test)

def get_token_onehot_vectors(token_lists_train,token_lists_test,vocabulary,onehotsize=100,tokennumber=10):
    tokensOhs_train = []
    for token_list in token_lists_train:
        tokenOhs = []
        for token in token_list:
            if token in vocabulary and vocabulary.index(token) < onehotsize and len(tokenOhs) < tokennumber:
                token_oh = [0]*onehotsize
                token_oh[vocabulary.index(token)] = 1
                tokenOhs.append(token_oh)
        while len(tokenOhs) < tokennumber:
            tokenOhs.append([0]*onehotsize)
        tokensOhs_train.append(tokenOhs)
    tokensOhs_test = []
    for token_list in token_lists_test:
        tokenOhs = []
        for token in token_list:
            if token in vocabulary and vocabulary.index(token) < onehotsize and len(tokenOhs) < tokennumber:
                token_oh = [0]*onehotsize
                token_oh[vocabulary.index(token)] = 1
                tokenOhs.append(token_oh)
        while len(tokenOhs) < tokennumber:
            tokenOhs.append([0]*onehotsize)
        tokensOhs_test.append(tokenOhs)
    return (tokensOhs_train,tokensOhs_test)

def get_pertarget_token_representations(representations,target):
    pttreps = []
    for rep in representations:
        pttrep = [0]*(5*len(rep))
        for i in range(0,len(rep)):
            pttrep[i+(target*len(rep))]=rep[i]
        pttreps.append(pttrep)
    return pttreps


def load_embeddings(file_name, vocabulary):
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
    print('load embedding weights '+glovepath+" "+savepath)
    embeddings = load_embeddings(glovepath,vocabulary)
    for token in vocabulary:
        if token not in embeddings:
            embeddings[token] = np.random.standard_normal(50)  
    for hashtag in htags_dict:
        for h in htags_dict[hashtag]:
            if h not in embeddings:
                embeddings[h] = np.random.standard_normal(50)            
        summary_embed = np.average([embeddings[h] for h in htags_dict[hashtag]], axis=0)
        embeddings[hashtag] = summary_embed
    return embeddings

def map_tokens_to_embedding_vectors(tokens_lists,embeddings,padtosize = 30):
    mapped = []
    mapped_summarized = []
    for token_list in tokens_lists:
        mapp = []
        for token in token_list:
            if token not in embeddings:
                embeddings[token] = np.random.standard_normal(50)
            if len(mapp) >= padtosize:
                continue
            mapp.append(embeddings[token])
        while len(mapp) < padtosize:
            mapp.append([0.0]*50)
        mapped.append(mapp)
        mapp_sum = np.average([embeddings[token] for token in token_list], axis=0)
        mapped_summarized.append(mapp_sum)
    return (mapped,mapped_summarized)

def set_hashtag_embeddings():
    for htag in htags_dict:
        embed_wiki = []
        embed_twit = []
        for part in htags_dict[htag]:
            embed_wiki.append(embeddings_GLOBAL[part])
            embed_twit.append(embeddings_twitter_GLOBAL[part])
        embeddings_GLOBAL[htag] = np.average(embed_wiki, axis=0)
        embeddings_twitter_GLOBAL[htag] = np.average(embed_twit, axis=0)

#BUILD GLOBAL VOCABULARY
all_tokens = set()
all_tokens.update(words_vocabulary)
all_tokens.update(lemmas_vocabulary)
all_tokens.update(stems_vocabulary)
all_tokens.update(words_vocabulary_test)
all_tokens.update(lemmas_vocabulary_test)
all_tokens.update(stems_vocabulary_test)
all_tokens.update(ners_vocabulary_test)
all_tokens.update(nouns_vocabulary_test)
all_tokens.update(list(hashtag_parts_vocabulary))
all_tokens.update(htags_dict.keys())
global_vocabulary = list(all_tokens)

print('EMBEDDINGS EXTRACTION')
#GLOBAL EMBEDDINGS
embeddings_GLOBAL = load_embedding_weights(global_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_ALL.pkl")
embeddings_twitter_GLOBAL = load_embedding_weights(global_vocabulary,glovepath='embedings\glove.twitter.27B.50d.txt',savepath="embedings\embeddings_ALL_twitter.pkl")

# embedding_matrix_w,embeddings_w = load_embedding_weights(words_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_words.pkl")
# embedding_matrix_w,embeddings_w = load_embedding_weights(words_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_words.pkl")
# embedding_matrix_wns, embeddings_wns = load_embedding_weights(words_nostops_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_words_NS.pkl")
# embedding_matrix_l, embeddings_l = load_embedding_weights(lemmas_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_lemmas.pkl")
# embedding_matrix_lns, embeddings_lns = load_embedding_weights(lemmas_nostops_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_lemmas_NS.pkl")
# embedding_matrix_s, embeddings_s = load_embedding_weights(stems_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_stems.pkl")
# embedding_matrix_sns , embeddings_sns= load_embedding_weights(stems_nostops_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_stems_NS.pkl")
# embedding_matrix_h ,embeddings_h= load_embedding_weights(htags_vocabulary,glovepath='embedings\glove.6B.50d.txt',savepath="embedings\embeddings_htags.pkl")

# embedding_matrix_twitter_w, embeddings_twitter_w = load_embedding_weights(words_vocabulary,glovepath='embedings\glove.twitter.27B.50d.txt',savepath="embedings\embeddings_words_twitter.pkl")
# embedding_matrix_twitter_wns, embeddings_twitter_wns = load_embedding_weights(words_nostops_vocabulary,glovepath='embedings\glove.twitter.27B.50d.txt',savepath="embedings\embeddings_words_NS_twitter.pkl")
# embedding_matrix_twitter_l, embeddings_twitter_l = load_embedding_weights(lemmas_vocabulary,glovepath='embedings\glove.twitter.27B.50d.txt',savepath="embedings\embeddings_lemmas_twitter.pkl")
# embedding_matrix_twitter_lns, embeddings_twitter_lns = load_embedding_weights(lemmas_nostops_vocabulary,glovepath='embedings\glove.twitter.27B.50d.txt',savepath="embedings\embeddings_lemmas_NS_twitter.pkl")
# embedding_matrix_twitter_s, embeddings_twitter_s = load_embedding_weights(stems_vocabulary,glovepath='embedings\glove.twitter.27B.50d.txt',savepath="embedings\embeddings_stems_twitter.pkl")
# embedding_matrix_twitter_sns , embeddings_twitter_sns= load_embedding_weights(stems_nostops_vocabulary,glovepath='embedings\glove.twitter.27B.50d.txt',savepath="embedings\embeddings_stems_NS_twitter.pkl")
# embedding_matrix_twitter_h ,embeddings_twitter_h= load_embedding_weights(htags_vocabulary,glovepath='embedings\glove.twitter.27B.50d.txt',savepath="embedings\embeddings_htags_twitter.pkl")

#REPRESENT EACH TOKEN AS GLOVE WIKIPEDIA EMBEDDING VECTOR
words_embeddings_train, words_embeddings_summary_train= map_tokens_to_embedding_vectors(words_train,embeddings_GLOBAL,padtosize=avg_w)
words_embeddings_ns_train,words_embeddings_ns_summary_train = map_tokens_to_embedding_vectors(words_nostops_train,embeddings_GLOBAL,padtosize=avg_wns)
words_embeddings_test,words_embeddings_summary_test = map_tokens_to_embedding_vectors(words_test,embeddings_GLOBAL,padtosize=avg_w)
words_embeddings_ns_test,words_embeddings_ns_summary_test = map_tokens_to_embedding_vectors(words_nostops_test,embeddings_GLOBAL,padtosize=avg_wns)

lemmas_embeddings_train,lemmas_embeddings_summary_train = map_tokens_to_embedding_vectors(lemmas_train,embeddings_GLOBAL,padtosize=avg_l)
lemmas_embeddings_ns_train,lemmas_embeddings_ns_summary_train = map_tokens_to_embedding_vectors(lemmas_nostops_train,embeddings_GLOBAL,padtosize=avg_lns)
lemmas_embeddings_test,lemmas_embeddings_summary_test = map_tokens_to_embedding_vectors(lemmas_test,embeddings_GLOBAL,padtosize=avg_l)
lemmas_embeddings_ns_test,lemmas_embeddings_ns_summary_test = map_tokens_to_embedding_vectors(lemmas_nostops_test,embeddings_GLOBAL,padtosize=avg_lns)

stems_embeddings_train,stems_embeddings_summary_train = map_tokens_to_embedding_vectors(stems_train,embeddings_GLOBAL,padtosize=avg_s)
stems_embeddings_ns_train,stems_embeddings_ns_summary_train = map_tokens_to_embedding_vectors(stems_nostops_train,embeddings_GLOBAL,padtosize=avg_sns)
stems_embeddings_test,stems_embeddings_summary_test = map_tokens_to_embedding_vectors(stems_test,embeddings_GLOBAL,padtosize=avg_s)
stems_embeddings_ns_test,stems_embeddings_ns_summary_test = map_tokens_to_embedding_vectors(stems_nostops_test,embeddings_GLOBAL,padtosize=avg_sns)

hashtags_embeddings_train,hashtags_embeddings_summary_train = map_tokens_to_embedding_vectors(htags_train,embeddings_GLOBAL,padtosize=avg_h)
hashtags_embeddings_test,hashtags_embeddings_summary_test = map_tokens_to_embedding_vectors(htags_test,embeddings_GLOBAL,padtosize=avg_h)

#REPRESENT EACH TOKEN AS GLOVE TWITTER EMBEDDING VECTOR
words_embeddings_TWT_train,words_embeddings_TWT_summary_train = map_tokens_to_embedding_vectors(words_train,embeddings_twitter_GLOBAL,padtosize=avg_w)
words_embeddings_ns_TWT_train,words_embeddings_ns_TWT_summary_train = map_tokens_to_embedding_vectors(words_nostops_train,embeddings_twitter_GLOBAL,padtosize=avg_wns)
words_embeddings_TWT_test,words_embeddings_TWT_summary_test = map_tokens_to_embedding_vectors(words_test,embeddings_twitter_GLOBAL,padtosize=avg_w)
words_embeddings_ns_TWT_test,words_embeddings_ns_TWT_summary_test = map_tokens_to_embedding_vectors(words_nostops_test,embeddings_twitter_GLOBAL,padtosize=avg_wns)

lemmas_embeddings_TWT_train,lemmas_embeddings_TWT_summary_train = map_tokens_to_embedding_vectors(lemmas_train,embeddings_twitter_GLOBAL,padtosize=avg_l)
lemmas_embeddings_ns_TWT_train,lemmas_embeddings_ns_TWT_summary_train = map_tokens_to_embedding_vectors(lemmas_nostops_train,embeddings_twitter_GLOBAL,padtosize=avg_lns)
lemmas_embeddings_TWT_test,lemmas_embeddings_TWT_summary_test = map_tokens_to_embedding_vectors(lemmas_test,embeddings_twitter_GLOBAL,padtosize=avg_l)
lemmas_embeddings_ns_TWT_test,lemmas_embeddings_ns_TWT_summary_test = map_tokens_to_embedding_vectors(lemmas_nostops_test,embeddings_twitter_GLOBAL,padtosize=avg_lns)

stems_embeddings_TWT_train,stems_embeddings_TWT_summary_train = map_tokens_to_embedding_vectors(stems_train,embeddings_twitter_GLOBAL,padtosize=avg_s)
stems_embeddings_ns_TWT_train,stems_embeddings_ns_TWT_summary_train = map_tokens_to_embedding_vectors(stems_nostops_train,embeddings_twitter_GLOBAL,padtosize=avg_sns)
stems_embeddings_TWT_test,stems_embeddings_TWT_summary_test = map_tokens_to_embedding_vectors(stems_test,embeddings_twitter_GLOBAL,padtosize=avg_s)
stems_embeddings_ns_TWT_test,stems_embeddings_ns_TWT_summary_test = map_tokens_to_embedding_vectors(stems_nostops_test,embeddings_twitter_GLOBAL,padtosize=avg_sns)

hashtags_embeddings_TWT_train,hashtags_embeddings_TWT_summary_train = map_tokens_to_embedding_vectors(htags_train,embeddings_twitter_GLOBAL,padtosize=avg_h)
hashtags_embeddings_TWT_test,hashtags_embeddings_TWT_summary_test = map_tokens_to_embedding_vectors(htags_test,embeddings_twitter_GLOBAL,padtosize=avg_h)

ners_embeddings_TWT_train,ners_embeddings_TWT_summary_train = map_tokens_to_embedding_vectors(ners_train,embeddings_twitter_GLOBAL,padtosize=avg_ner)
ners_embeddings_TWT_test,ners_embeddings_TWT_summary_test = map_tokens_to_embedding_vectors(ners_train,embeddings_twitter_GLOBAL,padtosize=avg_ner)

nouns_embeddings_TWT_train,nouns_embeddings_TWT_summary_train = map_tokens_to_embedding_vectors(nouns_train,embeddings_twitter_GLOBAL,padtosize=avg_noun)
nouns_embeddings_TWT_test,nouns_embeddings_TWT_summary_test = map_tokens_to_embedding_vectors(nouns_test,embeddings_twitter_GLOBAL,padtosize=avg_noun)




####### ENCODE EACH WORD WITH THE INTEGER IN THE FREQUENCY VECTOR
words_freq_vec_train,words_freq_vec_test = get_frequency_vectors(words_train,words_test, words_vocabulary,avg_w)
words_ns_freq_vec_train,words_ns_freq_vec_test = get_frequency_vectors(words_nostops_train,words_nostops_test, words_vocabulary,avg_wns)

lemmas_freq_vec_train,lemmas_freq_vec_test = get_frequency_vectors(lemmas_train,lemmas_test,lemmas_vocabulary,avg_l)
lemmas_ns_freq_vec_train,lemmas_ns_freq_vec_test = get_frequency_vectors(lemmas_nostops_train,lemmas_nostops_test,lemmas_nostops_vocabulary,avg_lns)

stems_freq_vec_train,stems_freq_vec_test = get_frequency_vectors(stems_train,stems_test,stems_vocabulary,avg_s)
stems_ns_freq_vec_train,stems_ns_freq_vec_test = get_frequency_vectors(stems_nostops_train,stems_nostops_test,stems_nostops_vocabulary,avg_sns)

hashtags_freq_vec_train, hashtags_freq_vec_test = get_frequency_vectors(htags_train,htags_test,htags_vocabulary,avg_h)
ners_freq_vec_train,ners_freq_vec_test = get_frequency_vectors(ners_train,ners_test,ners_vocabulary,avg_ner)
nouns_freq_vec_train,nouns_freq_vec_test = get_frequency_vectors(nouns_train,nouns_test,nouns_vocabulary,avg_noun)

stems_freq_vec_train,stems_freq_vec_test = get_frequency_vectors(stems_train,stems_test,stems_vocabulary,avg_s)
stems_ns_freq_vec_train,stems_ns_freq_vec_test = get_frequency_vectors(stems_nostops_train,stems_nostops_test,stems_nostops_vocabulary,avg_sns)

####### ENCODE EACH TWEET WITH ONE-HOT VOCABULARY ARRAY
words_vocab_vec_train,words_vocab_vec_test = get_vocabulary_vectors(words_train,words_test, words_vocabulary)
words_ns_vocab_vec_train,words_ns_vocab_vec_test = get_vocabulary_vectors(words_nostops_train,words_nostops_test, words_vocabulary)

lemmas_vocab_vec_train,lemmas_vocab_vec_test = get_vocabulary_vectors(lemmas_train,lemmas_test,lemmas_vocabulary)
lemmas_ns_vocab_vec_train,lemmas_ns_vocab_vec_test = get_vocabulary_vectors(lemmas_nostops_train,lemmas_nostops_test,lemmas_nostops_vocabulary)

stems_vocab_vec_train,stems_vocab_vec_test = get_vocabulary_vectors(stems_train,stems_test,stems_vocabulary)
stems_ns_vocab_vec_train,stems_ns_vocab_vec_test = get_vocabulary_vectors(stems_nostops_train,stems_nostops_test,stems_nostops_vocabulary)

hashtags_vocab_vec_train, hashtags_vocab_vec_test = get_vocabulary_vectors(htags_train,htags_test,htags_vocabulary)
ners_vocab_vec_train,ners_vocab_vec_test = get_vocabulary_vectors(ners_train,ners_test,ners_vocabulary)
nouns_vocab_vec_train,nouns_vocab_vec_test = get_vocabulary_vectors(nouns_train,nouns_test,nouns_vocabulary)

stems_vocab_vec_train,stems_vocab_vec_test = get_vocabulary_vectors(stems_train,stems_test,stems_vocabulary)
stems_ns_vocab_vec_train,stems_ns_vocab_vec_test = get_vocabulary_vectors(stems_nostops_train,stems_nostops_test,stems_nostops_vocabulary)

####### ENCODE EACH TOKEN AS 
words_tokenohs_train,words_tokenohs_test = get_token_onehot_vectors(words_train,words_test, words_vocabulary)
words_ns_tokenohs_train,words_ns_tokenohs_test = get_token_onehot_vectors(words_nostops_train,words_nostops_test, words_vocabulary)

lemmas_tokenohs_train,lemmas_tokenohs_test = get_token_onehot_vectors(lemmas_train,lemmas_test,lemmas_vocabulary)
lemmas_ns_tokenohs_train,lemmas_ns_tokenohs_test = get_token_onehot_vectors(lemmas_nostops_train,lemmas_nostops_test,lemmas_nostops_vocabulary)

stems_tokenohs_train,stems_tokenohs_test = get_token_onehot_vectors(stems_train,stems_test,stems_vocabulary)
stems_ns_tokenohs_train,stems_ns_tokenohs_test = get_token_onehot_vectors(stems_nostops_train,stems_nostops_test,stems_nostops_vocabulary)

hashtags_tokenohs_train, hashtags_tokenohs_test = get_token_onehot_vectors(htags_train,htags_test,htags_vocabulary)
ners_tokenohs_train,ners_tokenohs_test = get_token_onehot_vectors(ners_train,ners_test,ners_vocabulary)
nouns_tokenohs_train,nouns_tokenohs_test = get_token_onehot_vectors(nouns_train,nouns_test,nouns_vocabulary)

stems_tokenohs_train,stems_tokenohs_test = get_token_onehot_vectors(stems_train,stems_test,stems_vocabulary)
stems_ns_tokenohs_train,stems_ns_tokenohs_test = get_token_onehot_vectors(stems_nostops_train,stems_nostops_test,stems_nostops_vocabulary)
