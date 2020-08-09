from readDataFromFile import *
from utils import pad_element_list
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize
import operator
import numpy as np
from nltk.corpus import stopwords


def get_concatenated_phrases(phrases):
    concat = []
    for p in phrases:
        concat.append(p.lower().replace(" ",""))
    return concat

def get_counted_dictionary_from_list(everyword):
    dictionary = {}
    for word in everyword:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1],reverse = True)}

def get_disjunct_vocabularies(vocab_favor,vocab_against,vocab_none):
    disj_favor = []
    disj_against = []
    disj_none = []
    for word in vocab_favor:
        if word not in vocab_against and word not in vocab_none:
            disj_favor.append(word)
    for word in vocab_against:
        if word not in vocab_favor and word not in vocab_none:
            disj_against.append(word)
    for word in vocab_none:
        if word not in vocab_against and word not in vocab_favor:
            disj_none.append(word)
    return (disj_favor,disj_against,disj_none)

def get_intersection_vocabulary(total_vocab,vocab_favor,vocab_against,vocab_none):
    intersect = []
    for word in total_vocab:
        if word in vocab_favor and word in vocab_against and word in vocab_none:
            intersect.append(word)
    return intersect


def get_words_for_each_stance(tweets,stances):
    everyword_favor = []
    everyword_against = []
    everyword_none = []
    for (tweet,stance) in zip(tweets,stances):
        if(stance == "FAVOR"):
            everyword_favor.extend(tokenize(tweet.lower()))
        if(stance == "AGAINST"):
            everyword_against.extend(tokenize(tweet.lower()))
        if(stance == "NONE"):
            everyword_none.extend(tokenize(tweet.lower()))
    sorted_favor = get_counted_dictionary_from_list(everyword_favor)
    sorted_against = get_counted_dictionary_from_list(everyword_against)
    sorted_none = get_counted_dictionary_from_list(everyword_none)

    return (sorted_favor, sorted_against, sorted_none, list(sorted_favor.keys()),list(sorted_against.keys()),list(sorted_none.keys()))

def get_counted_vocabulary_from_sentence_list(sentence_list):
    everyword = []
    for sentence in sentence_list:
        everyword.extend(tokenize(sentence.replace("#SemST","").lower()))

    sorted_dict = get_counted_dictionary_from_list(everyword)
    return (sorted_dict,list(sorted_dict.keys()))

def filterout_meaningless_words_from_tweets(tweets,meaningless):
    filtered = []
    for tweet in tweets:
        wrds = tokenize(tweet.replace("#SemST","").lower())
        filtered_wrds = []
        for w in wrds:
            if w not in meaningless:
                filtered_wrds.append(w)
        filtered.append(filtered_wrds)
    return filtered

def get_tweets_as_word_lists(tweets,targets,includeTargets = False):
    tweetlist = []
    for (tweet,target) in zip(tweets,targets):
        if includeTargets:
            tweetlist.append(tokenize(((target.replace(" ","") + " "+tweet.replace("#SemST","").lower()))))
        else:
            tweetlist.append(tokenize(((tweet.replace("#SemST","").lower()))))
    return tweetlist

def get_tweet_length_stats(tweets):
    wordcount = 0
    max = 0
    min = 100
    for tweet in tweets:
        wordcount += len(tweet)
        if(len(tweet) > max):
            max = len(tweet)
        if(len(tweet) < min):
            min = len(tweet)
    avg = 1.0*wordcount/len(tweets)
    #print("avg: "+str(avg)+", min: "+str(min)+", max: "+str(max))
    return (avg, min, max)

def get_stances_as_integers(stances):
    intarr = []
    for stance in stances:
        if stance == 'NONE':
            intarr.append(0)
        elif stance == 'FAVOR':
            intarr.append(1)
        elif stance == 'AGAINST':
            intarr.append(2)
        else:
            intarr.append(-1)
    return intarr

def tweets_as_intarrays_given_vocab(tweets,vocabulary):
    arrints = []
    for tweet in tweets:
        arrint = []
        for word in tweet:
            if word in vocabulary:
                arrint.append(vocabulary.index(word))
        arrints.append(arrint)
    return arrints

def tweets_as_intarray_vocabVector(tweets,vocabulary):
    arrints = []
    for tweet in tweets:
        arrint = []
        for word in vocabulary:
            if word in tweet:
                arrint.append(1)
            else:
                arrint.append(0)
        arrints.append(arrint)
    return arrints

tweets_words, tweets_vocabulary = get_counted_vocabulary_from_sentence_list(tweets_train)
# print("tweets_vocabulary size:"+str(len(tweets_vocabulary)))
tweets_vocabulary.extend(get_concatenated_phrases(target_vocabulary))
#print("len full vocab: "+str(len(tweets_vocabulary)))
dict_favor, dict_against, dict_none, vocab_favor, vocab_against, vocab_none = get_words_for_each_stance(tweets_train,stances_train)
#print("len vocab favor: "+str(len(vocab_favor))+"   len vocab against: "+str(len(vocab_against))+"    len vocab none: "+str(len(vocab_none)))

disj_favor,disj_against,disj_none = get_disjunct_vocabularies(vocab_favor,vocab_against,vocab_none)
#print("len disj vocab favor: "+str(len(disj_favor))+"   len disj vocab against: "+str(len(disj_against))+"    len disj vocab none: "+str(len(disj_none)))

intersection_vocab = get_intersection_vocabulary(tweets_vocabulary,vocab_favor,vocab_against,vocab_none)
#print("len intersection vocab: "+str(len(intersection_vocab)))

filtered_tweets = filterout_meaningless_words_from_tweets(tweets_train,intersection_vocab)
tweets_as_lists = get_tweets_as_word_lists(tweets_train,targets_train)
#print("tweets_as_lists train len: "+str(len(tweets_as_lists)))

filtered_stopwords_tweets = filterout_meaningless_words_from_tweets(tweets_train,stopwords.words('english'))

# avgfiltered,minfiltered,maxfiltered = get_tweet_length_stats(filtered_tweets)
# avgraw, minraw,maxraw = get_tweet_length_stats(tweets_as_lists)
# avgstop, minstop,maxstop = get_tweet_length_stats(filtered_stopwords_tweets)

filtered_tweets_test = filterout_meaningless_words_from_tweets(tweets_test,intersection_vocab)
tweets_as_lists_test = get_tweets_as_word_lists(tweets_test,targets_test)
#print("tweets_as_lists_test test len: "+str(len(tweets_as_lists)))
filtered_stopwords_tweets_test = filterout_meaningless_words_from_tweets(tweets_test,stopwords.words('english'))


vocab_of_disjunct = []
vocab_of_disjunct.extend(disj_favor)
vocab_of_disjunct.extend(disj_against)
vocab_of_disjunct.extend(disj_none)
vocab_of_disjunct.extend(get_concatenated_phrases(target_vocabulary))

TW_WORD_LENGTH = 20         #avg: 17.151030561478322, min: 3, max: 32
TW_FILTERED_WORD_LENGTH = 7 #avg: 4.779673063255153, min: 0, max: 13
TW_STOP_WORD_LENGTH = 10    #avg: 9.58955223880597, min: 2, max: 18

stances_asints_train = get_stances_as_integers(stances_train)
stances_asints_test= get_stances_as_integers(stances_test)

tweets_intarray_train = tweets_as_intarrays_given_vocab(tweets_as_lists,tweets_vocabulary)
tweets_intarray_test = tweets_as_intarrays_given_vocab(tweets_as_lists_test,tweets_vocabulary)

tweets_intarray_padded_train = pad_element_list(tweets_intarray_train,padtosize=30,padwith=0)
tweets_intarray_padded_test = pad_element_list(tweets_intarray_test,padtosize=30,padwith=0)

tweets_vocabVector_train = tweets_as_intarray_vocabVector(tweets_as_lists,tweets_vocabulary[0:500])
tweets_vocabVector_test = tweets_as_intarray_vocabVector(tweets_as_lists_test,tweets_vocabulary[0:500])

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

tweets_wordOhs_train = get_tweets_as_wordOhs(tweets_as_lists,tweets_vocabulary,size=500,padto=TW_WORD_LENGTH)
tweets_wordOhs_test = get_tweets_as_wordOhs(tweets_as_lists_test,tweets_vocabulary,size=500,padto=TW_WORD_LENGTH)