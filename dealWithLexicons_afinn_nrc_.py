from readDataFromFile import tweets_test,tweets_train,tweets_wordlists_test,tweets_wordlists_train
import numpy as np
import re
import json
import pandas as pd
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
import more_itertools as mit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dealWith_SentiWordNet import *
from empath import Empath
empath_lexicon = Empath()

empath_lexicon.create_category("hillary_clinton",["twitter","hillary_clinton","woman_president","hillary","politics",'clinton','woman','sexism'])
empath_lexicon.create_category("atheism",["twitter","god","science","faith","believe",'bible','religion','jesus','sin'])
empath_lexicon.create_category("abortion",["twitter","pregnancy","women","choice","baby",'poor','rape',"planned_parenhood"])
empath_lexicon.create_category("feminism",["twitter","womens_rights","woman","equality","pay_gap",'sexism','abuse','metoo'])
empath_lexicon.create_category("climate_change",["twitter","climate_change","greenhouse_effect","pollution","heat_wave","waste","recycle"])
empath_categories = ['speaking', 'dominant_personality', 'anger', 'leader', 'worship', 'vacation', 'driving', 'phone', 'irritability', 'home', 'party', 'strength', 'heroic', 'swimming', 'wedding', 'sexual', 'appearance', 'deception', 'exercise', 'masculine', 'terrorism', 'blue_collar_job', 'medieval', 'neglect', 'dispute', 'cheerfulness', 'weather', 'programming', 'injury', 'plant', 'writing', 'reading', 'warmth', 'medical_emergency', 'weapon', 'exasperation', 'water', 'play', 'breaking', 'fabric', 'pride', 'business', 'royalty', 'zest', 'optimism', 'childish', 'negotiate', 'white_collar_job', 'fun', 'disgust', 'urban', 'sadness', 'achievement', 'hiking', 'traveling', 'morning', 'internet', 'furniture', 'healing', 'confusion', 'gain', 'ugliness', 'beauty', 'smell', 'abortion', 'aggression', 'sleep', 'politics', 'violence', 'noise', 'tool', 'tourism', 'communication', 'cold', 'surprise', 'exotic', 'meeting', 'pain', 'hate', 'stealing', 'fight', 'fashion', 'dance', 'crime', 'liquid', 'economics', 'fire', 'farming', 'domestic_work', 'leisure', 'war', 'envy', 'fear', 'listen', 'anonymity', 'car', 'positive_emotion', 'health', 'family', 'office', 'emotional', 'journalism', 'cleaning', 'suffering', 'rage', 'science', 'wealthy', 'youth', 'legend', 'attractive', 'superhero', 'shame', 'dominant_heirarchical', 'power', 'shape_and_size', 'school', 'ship', 'children', 'messaging', 'affection', 'animal', 'horror', 'government', 'ocean', 'eating', 'money', 'art', 'rural', 'ancient', 'law', 'occupation', 'feminism', 'lust', 'sports', 'celebration', 'military', 'sailing', 'death', 'shopping', 'torment', 'kill', 'joy', 'independence', 'air_travel', 'hearing', 'poor', 'cooking', 'pet', 'magic', 'trust', 'disappointment', 'monster', 'valuable', 'alcohol', 'hillary_clinton', 'restaurant', 'nervousness', 'sympathy', 'atheism', 'love', 'anticipation', 'body', 'work', 'help', 'hipster', 'night', 'computer', 'prison', 'timidity', 'real_estate', 'friends', 'weakness', 'technology', 'payment', 'competing', 'hygiene', 'clothing', 'politeness', 'swearing_terms', 'negative_emotion', 'feminine', 'toy', 'religion', 'banking', 'college', 'ridicule', 'giving', 'musical', 'sound', 'divine', 'movement', 'beach', 'vehicle', 'philosophy', 'climate_change', 'order', 'music', 'contentment', 'social_media']
from utils import *

def get_AFINN_lexicon():
    #word integer pairs
    #   20009 lines
    lexicon = dict()
    with open(r'lexicons\AFINN-111.txt') as f1:
        for line in f1:
            word, score = line.split('\t')
            lexicon[word] = float(score)
    return lexicon

def get_NRCVAD_lexicon():
    #Word	Valence	Arousal	Dominance
    #   2477 lines
    lexicon = dict()
    with open(r'lexicons\NRC-VAD.txt') as f1:
        next(f1)
        for line in f1:
            word, valence, arousal, dominance = line.split('\t')
            lexicon[word] = [float(valence), float(arousal), float(dominance)]
    return lexicon

def get_NRCAffectIntensity_lexicon():
    #word integer emotion  --> ['joy', 'fear', 'anger', 'sadness']
    #   5815 lines
    lexicon = dict()
    with open(r'lexicons\NRC-AffectIntensity.txt') as f1:
        for line in f1:
            word, score, emotion = line.split(' ')
            lexicon[word] = [float(score),emotion.strip()]
    return lexicon

afinn = get_AFINN_lexicon()
nrc_vad = get_NRCVAD_lexicon()
nrc_affect_intensity = get_NRCAffectIntensity_lexicon()

def wordslists_to_afinn_score_lists(wordslists): #integer list of affinscores for words for each wordlist
    afinnscoreslist = []
    for wordlist in wordslists:
        afinnscr = []
        for word in wordlist:
            if word in afinn:
                afinnscr.append(afinn[word])
            else:
                afinnscr.append(0.0) #default 0
        afinnscoreslist.append(afinnscr)
    return afinnscoreslist

def wordslists_to_summary_afinn_score(wordslists): #integer for summary affinscore for each wordlist
    afinnscoreslist = []
    for wordlist in wordslists:
        afinnscr = 0.0
        for word in wordlist:
            if word in afinn:
                afinnscr = afinnscr + afinn[word]
        afinnscoreslist.append(afinnscr)
    return afinnscoreslist

def wordslists_to_nrcvad_score_lists(wordslists): #[valence,arousal,dominance] vector for each word in each wordlist
    nrcvadscoreslist = []
    for wordlist in wordslists:
        nrcvadscr = []
        for word in wordlist:
            if word in nrc_vad:
                nrcvadscr.append(nrc_vad[word])
            else:
                nrcvadscr.append([0.0,0.0,0.0]) #default vector
        nrcvadscoreslist.append(nrcvadscr)
    return nrcvadscoreslist

def wordslists_to_nrcvad_summaryvectors_lists(wordslists): #[valence,arousal,dominance] vector for each wordlist
    nrcvadscoreslist = []
    for wordlist in wordslists:
        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        for word in wordlist:
            if word in nrc_vad:
                valence=valence+nrc_vad[word][0]
                arousal=arousal+nrc_vad[word][1]
                dominance=dominance+nrc_vad[word][2]
        cnt = len(wordlist)
        nrcvadscoreslist.append([valence/cnt,arousal/cnt,dominance/cnt])
    return nrcvadscoreslist

def wordslists_to_nrc_affin_score_lists(wordslists,intencodeemotion = False,onehotencodeemotion=False,flatten = True): #[score,emotion] vector for each word in each wordlist, emotion can be string(FF),int(TF),onehot(FT)
    nrcafinnscoreslist = []
    emotion_integers = {'joy':1,'fear':2,'anger':3,'sadness':4}
    emotion_onehots = {'joy':[1,0,0,0],'fear':[0,1,0,0],'anger':[0,0,1,0],'sadness':[0,0,0,1]}

    for wordlist in wordslists:
        nrcafinnscr = []
        for word in wordlist:
            if word in nrc_affect_intensity:
                if intencodeemotion == False and onehotencodeemotion == False:
                    nrcafinnscr.append(nrc_affect_intensity[word]) # (score,'string emotion')
                elif intencodeemotion == True:
                    nrcafinnscr.append([nrc_affect_intensity[word][0],emotion_integers[nrc_affect_intensity[word][1]]]) # (score,'int map for emotion')
                elif onehotencodeemotion == True and flatten == False:
                    nrcafinnscr.append([nrc_affect_intensity[word][0],emotion_onehots[nrc_affect_intensity[word][1]]]) # (score,'onehot map for emotion')
                elif onehotencodeemotion == True and flatten == True:
                    res = [nrc_affect_intensity[word][0]]
                    res.extend(emotion_onehots[nrc_affect_intensity[word][1]])
                    nrcafinnscr.append(res) # (score,'onehot map for emotion')
            else:
                if intencodeemotion == False and onehotencodeemotion == False:
                    nrcafinnscr.append([0.0,'none']) # (score,'string emotion')
                elif intencodeemotion == True:
                    nrcafinnscr.append([0.0,0]) # (score,'int map for emotion')
                elif onehotencodeemotion == True and flatten == False:
                    nrcafinnscr.append([0.0,[0,0,0,0]]) # (score,'onehot map for emotion')
                elif onehotencodeemotion == True and flatten == True:
                    nrcafinnscr.append([0.0,0,0,0,0]) # (score,'onehot map for emotion')
        nrcafinnscoreslist.append(nrcafinnscr)
    return nrcafinnscoreslist

def wordslists_to_nrc_affin_score_vectors(wordslists): #[joy fear anger sadness] (onehot) * score
    nrcafinnscoreslist = []
    emotion_onehots = {'joy':[1,0,0,0],'fear':[0,1,0,0],'anger':[0,0,1,0],'sadness':[0,0,0,1]}

    for wordlist in wordslists:
        nrcafinnscr = []
        for word in wordlist:
            if word in nrc_affect_intensity:
                value = nrc_affect_intensity[word][0]
                oh = emotion_onehots[nrc_affect_intensity[word][1]]
                res = [element * value for element in oh]
                nrcafinnscr.append(res)
            else:
                nrcafinnscr.append([0.0,0.0,0.0,0.0])            
        nrcafinnscoreslist.append(nrcafinnscr)
    return nrcafinnscoreslist

def wordslists_to_summary_nrc_affin_score(wordslists): #[joy,fear,anger,sadness] averages vector per wordlist
    nrcafinnscoreslist = []
    for wordlist in wordslists:
        emotscores = {'joy':0.0,'fear':0.0,'anger':0.0,'sadness':0.0}
        emotcnts = {'joy':0,'fear':0,'anger':0,'sadness':0}
        for word in wordlist:
            if word in nrc_affect_intensity:
                emotscores[nrc_affect_intensity[word][1]] = emotscores[nrc_affect_intensity[word][1]] + nrc_affect_intensity[word][0]
                emotcnts[nrc_affect_intensity[word][1]] = emotcnts[nrc_affect_intensity[word][1]] + 1
        result = []
        for key in ['joy', 'fear', 'anger', 'sadness']:
            if emotcnts[key] == 0:
                result.append(0)
            else:
                result.append(emotscores[key]/emotcnts[key])
        nrcafinnscoreslist.append(result)
    return nrcafinnscoreslist

def tweets_to_sentimentIntensityAnalysis_vectors(tweets): #SentimentIntensityAnalyzer [neg,neu,pos,compound] vector per tweet
    sentimentscores = []
    analyser = SentimentIntensityAnalyzer()
    for tweet in tweets:
        score = analyser.polarity_scores(tweet)
        sentimentscores.append([score['neg'],score['neu'],score['pos'],score['compound']])
    return sentimentscores

def tweets_to_empath_scores(tweets): #SentimentIntensityAnalyzer [neg,neu,pos,compound] vector per tweet
    empathscores = []
    for tweet in tweets:
        score = empath_lexicon.analyze(tweet, normalize=True)
        vector = []
        for key in empath_categories:
            vector.append(score[key])
        empathscores.append(vector)
    return empathscores

def tweets_to_liu_score(tweets,padtosize=20):
    positive_words = []
    negative_words = []
    with open('lexicons\liu_positive-words.txt') as file:
        line = file.readline() # to pass through the first line that has column headers
        while line:
            positive_words.append(line)
            line = file.readline()
    with open('lexicons\liu_negative-words.txt') as file:
        line = file.readline() # to pass through the first line that has column headers
        while line:
            negative_words.append(line)
            line = file.readline()

    tweets_scores = []
    tweets_scores_vectors = []

    for tweet in tweets:
        tweet_score = 0.0
        tweet_score_vector = []
        for word in tweet:
            if word in positive_words:
                tweet_score += 1
                tweet_score_vector.append(1)
            if word in negative_words:
                tweet_score -= 1
                tweet_score_vector.append(-1)
        tweets_scores.append(tweet_score)
        while len(tweet_score_vector) < padtosize:
            tweet_score_vector.append(0)
        tweets_scores_vectors.append(tweet_score_vector[0:padtosize])
    return (tweets_scores,tweets_scores_vectors)



# sentences = ['shit','It is very cloudy',"It isn’t sunny.",'I have sold the last newspaper. ','Someone has eaten all the cookies.','There are none in the bag.','Bad']
# print(tweets_to_sentimentIntensityAnalysis(sentences))

#---AFINN--#
#each tweet wordlist encoded as list of affin scores for its words
tweets_afinn_lists_train = wordslists_to_afinn_score_lists(tweets_wordlists_train)
tweets_afinn_lists_train = pad_element_list(tweets_afinn_lists_train,padtosize=20,padwith=0.0)
tweets_afinn_lists_test= wordslists_to_afinn_score_lists(tweets_wordlists_test)
tweets_afinn_lists_test = pad_element_list(tweets_afinn_lists_test,padtosize=20,padwith=0.0)

#each tweet wordlist encoded a single affin summary score (derived from its words affin scores)
tweets_afinn_summary_train = wordslists_to_summary_afinn_score(tweets_wordlists_train)
tweets_afinn_summary_test= wordslists_to_summary_afinn_score(tweets_wordlists_test)

#---NRC - VAD--#
#each tweet wordlist encoded as nrcvad list of vectors [valence,arousal,dominance] (a vector for each word)
tweets_nrcvad_vectorlist_train = pad_element_list(wordslists_to_nrcvad_score_lists(tweets_wordlists_train),20,[0.0,0.0,0.0])
tweets_nrcvad_vectorlist_test= pad_element_list(wordslists_to_nrcvad_score_lists(tweets_wordlists_test),20,[0.0,0.0,0.0])

#each tweet wordlist encoded as nrcvad vector [valence,arousal,dominance] (1 vector for 1 wordlist)
tweets_nrcvad_vectorsummary_train = wordslists_to_nrcvad_summaryvectors_lists(tweets_wordlists_train)
tweets_nrcvad_vectorsummary_test= wordslists_to_nrcvad_summaryvectors_lists(tweets_wordlists_test)

#---NRC Affect Intensity--#
#each tweet wordlist encoded as a list of (score,emotion as string) for each word in that wordlist
# tweets_nrcafinn_score_emotpairs_train = wordslists_to_nrc_affin_score_lists(tweets_wordlists_train)
# tweets_nrcafinn_score_emotpairs_test= wordslists_to_nrc_affin_score_lists(tweets_wordlists_test)

# #each tweet wordlist encoded as a list of (score,emotion as int) for each word in that wordlist
# tweets_nrcafinn_score_intemotpairs_train = wordslists_to_nrc_affin_score_lists(tweets_wordlists_train,intencodeemotion=True)
# tweets_nrcafinn_score_intemotpairs_test= wordslists_to_nrc_affin_score_lists(tweets_wordlists_test,intencodeemotion=True)

# #each tweet wordlist encoded as a list of (score,emotion as onehot) for each word in that wordlist
# tweets_nrcafinn_score_onehotemotpairs_train = wordslists_to_nrc_affin_score_lists(tweets_wordlists_train,onehotencodeemotion=True)
# tweets_nrcafinn_score_onehotemotpairs_test= wordslists_to_nrc_affin_score_lists(tweets_wordlists_test,onehotencodeemotion=True)

# #each tweet wordlist encoded as a list of (score,emotion as onehot flattened) for each word in that wordlist
# tweets_nrcafinn_score_onehotemotpairs_flat_train = wordslists_to_nrc_affin_score_lists(tweets_wordlists_train,onehotencodeemotion=True,flatten=True)
# tweets_nrcafinn_score_onehotemotpairs_flat_test= wordslists_to_nrc_affin_score_lists(tweets_wordlists_test,onehotencodeemotion=True,flatten=True)

#each tweet wordlist encoded as a list of [em1,em2,em3,em4]*score vectors
tweets_nrcafinn_score_ohvectors_train = pad_element_list(wordslists_to_nrc_affin_score_vectors(tweets_wordlists_train),20,[0.0,0.0,0.0,0.0])
tweets_nrcafinn_score_ohvectors_test= pad_element_list(wordslists_to_nrc_affin_score_vectors(tweets_wordlists_test),20,[0.0,0.0,0.0,0.0])

#each tweet wordlist encoded as vector [joy, fear, anger, sadness] (1 vector for 1 wordlist)
tweets_nrcafinn_summary_train = wordslists_to_summary_nrc_affin_score(tweets_wordlists_train)
tweets_nrcafinn_summary_test= wordslists_to_summary_nrc_affin_score(tweets_wordlists_test)


#---SentimentIntensityAnalyzer--#
#each tweet as a vector [neg,neu,pos,compund]
tweets_sia_vector_train = tweets_to_sentimentIntensityAnalysis_vectors(tweets_train)
tweets_sia_vector_test= tweets_to_sentimentIntensityAnalysis_vectors(tweets_test)


#---------Empath Scores---------#
tweets_empath_vector_train = tweets_to_empath_scores(tweets_train)
tweets_empath_vector_test = tweets_to_empath_scores(tweets_test)

#---------LIU Lexicon----------#
tweets_liu_score_train,tweets_liu_scores_vectors_train = tweets_to_liu_score(tweets_wordlists_train)
tweets_liu_score_test,tweets_liu_scores_vectors_test = tweets_to_liu_score(tweets_wordlists_test)

#(afinn for each word) + (nrc_vad for each word) + (nrc_afinn for each word) + (sia) + (empath) + (liu for each word) + (posnegobj for each word)
def get_tweets_as_combined_lexical_representation():
    combined_train = []
    combined_test = []
    for (afinn,nrc_vad,nrc_afinn,sia,empath,liu,posnegobj) in zip(tweets_afinn_lists_train,tweets_nrcvad_vectorlist_train,tweets_nrcafinn_score_ohvectors_train,tweets_sia_vector_train,tweets_empath_vector_train,tweets_liu_scores_vectors_train,tweets_posnegobj_train):
        tweet_combined = []
        tweet_combined.extend(afinn)
        for item in nrc_vad:
            tweet_combined.extend(item)
        for item in nrc_afinn:
            tweet_combined.extend(item)
        # tweet_combined.extend(empath)
        tweet_combined.extend(sia)
        tweet_combined.extend(liu)
        for item in posnegobj:
            tweet_combined.extend(item)
        combined_train.append(tweet_combined)
    for (afinn,nrc_vad,nrc_afinn,sia,empath,liu,posnegobj) in zip(tweets_afinn_lists_test,tweets_nrcvad_vectorlist_test,tweets_nrcafinn_score_ohvectors_test,tweets_sia_vector_test,tweets_empath_vector_test,tweets_liu_scores_vectors_test,tweets_posnegobj_test):
        tweet_combined = []
        tweet_combined.extend(afinn)
        for item in nrc_vad:
            tweet_combined.extend(item)
        for item in nrc_afinn:
            tweet_combined.extend(item)
        # tweet_combined.extend(empath)
        tweet_combined.extend(sia)
        tweet_combined.extend(liu)
        for item in posnegobj:
            tweet_combined.extend(item)
        combined_test.append(tweet_combined)
    return (combined_train,combined_test)


tweets_lexicons_combined_train, tweets_lexicons_combined_test = get_tweets_as_combined_lexical_representation()
print(len(tweets_lexicons_combined_train[0]))
# print(tweets_lexicons_combined_train[0])
# print(tweets_lexicons_combined_test[0])


# from model_builder import *

# print("_________________________LEXICONS NN____________________________________")
# print("--------------------LEXICONS--------------------------------------------")
# print('---------LEXICONS targets not considered----------')
# build_train_test_simpleDense_models(tweets_afinn_lists_train,tweets_afinn_lists_test,"afinn")

# build_train_test_models(tweets_nrcvad_vectorlist_train,tweets_nrcvad_vectorlist_test,"nrcvad")
# build_train_test_simpleDense_models(tweets_nrcvad_vectorsummary_train,tweets_nrcvad_vectorsummary_test,"nrcvad_sumamry")

# build_train_test_models(tweets_nrcafinn_score_ohvectors_train,tweets_nrcafinn_score_ohvectors_test,"nrcaffin")
# build_train_test_simpleDense_models(tweets_nrcafinn_summary_train,tweets_nrcafinn_summary_train,"nrcaffin_sumamry")

# build_train_test_models(tweets_posnegobj_train,tweets_posnegobj_test,"posnegobj")
# build_train_test_simpleDense_models(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,"posnegobj_summary")

# build_train_test_simpleDense_models(tweets_sia_vector_train,tweets_sia_vector_test,"vader_sia")
# build_train_test_simpleDense_models(tweets_empath_vector_train,tweets_empath_vector_test,"empath")
# build_train_test_simpleDense_models(tweets_liu_scores_vectors_train,tweets_liu_scores_vectors_test,"liu_vectors")

# build_train_test_models(tweets_posnegobj_train,tweets_posnegobj_test,"sentiwordnet")
# build_train_test_simpleDense_models(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,"sentiwordnet_sumamry")

# build_train_test_simpleDense_models(tweets_lexicons_combined_train,tweets_lexicons_combined_test,"combined_lexicons")

# print("--------------------LEXICONS Targets considered--------------------------------------------")
# build_train_test_simpleDense_models_targets(tweets_afinn_lists_train,tweets_afinn_lists_test,"afinn TARGETS")

# build_train_test_models_targets(tweets_nrcvad_vectorlist_train,tweets_nrcvad_vectorlist_test,"nrcvad TARGETS")
# build_train_test_simpleDense_models_targets(tweets_nrcvad_vectorsummary_train,tweets_nrcvad_vectorsummary_test,"nrcvad_sumamry TARGETS")

# build_train_test_models_targets(tweets_nrcafinn_score_ohvectors_train,tweets_nrcafinn_score_ohvectors_test,"nrcaffin TARGETS")
# build_train_test_simpleDense_models_targets(tweets_nrcafinn_summary_train,tweets_nrcafinn_summary_train,"nrcaffin_sumamry TARGETS")

# build_train_test_simpleDense_models_targets(tweets_sia_vector_train,tweets_sia_vector_test,"vader_sia TARGETS")
# build_train_test_simpleDense_models_targets(tweets_empath_vector_train,tweets_empath_vector_test,"empath TARGETS")
# build_train_test_simpleDense_models_targets(tweets_liu_scores_vectors_train,tweets_liu_scores_vectors_test,"liu_vectors TARGETS")

# build_train_test_models_targets(tweets_posnegobj_train,tweets_posnegobj_test,"sentiwordnet TARGETS")
# build_train_test_simpleDense_models_targets(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,"sentiwordnet_sumamry TARGETS")

# build_train_test_simpleDense_models_targets(tweets_lexicons_combined_train,tweets_lexicons_combined_test,"combined_lexicons TARGETS")

from classifier_builder import *
from sklearn.preprocessing import MinMaxScaler

def positivise_negative_values0(number_vectors_train,number_vectors_test,maxvalue=1):
    pos_train = []
    for i in number_vectors_train:
        pos_train.append(i+maxvalue)
    pos_test = []
    for i in number_vectors_test:
        pos_test.append(i+maxvalue)
    return (pos_train, pos_test)

def positivise_negative_values1(number_vectors_train,number_vectors_test,maxvalue=1):
    pos_train = []
    for vec in number_vectors_train:
        p = []
        for i in vec:
                p.append(i+maxvalue)
        pos_train.append(p)
    pos_test = []
    for vec in number_vectors_test:
        p = []
        for i in vec:
                p.append(i+maxvalue)
        pos_test.append(p)
    return pos_train, pos_test

def positivise_negative_values2(number_vectors_train,number_vectors_test,maxvalue=1):
    pos_train = []
    for vec in number_vectors_train:
        pos = []
        for v in vec:
            pos_v = []
            for i in v:
                pos_v.append(i+maxvalue)
            pos.append(pos_v)
        pos_train.append(pos)
    pos_test = []
    for vec in number_vectors_test:
        pos = []
        for v in vec:
            pos_v = []
            for i in v:
                pos_v.append(i+maxvalue)
            pos.append(pos_v)
        pos_test.append(pos)
    return (pos_train, pos_test)
    

# print("_________________________LEXICONS CLASSIFIERS____________________________________")
# print("--------------------LEXICONS--------------------------------------------")
# print('---------LEXICONS targets not considered----------')

train_in_afinn,test_in_afinn = positivise_negative_values1(tweets_afinn_lists_train,tweets_afinn_lists_test,maxvalue=5)
train_in_nrcvad_s,test_in_nrcvad_s = positivise_negative_values1(tweets_nrcvad_vectorsummary_train,tweets_nrcvad_vectorsummary_test,maxvalue=5)
train_in_nrcafinn_s,test_in_nrcafinn_s = positivise_negative_values1(tweets_nrcafinn_summary_train,tweets_nrcafinn_summary_test,maxvalue=5)
train_in_sia,test_in_sia = positivise_negative_values1(tweets_sia_vector_train,tweets_sia_vector_test,maxvalue=5)
train_in_empath,test_in_empath = positivise_negative_values1(tweets_empath_vector_train,tweets_empath_vector_test,maxvalue=1)
train_in_pno, test_in_pno = positivise_negative_values1(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,maxvalue=1)

# build_and_evaluate_classifier(train_in_afinn,test_in_afinn,name="afinn")
# build_and_evaluate_classifier(train_in_nrcvad_s,test_in_nrcvad_s,name="nrcvad_sumamry")
# build_and_evaluate_classifier(train_in_nrcafinn_s,test_in_nrcafinn_s,name="nrcaffin_sumamry")
# build_and_evaluate_classifier(train_in_sia,test_in_sia,name="vader_sia")
# build_and_evaluate_classifier(train_in_empath,test_in_empath,name="empath")
# build_and_evaluate_classifier(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,name="sentiwordnet_sumamry")




def get_tweets_as_combined_lexical_representation_pos():
    combined_train = []
    combined_test = []
    for (afinn,nrc_vad,nrc_afinn,sia,empath,pno) in zip(train_in_afinn,train_in_nrcvad_s,train_in_nrcafinn_s,train_in_sia,train_in_empath,train_in_pno):
        tweet_combined = []
        tweet_combined.extend(afinn)
        tweet_combined.extend(nrc_vad)
        tweet_combined.extend(nrc_afinn)
        tweet_combined.extend(empath)
        tweet_combined.extend(sia)
        tweet_combined.extend(pno)
        combined_train.append(tweet_combined)
    for (afinn,nrc_vad,nrc_afinn,sia,empath,pno) in zip(test_in_afinn,test_in_nrcvad_s,test_in_nrcafinn_s,test_in_sia,test_in_empath,test_in_pno):
        tweet_combined = []
        tweet_combined.extend(afinn)
        tweet_combined.extend(nrc_vad)
        tweet_combined.extend(nrc_afinn)
        tweet_combined.extend(empath)
        tweet_combined.extend(sia)
        tweet_combined.extend(pno)
        combined_test.append(tweet_combined)
    return (combined_train,combined_test)

train_in_lexicons_combined, test_in_lexicons_combined = get_tweets_as_combined_lexical_representation_pos()
# build_and_evaluate_classifier(train_in_lexicons_combined,test_in_lexicons_combined,name="combined_lexicons TARGETS")

# print("--------------------LEXICONS Targets considered--------------------------------------------")

# build_and_evaluate_classifiers_targets(train_in_afinn,test_in_afinn,name="afinn TARGETS")
# build_and_evaluate_classifiers_targets(train_in_nrcvad_s,test_in_nrcvad_s,name="nrcvad_sumamry TARGETS")
# build_and_evaluate_classifiers_targets(train_in_nrcafinn_s,test_in_nrcafinn_s,name="nrcaffin_sumamry TARGETS")
# build_and_evaluate_classifiers_targets(train_in_sia,test_in_sia,name="vader_sia TARGETS")
# build_and_evaluate_classifiers_targets(train_in_empath,test_in_empath,name="empath TARGETS")
# build_and_evaluate_classifiers_targets(train_in_pno, test_in_pno,name="sentiwordnet_sumamry TARGETS")
# build_and_evaluate_classifiers_targets(train_in_lexicons_combined,test_in_lexicons_combined,name="combined_lexicons TARGETS")
