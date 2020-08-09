from preprocess import *
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import re
import json
import pandas as pd
import seaborn as sns
from collections import Counter
import nltk
import more_itertools as mit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return wn.NOUN #None

def get_sentiment(word,tag):

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return [0.0,0.0,0.0]

    lemma = word
    if not lemma:
        return [0.0,0.0,0.0]

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return [0.0,0.0,0.0] #before it was []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

def lemmas_to_SentiWordNet_pno(lemmas_lists_train,lemmas_lists_test):
    lemmas_pnos_train = []
    lemmas_pnos_summary_train = []

    for lemmas_list in lemmas_lists_train:
        pos_val = nltk.pos_tag(lemmas_list)
        senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
        lemmas_pnos_train.append(senti_val)
        lemmas_pnos_summary_train.append(np.average(senti_val, axis=0))
    
    lemmas_pnos_test = []
    lemmas_pnos_summary_test = []
    for lemmas_list in lemmas_lists_test:
        pos_val = nltk.pos_tag(lemmas_list)
        senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
        lemmas_pnos_test.append(senti_val)
        lemmas_pnos_summary_test.append(np.average(senti_val, axis=0))
    return (lemmas_pnos_train,lemmas_pnos_test)


empath_lexicon = Empath()
empath_lexicon.create_category("hillary_clinton",["twitter","hillary_clinton","woman_president","hillary","politics",'clinton','woman','sexism'])
empath_lexicon.create_category("atheism",["twitter","god","science","faith","believe",'bible','religion','jesus','sin'])
empath_lexicon.create_category("abortion",["twitter","pregnancy","women","choice","baby",'poor','rape',"planned_parenhood"])
empath_lexicon.create_category("feminism",["twitter","womens_rights","woman","equality","pay_gap",'sexism','abuse','metoo'])
empath_lexicon.create_category("climate_change",["twitter","climate_change","greenhouse_effect","pollution","heat_wave","waste","recycle"])
empath_categories = ['speaking', 'dominant_personality', 'anger', 'leader', 'worship', 'vacation', 'driving', 'phone', 'irritability', 'home', 'party', 'strength', 'heroic', 'swimming', 'wedding', 'sexual', 'appearance', 'deception', 'exercise', 'masculine', 'terrorism', 'blue_collar_job', 'medieval', 'neglect', 'dispute', 'cheerfulness', 'weather', 'programming', 'injury', 'plant', 'writing', 'reading', 'warmth', 'medical_emergency', 'weapon', 'exasperation', 'water', 'play', 'breaking', 'fabric', 'pride', 'business', 'royalty', 'zest', 'optimism', 'childish', 'negotiate', 'white_collar_job', 'fun', 'disgust', 'urban', 'sadness', 'achievement', 'hiking', 'traveling', 'morning', 'internet', 'furniture', 'healing', 'confusion', 'gain', 'ugliness', 'beauty', 'smell', 'abortion', 'aggression', 'sleep', 'politics', 'violence', 'noise', 'tool', 'tourism', 'communication', 'cold', 'surprise', 'exotic', 'meeting', 'pain', 'hate', 'stealing', 'fight', 'fashion', 'dance', 'crime', 'liquid', 'economics', 'fire', 'farming', 'domestic_work', 'leisure', 'war', 'envy', 'fear', 'listen', 'anonymity', 'car', 'positive_emotion', 'health', 'family', 'office', 'emotional', 'journalism', 'cleaning', 'suffering', 'rage', 'science', 'wealthy', 'youth', 'legend', 'attractive', 'superhero', 'shame', 'dominant_heirarchical', 'power', 'shape_and_size', 'school', 'ship', 'children', 'messaging', 'affection', 'animal', 'horror', 'government', 'ocean', 'eating', 'money', 'art', 'rural', 'ancient', 'law', 'occupation', 'feminism', 'lust', 'sports', 'celebration', 'military', 'sailing', 'death', 'shopping', 'torment', 'kill', 'joy', 'independence', 'air_travel', 'hearing', 'poor', 'cooking', 'pet', 'magic', 'trust', 'disappointment', 'monster', 'valuable', 'alcohol', 'hillary_clinton', 'restaurant', 'nervousness', 'sympathy', 'atheism', 'love', 'anticipation', 'body', 'work', 'help', 'hipster', 'night', 'computer', 'prison', 'timidity', 'real_estate', 'friends', 'weakness', 'technology', 'payment', 'competing', 'hygiene', 'clothing', 'politeness', 'swearing_terms', 'negative_emotion', 'feminine', 'toy', 'religion', 'banking', 'college', 'ridicule', 'giving', 'musical', 'sound', 'divine', 'movement', 'beach', 'vehicle', 'philosophy', 'climate_change', 'order', 'music', 'contentment', 'social_media']

AFINN_lexicon = {}
NRCVAD_lexicon = {}
NRCAFINN_lexicon = {}
LIU_lexicon = {"positive":[],"negative":[]}

def set_AFINN_lexicon():
    with open(r'lexicons\AFINN-111.txt') as f1:
        for line in f1:
            word, score = line.split('\t')
            AFINN_lexicon[word] = float(score)

def set_NRCVAD_lexicon():
    with open(r'lexicons\NRC-VAD.txt') as f1:
        next(f1)
        for line in f1:
            word, valence, arousal, dominance = line.split('\t')
            NRCVAD_lexicon[word] = [float(valence), float(arousal), float(dominance)]

def set_NRCAffectIntensity_lexicon():
    emotions = ['joy', 'fear', 'anger', 'sadness']
    with open(r'lexicons\NRC-AffectIntensity.txt') as f1:
        for line in f1:
            word, score, emotion = line.split(' ')
            emot_vec = [0.0,0.0,0.0,0.0]
            emot_vec[emotions.index(emotion.strip())] = float(score)
            NRCAFINN_lexicon[word] = emot_vec

def set_Liu_lexicon():
    positive_words = []
    negative_words = []
    with open('lexicons\liu_positive-words.txt') as file:
        line = file.readline() # to pass through the first line that has column headers
        while line:
            LIU_lexicon['positive'].append(line.strip())
            line = file.readline()
    with open('lexicons\liu_negative-words.txt') as file:
        line = file.readline() # to pass through the first line that has column headers
        while line:
            LIU_lexicon['negative'].append(line.strip())
            line = file.readline()

set_AFINN_lexicon()
set_NRCVAD_lexicon()
set_NRCAffectIntensity_lexicon()
set_Liu_lexicon()

def tokens_to_AFINN(tokens_lists_train,tokens_lists_test): #integer list of affinscores for words for each wordlist
    afinn_scores_train = []
    afinn_scores_summaries_train = []
    for tokens_list in tokens_lists_train:
        afinnscr = []
        for token in tokens_list:
            if word in AFINN_lexicon:
                afinnscr.append(AFINN_lexicon[word])
            else:
                afinnscr.append(0.0) #default 0
        afinn_scores_train.append(afinnscr)
        afinn_scores_summaries_train.append(sum(afinnscr)*1.0/len(afinnscr))

    afinn_scores_test = []
    afinn_scores_summaries_test = []
    for tokens_list in tokens_lists_test:
        afinnscr = []
        for token in tokens_list:
            if word in AFINN_lexicon:
                afinnscr.append(AFINN_lexicon[word])
            else:
                afinnscr.append(0.0) #default 0
        afinn_scores_test.append(afinnscr)
        afinn_scores_summaries_test.append(sum(afinnscr)*1.0/len(afinnscr))
    return (afinn_scores_train,afinn_scores_test,afinn_scores_summaries_train,afinn_scores_summaries_test)

def tokens_to_NRCVAD(tokens_lists_train,tokens_lists_test): #[valence,arousal,dominance] vector for each word in each wordlist
    nrc_vad_train = []
    nrc_vad_summary_train = []
    for tokens_list in tokens_lists_train:
        nrcvadscr = []
        for token in tokens_list:
            if token in NRCVAD_lexicon:
                nrcvadscr.append(NRCVAD_lexicon[word])
            else:
                nrcvadscr.append([0.0,0.0,0.0]) #default vector
        nrc_vad_train.append(nrcvadscr)
        nrc_vad_summary_train.append(np.average(nrcvadscr,axis=0))

    nrc_vad_test = []
    nrc_vad_summary_test = []
    for tokens_list in tokens_lists_test:
        nrcvadscr = []
        for token in tokens_list:
            if token in NRCVAD_lexicon:
                nrcvadscr.append(NRCVAD_lexicon[word])
            else:
                nrcvadscr.append([0.0,0.0,0.0]) #default vector
        nrc_vad_test.append(nrcvadscr)
        nrc_vad_summary_test.append(np.average(nrcvadscr,axis=0))

    return (nrc_vad_train,nrc_vad_test,nrc_vad_summary_train,nrc_vad_summary_test)

def tokens_to_NRCAFINN(tokens_lists_train,tokens_lists_test):
    nrc_afinn_train = []
    nrc_afinn_summary_train = []
    for tokens_list in tokens_lists_train:
        nrcafinnscr = []
        for token in tokens_list:
            if token in NRCAFINN_lexicon:
                nrcafinnscr.append(NRCAFINN_lexicon[token])
            else:
                nrcafinnscr.append([0.0,0.0,0.0,0.0])
        nrc_afinn_train.append(nrcafinnscr)
        nrc_afinn_summary_train.append(np.average(nrcafinnscr,axis=0))
    
    nrc_afinn_test = []
    nrc_afinn_summary_test = []
    for tokens_list in tokens_lists_test:
        nrcafinnscr = []
        for token in tokens_list:
            if token in NRCAFINN_lexicon:
                nrcafinnscr.append(NRCAFINN_lexicon[token])
            else:
                nrcafinnscr.append([0.0,0.0,0.0,0.0])
        nrc_afinn_test.append(nrcafinnscr)
        nrc_afinn_summary_test.append(np.average(nrcafinnscr,axis=0))
    return (nrc_afinn_train,nrc_afinn_test,nrc_afinn_summary_train,nrc_afinn_summary_test)

def tweets_to_VADER_nnpc(tweets_train,tweets_test): #SentimentIntensityAnalyzer [neg,neu,pos,compound] vector per tweet
    vader_train = []
    vader_test = []
    analyser = SentimentIntensityAnalyzer()
    for tweet in tweets_train:
        score = analyser.polarity_scores(tweet)
        vader_train.append([score['neg'],score['neu'],score['pos'],score['compound']])
    for tweet in tweets_test:
        score = analyser.polarity_scores(tweet)
        vader_test.append([score['neg'],score['neu'],score['pos'],score['compound']])
    return (vader_train,vader_test)

def tweets_to_Empath(tweets_train,tweets_test): 
    empath_train = []
    for tweet in tweets_train:
        score = empath_lexicon.analyze(tweet, normalize=True)
        vector = []
        for key in empath_categories:
            vector.append(score[key])
        empath_train.append(vector)
    empath_test = []
    for tweet in tweets_test:
        score = empath_lexicon.analyze(tweet, normalize=True)
        vector = []
        for key in empath_categories:
            vector.append(score[key])
        empath_test.append(vector)
    return (empath_train,empath_test)

def tokens_to_Liu(tokens_lists_train,tokens_lists_test):
    liu_train = []
    liu_summary_train = []
    liu_test = []
    liu_summary_test = []

    for tokens_list in tokens_lists_train:
        liuscores = []
        cnt=0
        for token in tokens_list:
            if token in LIU_lexicon['positive']:
                liuscores.append(1)
                cnt+=1
            if token in LIU_lexicon['negative']:
                liuscores.append(-1)
                cnt+=1
        liu_train.append(liuscores)
        liu_summary_train.append(sum(liuscores)*1.0/cnt)
    for tokens_list in tokens_lists_test:
        liuscores = []
        cnt=0
        for token in tokens_list:
            if token in LIU_lexicon['positive']:
                liuscores.append(1)
                cnt+=1
            if token in LIU_lexicon['negative']:
                liuscores.append(-1)
                cnt+=1
        liu_test.append(liuscores)
        liu_summary_test.append(sum(liuscores)*1.0/cnt)
    return (liu_train,liu_test,liu_summary_train,liu_summary_test)


words_afinn_train,words_afinn_test,words_afinn_sum_train,words_afinn_sum_test = tokens_to_AFINN(words_train,words_test)
lemmas_afinn_train,lemmas_afinn_test,lemmas_afinn_sum_train,lemmas_afinn_sum_test = tokens_to_AFINN(lemmas_train,lemmas_test)
stems_afinn_train,stems_afinn_test,stems_afinn_sum_train,stems_afinn_sum_test = tokens_to_AFINN(stems_train,stems_test)
words_ns_afinn_train,words_ns_afinn_test,words_ns_afinn_sum_train,words_ns_afinn_sum_test = tokens_to_AFINN(words_nostops_train,words_nostops_test)
lemmas_ns_afinn_train,lemmas_ns_afinn_test,lemmas_ns_afinn_sum_train,lemmas_ns_afinn_sum_test = tokens_to_AFINN(lemmas_nostops_train,lemmas_nostops_test)
stems_ns_afinn_train,stems_ns_afinn_test,stems_ns_afinn_sum_train,stems_ns_afinn_sum_test = tokens_to_AFINN(stems_nostops_train,stems_nostops_test)
htags_afinn_train,htags_afinn_test,htags_afinn_sum_train,htags_afinn_sum_test = tokens_to_AFINN(htags_train,htags_test)
ners_afinn_train,ners_afinn_test,ners_afinn_sum_train,ners_afinn_sum_test = tokens_to_AFINN(ners_train,ners_test)
nouns_afinn_train,nouns_afinn_test,nouns_afinn_sum_train,nouns_afinn_sum_test = tokens_to_AFINN(nouns_train,nouns_test)

words_nrcvad_train,words_nrcvad_test,words_nrcvad_sum_train,words_nrcvad_sum_test = tokens_to_NRCVAD(words_train,words_test)
lemmas_nrcvad_train,lemmas_nrcvad_test,lemmas_nrcvad_sum_train,lemmas_nrcvad_sum_test = tokens_to_NRCVAD(lemmas_train,lemmas_test)
stems_nrcvad_train,stems_nrcvad_test,stems_nrcvad_sum_train,stems_nrcvad_sum_test = tokens_to_NRCVAD(stems_train,stems_test)
words_ns_nrcvad_train,words_ns_nrcvad_test,words_ns_nrcvad_sum_train,words_ns_nrcvad_sum_test = tokens_to_NRCVAD(words_nostops_train,words_nostops_test)
lemmas_ns_nrcvad_train,lemmas_ns_nrcvad_test,lemmas_ns_nrcvad_sum_train,lemmas_ns_nrcvad_sum_test = tokens_to_NRCVAD(lemmas_nostops_train,lemmas_nostops_test)
stems_ns_nrcvad_train,stems_ns_nrcvad_test,stems_ns_nrcvad_sum_train,stems_ns_nrcvad_sum_test = tokens_to_NRCVAD(stems_nostops_train,stems_nostops_test)
htags_nrcvad_train,htags_nrcvad_test,htags_nrcvad_sum_train,htags_nrcvad_sum_test = tokens_to_NRCVAD(htags_train,htags_test)
ners_nrcvad_train,ners_nrcvad_test,ners_nrcvad_sum_train,ners_nrcvad_sum_test = tokens_to_NRCVAD(ners_train,ners_test)
nouns_nrcvad_train,nouns_nrcvad_test,nouns_nrcvad_sum_train,nouns_nrcvad_sum_test = tokens_to_NRCVAD(nouns_train,nouns_test)

words_nrcafinn_train,words_nrcafinn_test,words_nrcafinn_sum_train,words_nrcafinn_sum_test = tokens_to_NRCAFINN(words_train,words_test)
lemmas_nrcafinn_train,lemmas_nrcafinn_test,lemmas_nrcafinn_sum_train,lemmas_nrcafinn_sum_test = tokens_to_NRCAFINN(lemmas_train,lemmas_test)
stems_nrcafinn_train,stems_nrcafinn_test,stems_nrcafinn_sum_train,stems_nrcafinn_sum_test = tokens_to_NRCAFINN(stems_train,stems_test)
words_ns_nrcafinn_train,words_ns_nrcafinn_test,words_ns_nrcafinn_sum_train,words_ns_nrcafinn_sum_test = tokens_to_NRCAFINN(words_nostops_train,words_nostops_test)
lemmas_ns_nrcafinn_train,lemmas_ns_nrcafinn_test,lemmas_ns_nrcafinn_sum_train,lemmas_ns_nrcafinn_sum_test = tokens_to_NRCAFINN(lemmas_nostops_train,lemmas_nostops_test)
stems_ns_nrcafinn_train,stems_ns_nrcafinn_test,stems_ns_nrcafinn_sum_train,stems_ns_nrcafinn_sum_test = tokens_to_NRCAFINN(stems_nostops_train,stems_nostops_test)
htags_nrcafinn_train,htags_nrcafinn_test,htags_nrcafinn_sum_train,htags_nrcafinn_sum_test = tokens_to_NRCAFINN(htags_train,htags_test)
ners_nrcafinn_train,ners_nrcafinn_test,ners_nrcafinn_sum_train,ners_nrcafinn_sum_test = tokens_to_NRCAFINN(ners_train,ners_test)
nouns_nrcafinn_train,nouns_nrcafinn_test,nouns_nrcafinn_sum_train,nouns_nrcafinn_sum_test = tokens_to_NRCAFINN(nouns_train,nouns_test)

words_liu_train,words_liu_test,words_liu_sum_train,words_liu_sum_test = tokens_to_Liu(words_train,words_test)
lemmas_liu_train,lemmas_liu_test,lemmas_liu_sum_train,lemmas_liu_sum_test = tokens_to_Liu(lemmas_train,lemmas_test)
stems_liu_train,stems_liu_test,stems_liu_sum_train,stems_liu_sum_test = tokens_to_Liu(stems_train,stems_test)
words_ns_liu_train,words_ns_liu_test,words_ns_liu_sum_train,words_ns_liu_sum_test = tokens_to_Liu(words_nostops_train,words_nostops_test)
lemmas_ns_liu_train,lemmas_ns_liu_test,lemmas_ns_liu_sum_train,lemmas_ns_liu_sum_test = tokens_to_Liu(lemmas_nostops_train,lemmas_nostops_test)
stems_ns_liu_train,stems_ns_liu_test,stems_ns_liu_sum_train,stems_ns_liu_sum_test = tokens_to_Liu(stems_nostops_train,stems_nostops_test)
htags_liu_train,htags_liu_test,htags_liu_sum_train,htags_liu_sum_test = tokens_to_Liu(htags_train,htags_test)
ners_liu_train,ners_liu_test,ners_liu_sum_train,ners_liu_sum_test = tokens_to_Liu(ners_train,ners_test)
nouns_liu_train,nouns_liu_test,nouns_liu_sum_train,nouns_liu_sum_test = tokens_to_Liu(nouns_train,nouns_test)

lemmas_swn_train,lemmas_swn_test,lemmas_swn_sum_train,lemmas_swn_sum_test = lemmas_to_SentiWordNet_pno(lemmas_train,lemmas_test)
lemmas_ns_swn_train,lemmas_ns_swn_test,lemmas_ns_swn_sum_train,lemmas_ns_swn_sum_test = lemmas_to_SentiWordNet_pno(lemmas_nostops_train,lemmas_nostops_test)

tweets_vader_train,tweets_vader_test = tweets_to_VADER_nnpc(tweets_train,tweets_test)
tweets_empath_train, tweets_empath_test = tweets_to_Empath(tweets_train,tweets_test)


