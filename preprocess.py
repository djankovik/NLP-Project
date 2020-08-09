from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import math
from nltk.tag import pos_tag
import spacy
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()

train_data_path = r'data\semeval2016-task6-trainingdata.txt'
test_data_path = r'data\SemEval2016-Task6-subtaskA-testdata-gold.txt'

import emoji
import regex

def get_emojis(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)
    return emoji_list

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

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def words_to_lemmas(words):
    tweet_lemmas = []
    tags = pos_tag(words)
    for (word,tag) in zip(words,tags):
        tweet_lemmas.append(lemmatizer.lemmatize(word,pos=get_wordnet_pos(tag[1])))
    return tweet_lemmas

def words_to_stems(words):
    tweet_stems = []
    for word in words:
        tweet_stems.append(stemmer.stem(word))
    return tweet_stems

def read_data(filesrc):
    targets = []
    tweets = []
    stances = []
    with open(filesrc) as file:
        line = file.readline() # to pass through the first line that has column headers
        line = file.readline()
        while line:
            parts = re.split(r'\t+', line)
            targets.append(parts[1].strip().lower())
            tweets.append(parts[2].strip().replace("#SemST",""))
            stances.append(parts[3].strip())
            line = file.readline()
    return [targets,tweets,stances]

def get_nouns(tweet_tokens):
    tagged = pos_tag(tweet_tokens)
    nouns = []
    for (token,tag) in tagged:
        if 'NN' in tag:
            nouns.append(token)
    return nouns

spacy_entity_types = ["PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE","MONEY","ORDINAL"]
def get_NER(tweet_text):
    doc = nlp(tweet_text)
    tokens_tagged = [(X.text.lower(), X.label_) for X in doc.ents]
    tweet_ners =[]
    for (word,tag) in tokens_tagged:
        if tag in spacy_entity_types:
            tweet_ners.append(word)
    return tweet_ners

def filter_stopwords(words):
    filtered = []
    for word in words:
        if word not in stopwords.words('english'):
            filtered.append(word)
    return filtered

htags_dict = {}
hashtag_parts_vocabulary = set()
def get_hashtags_words_translation(hashtags):
    for htag in hashtags:
        if htag.lower() == htag:
            htags_dict[htag] = [htag]
            hashtag_parts_vocabulary.add(htag)
        else:
            parts = re.findall('[A-Z][^A-Z]+', htag.replace('#',''))
            word_list = [Htag.lower() for Htag in parts]
            htags_dict[htag.lower()] = word_list
            hashtag_parts_vocabulary.update(word_list)

def tokenize_tweets(tweets):
    words = []
    words_nostops = []
    lemmas = []
    stems = []
    lemmas_nostops = []
    stems_nostops = []
    hashtags = []
    users = []
    for tweet in tweets:
        htags = [word.replace('#','') for word in re.findall(r'#[a-zA-Z0-9_]+',tweet.lower())]
        get_hashtags_words_translation([word.replace('#','') for word in re.findall(r'#[a-zA-Z0-9_]+',tweet)])
        usersmentioned = re.findall(r'@[a-zA-Z0-9_]+',tweet.lower())
        filteredtweet = re.sub(r'[^A-Za-z0-9?!\']+',' ',re.sub(r'(#)|(@[a-zA-Z0-9_]+)',' ',tweet.lower()))
        filteredtweetwords = word_tokenize(filteredtweet.lower())
        filteredtweetlemmas = words_to_lemmas(filteredtweetwords)
        filteredtweetstems = words_to_stems(filteredtweetwords)
        words.append(filteredtweetwords)
        words_nostops.append(filter_stopwords(filteredtweetwords))
        lemmas.append(filteredtweetlemmas)
        lemmas_nostops.append(filter_stopwords(filteredtweetwords))
        stems.append(filteredtweetstems)
        stems_nostops.append(filter_stopwords(filteredtweetwords))
        hashtags.append(htags)
        users.append(usersmentioned)
        ners = get_NER(tweet)
        nouns = get_nouns(filteredtweetwords)
    return (words,words_nostops,lemmas,lemmas_nostops,stems,stems_nostops,hashtags,ners,nouns,users)
        
def get_sorted_dictionary_vocab(tweets):
    dictionary = {}
    average = 0
    for tweet in tweets:
        average +=len(tweet)
        for nn in tweet:
            if nn in dictionary:
                dictionary[nn] += 1
            else:
                dictionary[nn] = 1
    sorteddict_asc = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    return (sorteddict_asc, list(sorteddict_asc),math.ceil(average*1.0/len(tweets)))

def get_ngrams_for_tweets(tweets,n):
    tweets_ngrams = []
    for tweet in tweets:
        tweet_ngrams = []
        for i in range(len(tweet)-n+1):
            ngram = []
            for j in range(0,n):
                ngram.append(tweet[i+j])             
            tweet_ngrams.append(tuple(ngram))
        tweets_ngrams.append(tweet_ngrams)
    return tweets_ngrams

targets_train,tweets_train,stances_train = read_data(train_data_path)
targets_test,tweets_test,stances_test = read_data(test_data_path)

stances_vocab = {'NONE':0,'FAVOR':1,'AGAINST':2}
targets_vocab = {'climate change is a real concern':0, 'legalization of abortion':1, 'hillary clinton':2, 'atheism':3, 'feminist movement':4} 

words_train,words_nostops_train,lemmas_train,lemmas_nostops_train,stems_train,stems_nostops_train,htags_train,ners_train,nouns_train,users_train = tokenize_tweets(tweets_train)
words_test,words_nostops_test,lemmas_test,lemmas_nostops_test,stems_test,stems_nostops_test,htags_test,ners_test,nouns_test,users_test = tokenize_tweets(tweets_test)

_,words_vocabulary,avg_w = get_sorted_dictionary_vocab(words_train)
_,lemmas_vocabulary,avg_l = get_sorted_dictionary_vocab(lemmas_train)
_,stems_vocabulary,avg_s = get_sorted_dictionary_vocab(stems_train)
_,words_nostops_vocabulary,avg_wns = get_sorted_dictionary_vocab(words_nostops_train)
_,lemmas_nostops_vocabulary,avg_lns = get_sorted_dictionary_vocab(lemmas_nostops_train)
_,stems_nostops_vocabulary,avg_sns = get_sorted_dictionary_vocab(stems_nostops_train)
_,htags_vocabulary,avg_h = get_sorted_dictionary_vocab(htags_train)
_,ners_vocabulary,avg_ner = get_sorted_dictionary_vocab(ners_train)
_,nouns_vocabulary,avg_noun = get_sorted_dictionary_vocab(nouns_train)
_,users_vocabulary,avg_u = get_sorted_dictionary_vocab(users_train)

print("words: "+str(len(words_vocabulary))+" ("+str(avg_w)+") | words_NS: "+str(len(words_nostops_vocabulary))+" ("+str(avg_wns)+") | lemmas: "+str(len(lemmas_vocabulary))+" ("+str(avg_l)+") | lemmas_nostops: "+str(len(lemmas_nostops_vocabulary))+" ("+str(avg_lns)+") | stems: "+str(len(stems_vocabulary))+" ("+str(avg_s)+") | stems_nostops: "+str(len(stems_nostops_vocabulary))+" ("+str(avg_sns)+") | htags: "+str(len(htags_vocabulary))+" ("+str(avg_h)+") | users: "+str(len(users_vocabulary))+" ("+str(avg_u)+")")

_,words_vocabulary_test,_ = get_sorted_dictionary_vocab(words_test)
_,lemmas_vocabulary_test,_ = get_sorted_dictionary_vocab(lemmas_test)
_,stems_vocabulary_test,_ = get_sorted_dictionary_vocab(stems_test)
_,ners_vocabulary_test,_ = get_sorted_dictionary_vocab(ners_test)
_,nouns_vocabulary_test,_ = get_sorted_dictionary_vocab(nouns_test)