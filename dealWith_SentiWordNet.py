from readDataFromFile import tweets_wordlists_train, tweets_wordlists_test
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from utils import *
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
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
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """
    lemmatizer = WordNetLemmatizer()

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return [0.0,0.0,0.0]

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return [0.0,0.0,0.0]

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return [0.0,0.0,0.0] #before it was []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

def list_of_wordlists_to_posnegobjscore(wordlists):
    scores = []
    for wordlist in wordlists:
        scores.append(wordlist_to_list_of_posnegobjscore(wordlist))
    return scores

def wordlist_to_list_of_posnegobjscore(wordlist):
    pos_val = nltk.pos_tag(wordlist)
    senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
    return senti_val

def wordlists_to_summary_posnegobjscore(wordlists):
    result = []
    for wordlist in wordlists:
        pos_val = nltk.pos_tag(wordlist)
        senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
        dictionary = {0:0.0,1:0.0,2:0.0}
        for sentval in senti_val:
            dictionary[0] = dictionary[0]+sentval[0]
            dictionary[1] = dictionary[1]+sentval[1]
            dictionary[2] = dictionary[2]+sentval[2]
        result.append([dictionary[0]/len(wordlist),dictionary[1]/len(wordlist),dictionary[2]/len(wordlist)])
    return result

tweets_posnegobj_train = pad_element_list(list_of_wordlists_to_posnegobjscore(tweets_wordlists_train), padtosize=20,padwith=[0.0,0.0,0.0])
tweets_posnegobj_test = pad_element_list(list_of_wordlists_to_posnegobjscore(tweets_wordlists_test), padtosize=20,padwith=[0.0,0.0,0.0])

tweets_posnegobj_summary_train = wordlists_to_summary_posnegobjscore(tweets_wordlists_train)
tweets_posnegobj_summary_test = wordlists_to_summary_posnegobjscore(tweets_wordlists_test)
