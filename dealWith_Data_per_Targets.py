from  readDataFromFile import *
from  dealWithPosTagsLemmaStem import *
from  dealWithEmbeddings import *
from  dealWith_SentiWordNet import *
from  dealWithLexicons_afinn_nrc_ import *
from  dealWithNGrams import *

import zipfile

def get_things_separated_by_targets(things,targets): #'Atheism':[[],[]...,[]] is obtained
    thingsfortarget={}
    for (thing,target) in zip(things,targets):
        if target not in thingsfortarget:
            thingsfortarget[target] = []
        thingsfortarget[target].append(thing)
    return thingsfortarget

#Plain text stances
stances_per_target_train = get_things_separated_by_targets(stances_train,targets_train)
stances_per_target_test = get_things_separated_by_targets(stances_test,targets_test)

#Onehot stances
stancesOH_per_target_train = get_things_separated_by_targets(stances_onehot_train,targets_train)
stancesOH_per_target_test = get_things_separated_by_targets(stances_onehot_test,targets_test)

#Plain Text tweets
tweets_per_target_train = get_things_separated_by_targets(tweets_train,targets_train)
tweets_per_target_test = get_things_separated_by_targets(tweets_test,targets_test)

#Word Lists
tweetsWL_per_target_train = get_things_separated_by_targets(tweets_train,targets_train)
tweetsWL_per_target_test = get_things_separated_by_targets(tweets_test,targets_test)

