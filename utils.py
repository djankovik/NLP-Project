from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Masking, Embedding, LSTM,TimeDistributed,  GRU, SimpleRNN, Bidirectional, Conv2D, Conv1D, MaxPooling2D,MaxPooling1D
import re
import numpy as np

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
    
def get_things_separated_by_targets(things,targets): #'Atheism':[[],[]...,[]] is obtained
    thingsfortarget={}
    for (thing,target) in zip(things,targets):
        if target not in thingsfortarget:
            thingsfortarget[target] = []
        thingsfortarget[target].append(thing)
    return thingsfortarget

def tokenize(text):
    # obtains tokens with a least 1 alphabet
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def pad_element_list(elementlists,padtosize=20,padwith=0.0):
    padded_lists = []
    for elementlist in elementlists:
        padded_list = []
        for element in elementlist:
            if len(padded_list) < padtosize:
                padded_list.append(element)
        while len(padded_list) < padtosize:
            padded_list.append(padwith)
        padded_lists.append(padded_list)
    return padded_lists

def make_np_arrays(arr):
    npified = []
    for a in arr:
        npified.append(np.array(a))
    return np.array(npified)

def make_np_arrays2(arr):
    npified = []
    for ar in arr:
        nparr = []
        for a in ar:
            nparr.append(np.array(a))
        npified.append(np.array(nparr))
    return np.array(npified)

def evaluate_model_predictions(modelpredictions,expected,tabular=False,detailedreport=True, singleLine=False, modelName = " "):
    dictcnt = {"AGAINST":{"total":0,"against":0,"favor":0,"none":0},"FAVOR":{"total":0,"against":0,"favor":0,"none":0},"NONE":{"total":0,"against":0,"favor":0,"none":0}}

    for (prediction,expectation) in zip(modelpredictions,expected):
        exp = expectation.index(max(expectation))
        pred = prediction.tolist().index(max(prediction.tolist()))
        if exp == 0:
            dictcnt["NONE"]["total"] = dictcnt["NONE"]["total"]+1
            if pred == 0:
                dictcnt["NONE"]["none"] = dictcnt["NONE"]["none"]+1
            if pred == 1:
                dictcnt["NONE"]["favor"] = dictcnt["NONE"]["favor"]+1
            if pred == 2:
                dictcnt["NONE"]["against"] = dictcnt["NONE"]["against"]+1
        if exp == 1:
            dictcnt["FAVOR"]["total"] = dictcnt["FAVOR"]["total"]+1
            if pred == 0:
                dictcnt["FAVOR"]["none"] = dictcnt["FAVOR"]["none"]+1
            if pred == 1:
                dictcnt["FAVOR"]["favor"] = dictcnt["FAVOR"]["favor"]+1
            if pred == 2:
                dictcnt["FAVOR"]["against"] = dictcnt["FAVOR"]["against"]+1
        if exp == 2:
            dictcnt["AGAINST"]["total"] = dictcnt["AGAINST"]["total"]+1
            if pred == 0:
                dictcnt["AGAINST"]["none"] = dictcnt["AGAINST"]["none"]+1
            if pred == 1:
                dictcnt["AGAINST"]["favor"] = dictcnt["AGAINST"]["favor"]+1
            if pred == 2:
                dictcnt["AGAINST"]["against"] = dictcnt["AGAINST"]["against"]+1
    
    precision_F = 0.0
    precision_N = 0.0
    precision_A = 0.0
    recall_F = 0.0
    recall_N = 0.0
    recall_A = 0.0
    f1_A = 0.0
    f1_N = 0.0
    f1_F = 0.0

    if dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['favor']+dictcnt["AGAINST"]['favor'] != 0:
        precision_F = (dictcnt["FAVOR"]['favor'])*1.0/(dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['favor']+dictcnt["AGAINST"]['favor'])
    if dictcnt["FAVOR"]['none']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['none'] !=0:
        precision_N = (dictcnt["NONE"]['none'])*1.0/(dictcnt["FAVOR"]['none']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['none'])
    if dictcnt["FAVOR"]['against']+dictcnt["NONE"]['against']+dictcnt["AGAINST"]['against'] != 0:
        precision_A = (dictcnt["AGAINST"]['against'])*1.0/(dictcnt["FAVOR"]['against']+dictcnt["NONE"]['against']+dictcnt["AGAINST"]['against'])
    
    if dictcnt["FAVOR"]['total'] != 0.0:
        recall_F = (dictcnt["FAVOR"]['favor'])*1.0/(dictcnt["FAVOR"]['total'])
    if dictcnt["NONE"]['total'] != 0.0:
        recall_N = (dictcnt["NONE"]['none'])*1.0/(dictcnt["NONE"]['total'])
    if dictcnt["NONE"]['total'] != 0.0:
        recall_A = (dictcnt["AGAINST"]['against'])*1.0/(dictcnt["AGAINST"]['total'])

    accuracy_ttl = (dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['against'])*1.0/(dictcnt["FAVOR"]['total']+dictcnt["NONE"]['total']+dictcnt["AGAINST"]['total'])

    if recall_A+precision_A != 0.0:
        f1_A = 2.0*(recall_A*precision_A)/(recall_A+precision_A)
    if recall_F+precision_F != 0.0:
        f1_F = 2.0*(recall_F*precision_F)/(recall_F+precision_F)
    if recall_N+precision_N != 0.0:
        f1_N = 2.0*(recall_N*precision_N)/(recall_N+precision_N)    
   
    micro_average_p = (dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['against'])*1.0 / (dictcnt["FAVOR"]['against']+dictcnt["NONE"]['against']+dictcnt["AGAINST"]['against']+dictcnt["FAVOR"]['none']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['none']+dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['favor']+dictcnt["AGAINST"]['favor'])
    micro_average_r= (dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['against'])*1.0/(dictcnt["FAVOR"]['total']+dictcnt["NONE"]['total']+dictcnt["AGAINST"]['total'])
    macro_average_p = (precision_F + precision_A + precision_N)/3.0
    macro_average_r = (recall_F+recall_A+recall_N)/3.0

    f1_ttl = (f1_A+f1_N+f1_F)/3.0

    print("Model "+modelName+" results:")
    print(" accuracy: "+str(round(accuracy_ttl,3)))
    print(" f1: "+str(round(f1_ttl,3))+" | "+"F: "+str(round(f1_F,3))+", A: "+str(round(f1_A,3))+", N: "+str(round(f1_N,3)))
    print(" micro average precision: "+str(round(micro_average_p,3)))
    print(" micro average recall: "+str(round(micro_average_r,3)))
    print(" macro average precision: "+str(round(macro_average_p,3))+" | "+"F: "+str(round(precision_F,3))+", A: "+str(round(precision_A,3))+", N: "+str(round(precision_N,3)))
    print(" macro average recall: "+str(round(macro_average_r,3))+" | "+"F: "+str(round(recall_F,3))+", A: "+str(round(recall_A,3))+", N: "+str(round(recall_N,3)))
    print(" (f_favor+f_against)/2: "+str((f1_A+f1_F)/2.0)+" (official metric)")

    return dictcnt

def evaluate_classifier(predicted,expected,onlyofficialmetric=True,name=""):
    dictcnt = {"AGAINST":{"total":0,"against":0,"favor":0,"none":0},"FAVOR":{"total":0,"against":0,"favor":0,"none":0},"NONE":{"total":0,"against":0,"favor":0,"none":0}}

    for (pred,exp) in zip(predicted,expected):
        if exp == 0:
            dictcnt["NONE"]["total"] = dictcnt["NONE"]["total"]+1
            if pred == 0:
                dictcnt["NONE"]["none"] = dictcnt["NONE"]["none"]+1
            if pred == 1:
                dictcnt["NONE"]["favor"] = dictcnt["NONE"]["favor"]+1
            if pred == 2:
                dictcnt["NONE"]["against"] = dictcnt["NONE"]["against"]+1
        if exp == 1:
            dictcnt["FAVOR"]["total"] = dictcnt["FAVOR"]["total"]+1
            if pred == 0:
                dictcnt["FAVOR"]["none"] = dictcnt["FAVOR"]["none"]+1
            if pred == 1:
                dictcnt["FAVOR"]["favor"] = dictcnt["FAVOR"]["favor"]+1
            if pred == 2:
                dictcnt["FAVOR"]["against"] = dictcnt["FAVOR"]["against"]+1
        if exp == 2:
            dictcnt["AGAINST"]["total"] = dictcnt["AGAINST"]["total"]+1
            if pred == 0:
                dictcnt["AGAINST"]["none"] = dictcnt["AGAINST"]["none"]+1
            if pred == 1:
                dictcnt["AGAINST"]["favor"] = dictcnt["AGAINST"]["favor"]+1
            if pred == 2:
                dictcnt["AGAINST"]["against"] = dictcnt["AGAINST"]["against"]+1
    
    precision_F = 0.0
    precision_N = 0.0
    precision_A = 0.0
    recall_F = 0.0
    recall_N = 0.0
    recall_A = 0.0
    f1_A = 0.0
    f1_N = 0.0
    f1_F = 0.0

    if dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['favor']+dictcnt["AGAINST"]['favor'] != 0:
        precision_F = (dictcnt["FAVOR"]['favor'])*1.0/(dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['favor']+dictcnt["AGAINST"]['favor'])
    if dictcnt["FAVOR"]['none']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['none'] !=0:
        precision_N = (dictcnt["NONE"]['none'])*1.0/(dictcnt["FAVOR"]['none']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['none'])
    if dictcnt["FAVOR"]['against']+dictcnt["NONE"]['against']+dictcnt["AGAINST"]['against'] != 0:
        precision_A = (dictcnt["AGAINST"]['against'])*1.0/(dictcnt["FAVOR"]['against']+dictcnt["NONE"]['against']+dictcnt["AGAINST"]['against'])
    
    if dictcnt["FAVOR"]['total'] != 0.0:
        recall_F = (dictcnt["FAVOR"]['favor'])*1.0/(dictcnt["FAVOR"]['total'])
    if dictcnt["NONE"]['total'] != 0.0:
        recall_N = (dictcnt["NONE"]['none'])*1.0/(dictcnt["NONE"]['total'])
    if dictcnt["NONE"]['total'] != 0.0:
        recall_A = (dictcnt["AGAINST"]['against'])*1.0/(dictcnt["AGAINST"]['total'])

    accuracy_ttl = (dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['against'])*1.0/(dictcnt["FAVOR"]['total']+dictcnt["NONE"]['total']+dictcnt["AGAINST"]['total'])

    if recall_A+precision_A != 0.0:
        f1_A = 2.0*(recall_A*precision_A)/(recall_A+precision_A)
    if recall_F+precision_F != 0.0:
        f1_F = 2.0*(recall_F*precision_F)/(recall_F+precision_F)
    if recall_N+precision_N != 0.0:
        f1_N = 2.0*(recall_N*precision_N)/(recall_N+precision_N)    
   
    micro_average_p = (dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['against'])*1.0 / (dictcnt["FAVOR"]['against']+dictcnt["NONE"]['against']+dictcnt["AGAINST"]['against']+dictcnt["FAVOR"]['none']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['none']+dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['favor']+dictcnt["AGAINST"]['favor'])
    micro_average_r= (dictcnt["FAVOR"]['favor']+dictcnt["NONE"]['none']+dictcnt["AGAINST"]['against'])*1.0/(dictcnt["FAVOR"]['total']+dictcnt["NONE"]['total']+dictcnt["AGAINST"]['total'])
    macro_average_p = (precision_F + precision_A + precision_N)/3.0
    macro_average_r = (recall_F+recall_A+recall_N)/3.0

    f1_ttl = (f1_A+f1_N+f1_F)/3.0

    if onlyofficialmetric == False:
        print("\t"+name+":")
        print(" accuracy: "+str(round(accuracy_ttl,3)))
        print(" f1: "+str(round(f1_ttl,3))+" | "+"F: "+str(round(f1_F,3))+", A: "+str(round(f1_A,3))+", N: "+str(round(f1_N,3)))
        print(" micro average precision: "+str(round(micro_average_p,3)))
        print(" micro average recall: "+str(round(micro_average_r,3)))
        print(" macro average precision: "+str(round(macro_average_p,3))+" | "+"F: "+str(round(precision_F,3))+", A: "+str(round(precision_A,3))+", N: "+str(round(precision_N,3)))
        print(" macro average recall: "+str(round(macro_average_r,3))+" | "+"F: "+str(round(recall_F,3))+", A: "+str(round(recall_A,3))+", N: "+str(round(recall_N,3)))
    print(" (f_favor+f_against)/2: "+str((f1_A+f1_F)/2.0)+" (official metric)")

    return dictcnt