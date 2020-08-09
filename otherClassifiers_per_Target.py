import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn import svm
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
import numpy as np

from  readDataFromFile import targets_train,targets_test
from  processData import stances_asints_train,stances_asints_test,tweets_intarray_train,tweets_intarray_test,tweets_intarray_padded_train,tweets_intarray_padded_test,tweets_vocabVector_train,tweets_vocabVector_test
from  dealWithNGrams import *
from  dealWithLexicons_afinn_nrc_ import *
from  dealWith_SentiWordNet import tweets_posnegobj_summary_train,tweets_posnegobj_summary_test
from  dealWithPosTagsLemmaStem import *
from  dealWith_Hashtags import *
from  utils import evaluate_classifier,pad_element_list,get_things_separated_by_targets
# from stanceDetectionModel_build_train_evaluate import gram_hash_lexi_summarized_test_pt,gram_hash_lexi_summarized_train_pt

def classify_with_target_knowledge(train_in,test_in,classifier,train_out=stances_asints_train,test_out=stances_asints_test,targets_in = targets_train, targets_out = targets_test):
    train_in_pt = get_things_separated_by_targets(train_in,targets_in)
    train_out_pt = get_things_separated_by_targets(train_out,targets_in)
    
    #['feminist movement', 'climate change is a real concern', 'hillary clinton', 'legalization of abortion', 'atheism']
    classifiers_per_target = dict()
    for target in list(set(targets_train)):
        classifiers_per_target[target] = clone(classifier)
        classifiers_per_target[target].fit(np.array(train_in_pt[target]), np.array(train_out_pt[target]))

    predictions = []
    for (tst_in,target) in zip(test_in,targets_test):
        pred = classifiers_per_target[target].predict(np.array([tst_in]))
        predictions.append(pred[0])
    
    classifier_name = type(classifier).__name__
    evaluate_classifier(predictions,test_out,name=classifier_name)

def build_and_evaluate_classifiers(train_in,test_in,train_out=stances_asints_train,test_out=stances_asints_test,name=""):
    print("--------------------"+name+"---------------------")
    classify_with_target_knowledge(train_in,test_in,BernoulliNB())
    classify_with_target_knowledge(train_in,test_in,GaussianNB())
    classify_with_target_knowledge(train_in,test_in,DecisionTreeClassifier(random_state=42))
    classify_with_target_knowledge(train_in,test_in,ExtraTreeClassifier(random_state=42))   
    classify_with_target_knowledge(train_in,test_in,NearestCentroid())
    classify_with_target_knowledge(train_in,test_in,svm.SVC(decision_function_shape='ovo',degree=5,gamma='auto'))
    classify_with_target_knowledge(train_in,test_in,svm.SVC(decision_function_shape='ovr',degree=5,gamma='auto'))
    classify_with_target_knowledge(train_in,test_in,svm.LinearSVC(max_iter=10000))
    classify_with_target_knowledge(train_in,test_in,RandomForestClassifier())
   
# #tweets as intarrays (oh and vectors)
# print("---TWEETS AS INTARRAYS (onehots and vocab_vector)---")
# build_and_evaluate_classifiers(tweets_vocabVector_train,tweets_vocabVector_test,name="tweetsAs_vocabVector")
# build_and_evaluate_classifiers(tweets_intarray_padded_train,tweets_intarray_padded_test,name="tweetsAs_intArray")

# #Ngrams
# print("---TWEETS NGRAMS (uni,bi,tri,four - onehot and vocab_vector)---")
# build_and_evaluate_classifiers(unigrams_tain_ohp,unigrams_test_ohp,name="1grams_ohp")
# build_and_evaluate_classifiers(bigrams_tain_ohp,bigrams_test_ohp,name="2grams_ohp")
# build_and_evaluate_classifiers(trigrams_tain_ohp,trigrams_test_ohp,name="3grams_ohp")
# build_and_evaluate_classifiers(fourgrams_tain_ohp,fourgrams_test_ohp,name="4grams_ohp")
# build_and_evaluate_classifiers(unigram_train_ohvector,unigram_test_ohvector,name="1gram_ohvector")
# build_and_evaluate_classifiers(bigram_train_ohvector,bigram_test_ohvector,name="2gram_ohvector")
# build_and_evaluate_classifiers(trigram_train_ohvector,trigram_test_ohvector,name="3gram_ohvector")
# build_and_evaluate_classifiers(fourgram_train_ohvector,fourgram_test_ohvector,name="4gram_ohvector")

#Lexicons
print("---TWEETS LEXICON SCORES (AFINN, NRCVADM, NRCAFINN, SIA)---")
build_and_evaluate_classifiers(pad_element_list(tweets_afinn_lists_train),pad_element_list(tweets_afinn_lists_test),name="afinn")
build_and_evaluate_classifiers(tweets_nrcvad_vectorsummary_train,tweets_nrcvad_vectorsummary_test,name="nrcvad_summ")
build_and_evaluate_classifiers(tweets_nrcafinn_summary_train,tweets_nrcafinn_summary_test,name="nrcafinn_summ")
build_and_evaluate_classifiers(tweets_sia_vector_train,tweets_sia_vector_test,name="sia")
build_and_evaluate_classifiers(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,name="posnegobj")
build_and_evaluate_classifiers(tweets_empath_vector_train,tweets_empath_vector_test,name="empath")
# build_and_evaluate_classifiers(tweets_liu_score_train,tweets_liu_score_test,name="liu")

#Lemmed and Stemmed
print("---TWEETS Lemma Stem (onegot and vocab_vector)---")
build_and_evaluate_classifiers(tweets_lemm_onehot_vocab_train,tweets_lemm_onehot_vocab_test,name="lemm_oh_vocab")
build_and_evaluate_classifiers(tweets_stem_onehot_vocab_train,tweets_stem_onehot_vocab_test,name="stem_oh_vocab")
build_and_evaluate_classifiers(tweets_lemm_onehot_vector_train,tweets_lemm_onehot_vector_test,name="lemm_oh_vector")
build_and_evaluate_classifiers(tweets_stem_onehot_vector_train,tweets_stem_onehot_vector_test,name="stemm_oh_vector")

#HASHTAGS
print("---TWEET Hashtags (onehot and vocab_vector)---")
build_and_evaluate_classifiers(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,name="htags_oh")
build_and_evaluate_classifiers(tweets_hashtags_vectors_train,tweets_hashtags_vectors_test,name="htags_vector")

# #COMBINED MODEL
print("--- Combined model ---")
gram_hash_lexi_summarized_train_pt = []
for (gram,hash,afinn,nrcvad,nrcafin,pno,empath) in zip(fourgram_train_ohvector,tweets_hashtags_vectors_train,tweets_afinn_summary_train,tweets_nrcvad_vectorsummary_train,tweets_nrcafinn_summary_train,tweets_posnegobj_summary_train,tweets_empath_vector_train):
    instance = []
    instance.extend(gram)
    instance.extend(hash)
    instance.append(afinn)
    instance.extend(nrcvad)
    instance.extend(nrcafin)
    instance.extend(pno)
    instance.extend(empath)
    gram_hash_lexi_summarized_train_pt.append(instance)

gram_hash_lexi_summarized_test_pt = []
for (gram,hash,afinn,nrcvad,nrcafin,pno,empath) in zip(fourgram_test_ohvector,tweets_hashtags_vectors_test,tweets_afinn_summary_test,tweets_nrcvad_vectorsummary_test,tweets_nrcafinn_summary_test,tweets_posnegobj_summary_test,tweets_empath_vector_test):
    instance = []
    instance.extend(gram)
    instance.extend(hash)
    instance.append(afinn)
    instance.extend(nrcvad)
    instance.extend(nrcafin)
    instance.extend(pno)
    instance.extend(empath)
    gram_hash_lexi_summarized_test_pt.append(instance)
build_and_evaluate_classifiers(gram_hash_lexi_summarized_train_pt,gram_hash_lexi_summarized_test_pt,name="combined")