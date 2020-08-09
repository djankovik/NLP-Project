from sklearn import svm
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn import metrics
import numpy as np

from  processData import *
from  dealWithNGrams import *
from  dealWithLexicons_afinn_nrc_ import *
from  dealWith_SentiWordNet import tweets_posnegobj_summary_train,tweets_posnegobj_summary_test
from  dealWithPosTagsLemmaStem import *
from dealWithNER import *
from  dealWith_Hashtags import *
from  utils import evaluate_classifier,pad_element_list

def run_classifier(train_in,test_in,classifier,train_out=stances_asints_train,test_out=stances_asints_test,name=""):
    if name == "":
        name = type(classifier).__name__
    clf = classifier
    clf.fit(train_in, train_out)
    y_pred = clf.predict(test_in)
    print(' '+name)
    print("\taccuracy:",metrics.accuracy_score(test_out, y_pred))
    print("\tpecision | pweighted: "+str(metrics.precision_score(test_out, y_pred,average="weighted"))+", pmicro: "+str(metrics.precision_score(test_out, y_pred,average="micro"))+", pmacro: "+str(metrics.precision_score(test_out, y_pred,average="macro")))
    print("\trecall | rweighted: "+str(metrics.recall_score(test_out, y_pred,average="weighted"))+", rmicro: "+str(metrics.recall_score(test_out, y_pred,average="micro"))+", rmacro: "+str(metrics.recall_score(test_out, y_pred,average="macro")))
    print("\tf1 | f1weighted: "+str(metrics.f1_score(test_out, y_pred,average="weighted"))+", f1micro: "+str(metrics.f1_score(test_out, y_pred,average="micro"))+", f1macro: "+str(metrics.f1_score(test_out, y_pred,average="macro")))
    evaluate_classifier(y_pred,test_out,name)

def build_and_evaluate_classifier(train_in,test_in,train_out=stances_asints_train,test_out=stances_asints_test, name = ""):
    print("---------------"+str(name)+"-----------------------")
    run_classifier(train_in, test_in,MultinomialNB())
    run_classifier(train_in, test_in,BernoulliNB())
    run_classifier(train_in, test_in,GaussianNB())
    run_classifier(train_in, test_in,DecisionTreeClassifier())   
    run_classifier(train_in, test_in,ExtraTreeClassifier(random_state=42))
    run_classifier(train_in, test_in,NearestCentroid())
    run_classifier(train_in, test_in,KNeighborsClassifier())
    run_classifier(train_in, test_in,svm.SVC(kernel='rbf',decision_function_shape='ovo',degree=5,gamma='auto'),name="SVM ovo")
    run_classifier(train_in, test_in,svm.SVC(kernel='rbf',decision_function_shape='ovr',degree=5,gamma='auto'),name="SVM ovr")
    run_classifier(train_in, test_in,svm.LinearSVC(max_iter=10000),name="LinearSVM")
    run_classifier(train_in, test_in,RandomForestClassifier(random_state=42))


#tweets as intarrays (oh and vectors)
print("---TWEETS AS INTARRAYS (onehots and vocab_vector)---")
build_and_evaluate_classifier(tweets_vocabVector_train,tweets_vocabVector_test,name="tweetsAs_vocabVector")
build_and_evaluate_classifier(tweets_intarray_padded_train,tweets_intarray_padded_test,name="tweetsAs_intArray")
build_and_evaluate_classifier(tweets_wordOhs_train,tweets_wordOhs_test,name="tweets_wordOhs") #each word encoded as OH array

#Ngrams
print("---TWEETS NGRAMS (uni,bi,tri,four - onehot and vocab_vector)---")
build_and_evaluate_classifier(unigrams_tain_ohp,unigrams_test_ohp,name="1grams_ohp")
build_and_evaluate_classifier(bigrams_tain_ohp,bigrams_test_ohp,name="2grams_ohp")
build_and_evaluate_classifier(trigrams_tain_ohp,trigrams_test_ohp,name="3grams_ohp")
build_and_evaluate_classifier(fourgrams_tain_ohp,fourgrams_test_ohp,name="4grams_ohp")
build_and_evaluate_classifier(unigram_train_ohvector,unigram_test_ohvector,name="1gram_ohvector")
build_and_evaluate_classifier(bigram_train_ohvector,bigram_test_ohvector,name="2gram_ohvector")
build_and_evaluate_classifier(trigram_train_ohvector,trigram_test_ohvector,name="3gram_ohvector")
build_and_evaluate_classifier(fourgram_train_ohvector,fourgram_test_ohvector,name="4gram_ohvector")
build_and_evaluate_classifier(unigrams_per_target_onehot_train,unigrams_per_target_onehot_test,name="1gram_ohvector_pertarget")
build_and_evaluate_classifier(bigrams_per_target_onehot_train,bigrams_per_target_onehot_test,name="2gram_ohvector_pertarget")
build_and_evaluate_classifier(trigrams_per_target_onehot_train,trigrams_per_target_onehot_test,name="3gram_ohvector_pertarget")
build_and_evaluate_classifier(fourgrams_per_target_onehot_train,fourgrams_per_target_onehot_test,name="4gram_ohvector_pertarget")

# #Lexicons
print("---TWEETS LEXICON SCORES (AFINN, NRCVADM, NRCAFINN, SIA)---")
build_and_evaluate_classifier(pad_element_list(tweets_afinn_lists_train),pad_element_list(tweets_afinn_lists_test),name="afinn")
build_and_evaluate_classifier(tweets_nrcvad_vectorsummary_train,tweets_nrcvad_vectorsummary_test,name="nrcvad_summ")
build_and_evaluate_classifier(tweets_nrcafinn_summary_train,tweets_nrcafinn_summary_test,name="nrcafinn_summ")
build_and_evaluate_classifier(tweets_sia_vector_train,tweets_sia_vector_test,name="sia")
build_and_evaluate_classifier(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,name="posnegobj")
build_and_evaluate_classifier(tweets_empath_vector_train,tweets_empath_vector_test,name="empath")
build_and_evaluate_classifier(tweets_liu_score_train,tweets_liu_score_test,name="liu")

# #Lemmed and Stemmed
# print("---TWEETS Lemma Stem (onegot and vocab_vector)---")
# build_and_evaluate_classifier(tweets_lemm_onehot_vocab_train,tweets_lemm_onehot_vocab_test,name="lemm_oh_vocab")
# build_and_evaluate_classifier(tweets_stem_onehot_vocab_train,tweets_stem_onehot_vocab_test,name="stem_oh_vocab")
# build_and_evaluate_classifier(tweets_lemm_onehot_vector_train,tweets_lemm_onehot_vector_test,name="lemm_oh_vector")
# build_and_evaluate_classifier(tweets_stem_onehot_vector_train,tweets_stem_onehot_vector_test,name="stemm_oh_vector")

# #HASHTAGS
# print("---TWEET Hashtags (onehot and vocab_vector)---")
# build_and_evaluate_classifier(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,name="htags_oh")
# build_and_evaluate_classifier(tweets_hashtags_vectors_train,tweets_hashtags_vectors_test,name="htags_vector")

#NER
build_and_evaluate_classifier(tweets_ner_vocabohvec_train,tweets_ner_vocabohvec_test,name="ner_vector")
build_and_evaluate_classifier(tweets_ner_ohvec_train,tweets_ner_ohvec_test,name="ner_oh")

# # #COMBINED MODEL

# print("______ 4-Grams, Hashtags, Lexicon summaries - all concatenated ______")
# gram_hash_lexi_summarized_train = []
# for (gram,hash,afinn,nrcvad,nrcafin,pno,empath) in zip(fourgrams_per_target_onehot_train,tweets_hashtags_vectors_train,tweets_afinn_summary_train,tweets_nrcvad_vectorsummary_train,tweets_nrcafinn_summary_train,tweets_posnegobj_summary_train,tweets_empath_vector_train):
#     instance = []
#     instance.extend(gram)
#     instance.extend(hash)
#     instance.append(afinn)
#     instance.extend(nrcvad)
#     instance.extend(nrcafin)
#     instance.extend(pno)
#     instance.extend(empath)
#     gram_hash_lexi_summarized_train.append(instance)

# gram_hash_lexi_summarized_test = []
# for (gram,hash,afinn,nrcvad,nrcafin,pno,empath) in zip(fourgrams_per_target_onehot_test,tweets_hashtags_vectors_test,tweets_afinn_summary_test,tweets_nrcvad_vectorsummary_test,tweets_nrcafinn_summary_test,tweets_posnegobj_summary_test,tweets_empath_vector_test):
#     instance = []
#     instance.extend(gram)
#     instance.extend(hash)
#     instance.append(afinn)
#     instance.extend(nrcvad)
#     instance.extend(nrcafin)
#     instance.extend(pno)
#     instance.extend(empath)
#     gram_hash_lexi_summarized_test.append(instance)

# print("--- Combined model ---")
# build_and_evaluate_classifier(gram_hash_lexi_summarized_train,gram_hash_lexi_summarized_test,name="combined")