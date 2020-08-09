from dealWithNER import tweets_ner_vocabohvec_train,tweets_ner_vocabohvec_test
from dealWithLexicons_afinn_nrc_ import tweets_lexicons_combined_train, tweets_lexicons_combined_test,train_in_lexicons_combined, test_in_lexicons_combined,tweets_empath_vector_train,tweets_empath_vector_test
from dealWithNGrams import *
from dealWithPosTagsLemmaStem import *
from dealWith_Hashtags import tweets_hashtags_onehots_train,tweets_hashtags_onehots_test
from readDataFromFile import *

def getcombos(forclassifiers = False):
    if forclassifiers:
        lexicombo_train = train_in_lexicons_combined
        lexicombo_test = test_in_lexicons_combined
    else:
        lexicombo_train = tweets_lexicons_combined_train
        lexicombo_test = tweets_lexicons_combined_test

    combos_train = []
    combos_test = []
    for (ner,lexicombo,gram1,gram4,hashtags,lemmvec,stemvec) in zip(tweets_ner_vocabohvec_train,lexicombo_train,unigrams_per_target_onehot_train,fourgrams_per_target_onehot_train,tweets_hashtags_onehots_train,tweets_lemm_onehot_vocab_train,tweets_stem_onehot_vocab_train):
        combo_train=[]
        combo_train.extend(ner)
        combo_train.extend(lexicombo)
        combo_train.extend(gram1)
        combo_train.extend(gram4)
        combo_train.extend(hashtags)
        combo_train.extend(lemmvec)
        combo_train.extend(stemvec)
        combos_train.append(combo_train)
    for (ner,lexicombo,gram1,gram4,hashtags,lemmvec,stemvec) in zip(tweets_ner_vocabohvec_test,lexicombo_test,unigrams_per_target_onehot_test,fourgrams_per_target_onehot_test,tweets_hashtags_onehots_test,tweets_lemm_onehot_vocab_test,tweets_stem_onehot_vocab_test):
        combo_test=[]
        combo_test.extend(ner)
        combo_test.extend(lexicombo)
        combo_test.extend(gram1)
        combo_test.extend(gram4)
        combo_test.extend(hashtags)
        combo_test.extend(lemmvec)
        combo_test.extend(stemvec)
        combos_test.append(combo_test)
    return (combos_train,combos_test)
    
train_in_c,test_in_c = getcombos(forclassifiers=True)
train_in,test_in = getcombos()
print("LENGTHSSSSSS")
print(len(train_in_c[0]))
print(len(train_in[0]))

# from model_builder import *
# print("________________________COMBINED (ner,lexicombo,gram1,gram4,hashtags,lemmvec,stemvec) NN________________________")
# print("----------------COMBINED targets not considered-----------------------")
# build_train_test_simpleDense_models(train_in,train_in,"combinedALL")

# print("----------------COMBINED targets considered-----------------------")
# build_train_test_simpleDense_models_targets(train_in,test_in,"combinedALL TARGETS")

# from classifier_builder import *

# print("________________________COMBINED (ner,lexicombo,gram1,gram4,hashtags,lemmvec,stemvec) CLASSIFIERS________________________")
# print("----------------COMBINED targets not considered-----------------------")
# build_and_evaluate_classifier(train_in_c,test_in_c,name="combinedALL")

# print("----------------COMBINED targets considered-----------------------")
# build_and_evaluate_classifiers_targets(train_in,test_in,name="combinedALL TARGETS")
