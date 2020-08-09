from  readDataFromFile import stances_onehot_test,stances_onehot_train,targets_test,targets_train
from  dealWithPosTagsLemmaStem import *
from  dealWithEmbeddings import *
from  dealWith_SentiWordNet import *
from  dealWithLexicons_afinn_nrc_ import *
from  dealWithNGrams import *
from  dealWith_Hashtags import *
from  processData import TW_FILTERED_WORD_LENGTH,TW_STOP_WORD_LENGTH,TW_WORD_LENGTH
from  utils import *

def get_simpler_model(inputshape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=inputshape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def get_multidimensional_model():
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) #return_sequences = true ako sakam odma posle ova Tiem DIstributed
    model.add(TimeDistributed(Dense(32, activation='relu'))) #3 d vlez dimension time
    model.add(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) #return_sequences = true ako sakam odma posle ova Tiem DIstributed
    model.add(TimeDistributed(Dense(128, activation='relu'))) #3 d vlez dimension time
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) #return_sequences = true ako sakam odma posle ova Tiem DIstributed
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def get_multidimensional_embeddings_model(vocabsize,inputlength):
    model = Sequential()
    model.add(Embedding(vocabsize,50,weights=[embedding_matrix],input_length=inputlength,trainable=False))
    model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(TimeDistributed(Dense(32, activation='relu'))) 
    model.add(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
    model.add(TimeDistributed(Dense(128, activation='relu'))) 
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def create_and_run_multidim_embeddings_model_per_target(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test,vocabsize = 9432,inputlength=20,modelName=" "):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = make_np_arrays(train_out_per_targets[target])
        model = get_multidimensional_embeddings_model(vocabsize,inputlength)
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model

    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True,modelName=modelName)
    return models_predictions

def create_and_run_multidim_embeddings_model(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test,vocabsize = 9432,inputlength=20,modelName=" "):
    model = get_multidimensional_embeddings_model(vocabsize,inputlength)
    model.fit(make_np_arrays(train_in), np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(make_np_arrays(test_in))
    evaluate_model_predictions(y_pred,test_out,detailedreport=True,singleLine=True,modelName=modelName)
    return y_pred

def create_and_run_model_per_target(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test,inputshape =(20,), modelName=' '):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = make_np_arrays(train_out_per_targets[target])
        model = get_simpler_model(inputshape)
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0)
        models_per_targets[target] = model

    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True,modelName=modelName)
    return models_predictions

def create_and_run_model(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test,inputshape=(20,),modelName=" "): #receive numpy arrays as input train and input test
    model = get_simpler_model(inputshape)
    model.fit(make_np_arrays(train_in), np.array(train_out), batch_size=32, epochs=15, verbose=0)
    y_pred = model.predict(make_np_arrays(test_in))
    evaluate_model_predictions(y_pred,test_out,detailedreport=True,singleLine=True,modelName=modelName)
    return y_pred

def create_and_run_multidim_model_per_target(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test,inputshape =(20,), modelName=' '):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = make_np_arrays(train_out_per_targets[target])
        model = get_multidimensional_model()
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0)
        models_per_targets[target] = model

    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True,modelName=modelName)
    return models_predictions

def create_and_run_multidim_model(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test,inputshape=(20,), modelName=" "):
    model = get_multidimensional_model()
    model.fit(make_np_arrays(train_in), np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(make_np_arrays(test_in))
    evaluate_model_predictions(y_pred,test_out,detailedreport=True,singleLine=True,modelName=modelName)
    return y_pred


# print("______ GLOVE EMBEDDINGS (raw, filtered unimportant, stopwords filtered) ______")
# print("-----targets not considered-----")
# predictions_raw = create_and_run_multidim_embeddings_model(tweets_intarray_padded,tweets_intarray_test_padded,inputlength=TW_WORD_LENGTH,modelName=" gloveRaw ")
# predictions_fltr = create_and_run_multidim_embeddings_model(tweets_intarray_filtered_padded,tweets_intarray_filtered_test_padded,inputlength=TW_FILTERED_WORD_LENGTH,modelName=" gloveFilterOverlap ")
# predictions_stops = create_and_run_multidim_embeddings_model(tweets_intarray_filteredstops_padded,tweets_intarray_filteredstops_test_padded,inputlength=TW_STOP_WORD_LENGTH,modelName=" gloveFilterStops ")

# print("-----targets considered-----")
# predictions_raw_t = create_and_run_multidim_embeddings_model_per_target(tweets_intarray_padded,tweets_intarray_test_padded,inputlength=TW_WORD_LENGTH,modelName=" gloveRaw_t ")
# predictions_fltr_t = create_and_run_multidim_embeddings_model_per_target(tweets_intarray_filtered_padded,tweets_intarray_filtered_test_padded,inputlength=TW_FILTERED_WORD_LENGTH,modelName=" gloveFilterOverlap_t ")
# predictions_stops_t = create_and_run_multidim_embeddings_model_per_target(tweets_intarray_filteredstops_padded,tweets_intarray_filteredstops_test_padded,inputlength=TW_STOP_WORD_LENGTH,modelName=" gloveFilterStops_t ")

# print("______ LEMMATIZED/STEMMED (lemma onehot, stem oneot, lemma vocab vector, stem vocab vector) ______")
# print("-----targets not considered-----")
# predictions_lemm_oh = create_and_run_model(tweets_lemm_onehot_vocab_train,tweets_lemm_onehot_vocab_test,inputshape=(len(tweets_lemm_onehot_vocab_train[0]),), modelName=" lemmaOnehot ")
# predictions_stem_oh = create_and_run_model(tweets_stem_onehot_vocab_train,tweets_stem_onehot_vocab_test,inputshape=(len(tweets_stem_onehot_vocab_train[0]),),modelName=" stemOnehot ")
# predictions_lemm_vec = create_and_run_model(tweets_lemm_onehot_vector_train,tweets_lemm_onehot_vector_test,inputshape=(len(tweets_lemm_onehot_vector_train[0]),),modelName="lemmaVector ")
# predictions_stem_vec = create_and_run_model(tweets_stem_onehot_vector_train,tweets_stem_onehot_vector_test,inputshape=(len(tweets_stem_onehot_vector_train[0]),),modelName=" stemVector ")

# print("-----targets considered-----")
# predictions_lemm_oh_t = create_and_run_model_per_target(tweets_lemm_onehot_vocab_train,tweets_lemm_onehot_vocab_test,inputshape=(len(tweets_lemm_onehot_vocab_train[0]),),modelName=" lemmaOnehot_t ")
# predictions_stem_oh_t = create_and_run_model_per_target(tweets_stem_onehot_vocab_train,tweets_stem_onehot_vocab_test,inputshape=(len(tweets_stem_onehot_vocab_train[0]),), modelName=" stemOnehot_t ")
# predictions_lemm_vec_t = create_and_run_model_per_target(tweets_lemm_onehot_vector_train,tweets_lemm_onehot_vector_test,inputshape=(len(tweets_lemm_onehot_vector_train[0]),),modelName="lemmaVector_t ")
# predictions_stem_vec_t = create_and_run_model_per_target(tweets_stem_onehot_vector_train,tweets_stem_onehot_vector_test,inputshape=(len(tweets_stem_onehot_vector_train[0]),),modelName=" stemVector_t ")

print("______ LEXICONS (afinn,afinn sum, nrcvad, nrcvad sum, nrcafinn, nrcafinn sum, vader, empath, liu) ______")
print("-----targets not considered-----")
tweets_afinn_lists_train_padded = pad_element_list(tweets_afinn_lists_train)
tweets_afinn_lists_test_padded = pad_element_list(tweets_afinn_lists_test)
predictions_afin = create_and_run_multidim_model(tweets_afinn_lists_train_padded,tweets_afinn_lists_test_padded, inputshape=(len(tweets_afinn_lists_train[0]),), modelName=" afinn ")
# predictions_afinsum = create_and_run_model(tweets_afinn_summary_train,tweets_afinn_summary_test,inputshape=(1,), modelName=" afinnSummary ")
# tweets_nrcvad_vectorlist_train_padded = pad_element_list(tweets_nrcvad_vectorlist_train,padwith=[0.0,0.0,0.0])
# tweets_nrcvad_vectorlist_test_padded = pad_element_list(tweets_nrcvad_vectorlist_test,padwith=[0.0,0.0,0.0])
# predictions_nrcvad = create_and_run_multidim_model(tweets_nrcvad_vectorlist_train_padded,tweets_nrcvad_vectorlist_test_padded,inputshape=(20,3,), modelName=" nrcVad ")
# predictions_nrcvadsum = create_and_run_model(tweets_nrcvad_vectorsummary_train,tweets_nrcvad_vectorsummary_test,inputshape=(3,), modelName=" nrcVadSummary ")
# tweets_nrcafinn_score_ohvectors_train_padded = pad_element_list(tweets_nrcafinn_score_ohvectors_train,padwith=[0.0,0.0,0.0,0.0])
# tweets_nrcafinn_score_ohvectors_test_padded = pad_element_list(tweets_nrcafinn_score_ohvectors_test,padwith=[0.0,0.0,0.0,0.0])
# predictions_nrcaffin = create_and_run_multidim_model(tweets_nrcafinn_score_ohvectors_train_padded,tweets_nrcafinn_score_ohvectors_test_padded,inputshape=(20,4,), modelName=" nrcAfinn ")
# predictions_nrcaffin_sum = create_and_run_model(tweets_nrcafinn_summary_train,tweets_nrcafinn_summary_test,inputshape=(4,), modelName=" nrcAfinnSummary ")
# predictions_sia = create_and_run_model(tweets_sia_vector_train,tweets_sia_vector_test,inputshape=(4,), modelName=" siaLexion ")
# predictions_empath = create_and_run_model(tweets_empath_vector_train,tweets_empath_vector_test,inputshape=(len(tweets_empath_vector_train[0]),),modelName=" empathLexicon ")
# predictions_liu = create_and_run_model(tweets_liu_score_train,tweets_liu_score_test,inputshape=(1,),modelName=" liuLexicon ")

# print("-----targets considered-----")
# predictions_afin_t = create_and_run_model_per_target(tweets_afinn_lists_train_padded,tweets_afinn_lists_test_padded, modelName=" afinn_t ")
# predictions_afinsum_t = create_and_run_model_per_target(tweets_afinn_summary_train,tweets_afinn_summary_test,inputshape=(1,), modelName=" afinnSummary_t ")
# predictions_nrcvad_t = create_and_run_multidim_model_per_target(tweets_nrcvad_vectorlist_train_padded,tweets_nrcvad_vectorlist_test_padded,inputshape=(20,3,), modelName=" nrcVad_t ")
# predictions_nrcvadsum_t = create_and_run_model_per_target(tweets_nrcvad_vectorsummary_train,tweets_nrcvad_vectorsummary_test,inputshape=(3,), modelName=" nrcVadSummary_t ")
# predictions_nrcaffin_t = create_and_run_multidim_model_per_target(tweets_nrcafinn_score_ohvectors_train_padded,tweets_nrcafinn_score_ohvectors_test_padded,inputshape=(20,4,), modelName=" nrcAfinn_t ")
# predictions_nrcaffin_sum_t = create_and_run_model_per_target(tweets_nrcafinn_summary_train,tweets_nrcafinn_summary_test,inputshape=(4,), modelName=" nrcAfinnSummary_t ")
# predictions_sia_t = create_and_run_model_per_target(tweets_sia_vector_train,tweets_sia_vector_test,inputshape=(4,), modelName=" siaLexion_t ")
# predictions_empath_t = create_and_run_model_per_target(tweets_empath_vector_train,tweets_empath_vector_test,inputshape=(len(tweets_empath_vector_train[0]),),modelName=" empathLexicon_t ")
# predictions_liu_t = create_and_run_model_per_target(tweets_liu_score_train,tweets_liu_score_test,inputshape=(1,),modelName=" liuLexicon_t ")

# print("______ NGRAMS ([uni bi tri four] -> onehots, vectors, vectors but custom per target) ______")
# print("-----targets not considered-----")
# predictions_uni = create_and_run_model(unigrams_tain_ohp,unigrams_test_ohp,inputshape=(len(unigrams_tain_ohp[0]),),modelName=" unigramsOnehot ")
# predictions_bi = create_and_run_model(bigrams_tain_ohp,bigrams_test_ohp,inputshape=(len(bigrams_tain_ohp[0]),),modelName=" bigramsOnehot ")
# predictions_tri = create_and_run_model(trigrams_tain_ohp,trigrams_test_ohp,inputshape=(len(trigrams_tain_ohp[0]),),modelName=" trigramsOnehot ")
# predictions_four = create_and_run_model(fourgrams_tain_ohp,fourgrams_test_ohp,inputshape=(len(fourgrams_tain_ohp[0]),),modelName=" fourgramsOnehot ")

# predictions_uni_vec = create_and_run_model(unigram_train_ohvector,unigram_test_ohvector,inputshape=(len(unigram_train_ohvector[0]),),modelName=" unigramsVector ")
# predictions_bi_vec = create_and_run_model(bigram_train_ohvector,bigram_test_ohvector,inputshape=(len(bigram_train_ohvector[0]),),modelName=" bigramsVector ")
# predictions_tri_vec = create_and_run_model(trigram_train_ohvector,trigram_test_ohvector,inputshape=(len(trigram_train_ohvector[0]),),modelName=" trigramsVector ")
# predictions_four_vec = create_and_run_model(fourgram_train_ohvector,fourgram_test_ohvector,inputshape=(len(fourgram_train_ohvector[0]),),modelName=" fourgramsVector ")

# predictions_uni_specvec = create_and_run_model(unigrams_per_target_onehot_train,unigrams_per_target_onehot_test,inputshape=(len(unigrams_per_target_onehot_train[0]),),modelName=" unigramsTargetsVector ")
# predictions_bi_specvec = create_and_run_model(bigrams_per_target_onehot_train,bigrams_per_target_onehot_test,inputshape=(len(bigrams_per_target_onehot_train[0]),),modelName=" bigramsTargetsVector ")
# predictions_tri_specvec = create_and_run_model(trigrams_per_target_onehot_train,trigrams_per_target_onehot_test,inputshape=(len(trigrams_per_target_onehot_train[0]),),modelName=" trigramsTargetsVector ")
# predictions_four_specvec = create_and_run_model(fourgrams_per_target_onehot_train,fourgrams_per_target_onehot_test,inputshape=(len(fourgrams_per_target_onehot_train[0]),),modelName=" fourgramsTargetsVector ")

# print("-----targets considered-----")
# predictions_uni_t = create_and_run_model_per_target(unigrams_tain_ohp,unigrams_test_ohp,inputshape=(len(unigrams_tain_ohp[0]),),modelName=" unigramsOnehot_t ")
# predictions_bi_t = create_and_run_model_per_target(bigrams_tain_ohp,bigrams_test_ohp,inputshape=(len(bigrams_tain_ohp[0]),),modelName=" bigramsOnehot_t ")
# predictions_tri_t = create_and_run_model_per_target(trigrams_tain_ohp,trigrams_test_ohp,inputshape=(len(trigrams_tain_ohp[0]),),modelName=" trigramsOnehot_t ")
# predictions_four_t = create_and_run_model_per_target(fourgrams_tain_ohp,fourgrams_test_ohp,inputshape=(len(fourgrams_tain_ohp[0]),),modelName=" fourgramsOnehot_t ")

# predictions_uni_vec_t = create_and_run_model_per_target(unigram_train_ohvector,unigram_test_ohvector,inputshape=(len(unigram_train_ohvector[0]),),modelName=" unigramsVector_t ")
# predictions_bi_vec_t = create_and_run_model_per_target(bigram_train_ohvector,bigram_test_ohvector,inputshape=(len(bigram_train_ohvector[0]),),modelName=" bigramsVector_t ")
# predictions_tri_vec_t = create_and_run_model_per_target(trigram_train_ohvector,trigram_test_ohvector,inputshape=(len(trigram_train_ohvector[0]),),modelName=" trigramsVector_t ")
# predictions_four_vec_t = create_and_run_model_per_target(fourgram_train_ohvector,fourgram_test_ohvector,inputshape=(len(fourgram_train_ohvector[0]),),modelName=" fourgramsVector_t ")

# predictions_uni_specvec_t = create_and_run_model_per_target(unigrams_per_target_onehot_train,unigrams_per_target_onehot_test,inputshape=(len(unigrams_per_target_onehot_train[0]),),modelName=" unigramsTargetsVector_t ")
# predictions_bi_specvec_t = create_and_run_model_per_target(bigrams_per_target_onehot_train,bigrams_per_target_onehot_test,inputshape=(len(bigrams_per_target_onehot_train[0]),),modelName=" bigramsTargetsVector_t ")
# predictions_tri_specvec_t = create_and_run_model_per_target(trigrams_per_target_onehot_train,trigrams_per_target_onehot_test,inputshape=(len(trigrams_per_target_onehot_train[0]),),modelName=" trigramsTargetsVector_t ")
# predictions_four_specvec_t = create_and_run_model_per_target(fourgrams_per_target_onehot_train,fourgrams_per_target_onehot_test,inputshape=(len(fourgrams_per_target_onehot_train[0]),),modelName=" fourgramsTargetsVector_t ")


# print("______ NO EMBEDDINGS - jusr INTARRAYS (all, filtered, stopw filtered) ______")
# print("-----targets not considered-----")
# predictions_iar = create_and_run_model(tweets_intarray_padded,tweets_intarray_test_padded,inputshape=(len(tweets_intarray_padded[0]),), modelName=" intArray ")
# predictions_iarfilt = create_and_run_model(tweets_intarray_filtered_padded,tweets_intarray_filtered_test_padded,inputshape=(len(tweets_intarray_filtered_padded[0]),), modelName=" intArrayFilteredCommon ")
# predictions_iarstop = create_and_run_model(tweets_intarray_filteredstops_padded,tweets_intarray_filteredstops_test_padded,inputshape=(len(tweets_intarray_filteredstops_padded[0]),), modelName=" intArrayFilteredStops ")

# print("-----targets considered-----")
# predictions_iar_t = create_and_run_model_per_target(tweets_intarray_padded,tweets_intarray_test_padded,inputshape=(len(tweets_intarray_padded[0]),), modelName=" intArray_t ")
# predictions_iarfilt_t = create_and_run_model_per_target(tweets_intarray_filtered_padded,tweets_intarray_filtered_test_padded,inputshape=(len(tweets_intarray_filtered_padded[0]),), modelName=" intArrayFilteredCommon_t ")
# predictions_iarstop_t = create_and_run_model_per_target(tweets_intarray_filteredstops_padded,tweets_intarray_filteredstops_test_padded,inputshape=(len(tweets_intarray_filteredstops_padded[0]),), modelName=" intArrayFilteredStops_t ")

# print("______ SENTIWORDNET (posnegobj, posnegobj summary) ______")
# print("-----targets not considered-----")
# tweets_posnegobj_train_padded = pad_element_list(tweets_posnegobj_train,padwith=[0.0,0.0,0.0])
# tweets_posnegobj_test_padded = pad_element_list(tweets_posnegobj_test,padwith=[0.0,0.0,0.0])
# predictions_pno = create_and_run_multidim_model(tweets_posnegobj_train_padded,tweets_posnegobj_test_padded,inputshape=(len(tweets_posnegobj_train_padded[0]),len(tweets_posnegobj_train_padded[0][0]),), modelName=" posnegobj ")
# predictions_pno_summary =  create_and_run_model(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,inputshape=(len(tweets_posnegobj_summary_train[0]),), modelName=" posnegobjSummary ")

# print("-----targets considered-----")
# predictions_pno_t = create_and_run_multidim_model_per_target(tweets_posnegobj_train_padded,tweets_posnegobj_test_padded,inputshape=(len(tweets_posnegobj_train_padded[0]),len(tweets_posnegobj_train_padded[0][0]),), modelName=" posnegobj_t ")
# predictions_pno_summary_t =  create_and_run_model_per_target(tweets_posnegobj_summary_train,tweets_posnegobj_summary_test,inputshape=(len(tweets_posnegobj_summary_train[0]),), modelName=" posnegobjSummary_t ")

# print("______ HASHTAGS AS INTARRAYS (vectors (len vocab), onehots(len 5)) ______")
# print("-----targets not considered-----")
# predictions_hashtags_vec = create_and_run_model(tweets_hashtags_vectors_train,tweets_hashtags_vectors_test,inputshape=(len(tweets_hashtags_vectors_train[0]),), modelName=" hashtags_vector ")
# predictions_hashtags_oh = create_and_run_model(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,inputshape=(len(tweets_hashtags_onehots_train[0]),), modelName=" hashtags_ohot ")
# print("-----targets considered-----")
# predictions_hashtags_vec_t = create_and_run_model_per_target(tweets_hashtags_vectors_train,tweets_hashtags_vectors_test,inputshape=(len(tweets_hashtags_vectors_train[0]),), modelName=" hashtags_vector_t ")
# predictions_hashtags_oh_t = create_and_run_model_per_target(tweets_hashtags_onehots_train,tweets_hashtags_onehots_test,inputshape=(len(tweets_hashtags_onehots_train[0]),), modelName=" hashtags_ohot_t ")

# print("______ 4-Grams, Hashtags, Lexicon summaries - all concatenated ______")
# print("-----targets not considered-----")
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

# create_and_run_model(gram_hash_lexi_summarized_train,gram_hash_lexi_summarized_test,inputshape=(len(gram_hash_lexi_summarized_train[0]),),modelName=" gram_hash_lexi_summarized ")
# print("-----targets considered-----")
# gram_hash_lexi_summarized_train_pt = []
# for (gram,hash,afinn,nrcvad,nrcafin,pno,empath) in zip(fourgram_train_ohvector,tweets_hashtags_vectors_train,tweets_afinn_summary_train,tweets_nrcvad_vectorsummary_train,tweets_nrcafinn_summary_train,tweets_posnegobj_summary_train,tweets_empath_vector_train):
#     instance = []
#     instance.extend(gram)
#     instance.extend(hash)
#     instance.append(afinn)
#     instance.extend(nrcvad)
#     instance.extend(nrcafin)
#     instance.extend(pno)
#     instance.extend(empath)
#     gram_hash_lexi_summarized_train_pt.append(instance)

# gram_hash_lexi_summarized_test_pt = []
# for (gram,hash,afinn,nrcvad,nrcafin,pno,empath) in zip(fourgram_test_ohvector,tweets_hashtags_vectors_test,tweets_afinn_summary_test,tweets_nrcvad_vectorsummary_test,tweets_nrcafinn_summary_test,tweets_posnegobj_summary_test,tweets_empath_vector_test):
#     instance = []
#     instance.extend(gram)
#     instance.extend(hash)
#     instance.append(afinn)
#     instance.extend(nrcvad)
#     instance.extend(nrcafin)
#     instance.extend(pno)
#     instance.extend(empath)
#     gram_hash_lexi_summarized_test_pt.append(instance)
# create_and_run_model_per_target(gram_hash_lexi_summarized_train_pt,gram_hash_lexi_summarized_test_pt,inputshape=(len(gram_hash_lexi_summarized_train_pt[0]),),modelName=" gram_hash_lexi_summarized per target ")