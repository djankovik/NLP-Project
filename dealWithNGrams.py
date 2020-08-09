from readDataFromFile import *
from utils import *

def get_ngrams_for_targets(tweets_ngrams, ngrams_vocab, targets_tweets):
    dictionary = {}
    vocabularies = {}
    for target in list(set(targets_tweets)):
        dictionary[target] = {}
    #print(dictionary)

    for (tweet_ngrams,target) in zip(tweets_ngrams,targets_tweets):
        for ngram in tweet_ngrams:
            if ngram in dictionary[target]:
                dictionary[target][ngram] = dictionary[target][ngram] +1
            else:
                dictionary[target][ngram] = 1
    
    for target in list(set(targets_tweets)):
        dictionary[target] = {k: v for k, v in sorted(dictionary[target].items(), key=lambda item: item[1], reverse=True)}
        vocabularies[target] = list(dictionary[target].keys())

    return dictionary,vocabularies

def get_onehot_for_tweet_knowing_target_and_vector(dictionary, vocabularies, tweets_ngrams, targets_tweets,vectorlength = 100):
    onehots = []
    for (tweet,target) in zip(tweets_ngrams,targets_tweets):
        onehot = []
        oh = get_onehot_given_vector(tweet,vocabularies[target][0:vectorlength])
        if target == 'feminist movement':
            onehot.extend(oh)
            onehot.extend([0]*vectorlength*4)
        if target == 'climate change is a real concern':
            onehot.extend([0]*vectorlength)
            onehot.extend(oh)
            onehot.extend([0]*vectorlength*3)
        if target == 'legalization of abortion':
            onehot.extend([0]*vectorlength*2)
            onehot.extend(oh)
            onehot.extend([0]*vectorlength*2)
        if target == 'atheism':
            onehot.extend([0]*vectorlength*3)
            onehot.extend(oh)
            onehot.extend([0]*vectorlength)
        if target == 'hillary clinton':
            onehot.extend([0]*vectorlength*4)
            onehot.extend(oh)
        onehots.append(onehot)
    return onehots
            
def get_length_stats_of_tweets_as_ngrams(tweets_ngrams):
    max = 0
    min = 100
    avg = 0.0
    total = 0
    for tweet_ngrams in tweets_ngrams:
        length = len(tweet_ngrams)
        total += length
        if length > max:
            max = length
        if length < min:
            min = length
    avg = total / len(tweets_ngrams)
    #print("Ngram stats: "+"avg: "+str(avg)+", min: "+str(min)+", avg: "+str(max))
    return avg,min,max

def get_ngram_sorted_dictionary_vocab(tweets_ngr):
    dictionary = {}
    for tweet in tweets_ngr:
        for ngram in tweet:
            if ngram in dictionary:
                dictionary[ngram] += 1
            else:
                dictionary[ngram] = 1
    sorteddict_asc = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    return (sorteddict_asc, list(sorteddict_asc))

def get_ngrams_for_tweets(tweets,n):
    tweets_ngrams = []
    for tweet in tweets:
        words = tokenize(tweet)
        tweet_ngrams = []
        for i in range(len(words)-n+1):
            ngram = []
            for j in range(0,n):
                ngram.append(words[i+j])             
            tweet_ngrams.append(tuple(ngram))
        tweets_ngrams.append(tweet_ngrams)
    return tweets_ngrams

def get_unigrams_for_tweets(tweets):
    tweet_unigrams = []
    for tweet in tweets:
        tweet_unigrams.append(tokenize(tweet))
    return tweet_unigrams

def get_bigrams_for_tweets(tweets):
    tweet_bigrams = []
    for tweet in tweets:
        words = tokenize(tweet)
        bgrms = []
        for i in range(len(words)-1):
            bgrms.append((words[i],words[i+1]))
        tweet_bigrams.append(bgrms)
    return tweet_bigrams

def get_ngrams_onehot_encoded(tweets_ngrams,vocabulary):
    onehot_tweets = []
    for tweet_ngrams in tweets_ngrams:
        onehot_tweet = []
        for ngram in tweet_ngrams:
            if ngram in vocabulary:
                onehot_tweet.append(vocabulary.index(ngram)+1) #+1 because 0 is reserved for padding or unknown
            else:
                onehot_tweet.append(0) #so that padding with 0 is meaningless later
        onehot_tweets.append(onehot_tweet)
    return onehot_tweets

def get_onehot_padded(onehots,padtosize):
    padded_ohs = []
    for onehot in onehots:
        padded_oh = []
        for i in onehot:
            if len(padded_oh) < padtosize:
                padded_oh.append(i)
        while len(padded_oh) < padtosize:
            padded_oh.append(0)
        padded_ohs.append(padded_oh)
    return padded_ohs

def get_onehots_given_vector(tweets_ngrams,vector,multiple = True):
    onehots = []
    for tweet_ngrams in tweets_ngrams:
        onehot = get_onehot_given_vector(tweet_ngrams,vector)
        onehots.append(onehot)
    return onehots

def get_onehot_given_vector(tweet_ngrams,vector,multiple = True):
    onehot = []
    for ngram in vector:
        if ngram in tweet_ngrams:
            if multiple:
                onehot.append(tweet_ngrams.count(ngram))
            else:
                onehot.append(1)
        else:
            onehot.append(0)
    return onehot


#train
unigrams_train = get_unigrams_for_tweets(tweets_train)
bigrams_train = get_ngrams_for_tweets(tweets_train,2)
trigrams_train = get_ngrams_for_tweets(tweets_train, 3)
fourgrams_train = get_ngrams_for_tweets(tweets_train, 4)

#test
unigrams_test = get_unigrams_for_tweets(tweets_test)
bigrams_test = get_ngrams_for_tweets(tweets_test,2)
trigrams_test = get_ngrams_for_tweets(tweets_test, 3)
fourgrams_test = get_ngrams_for_tweets(tweets_test, 4)

#extract vocab and dict data from train
unigram_dict, unigram_vocab = get_ngram_sorted_dictionary_vocab(unigrams_train)
bigram_dict, bigram_vocab = get_ngram_sorted_dictionary_vocab(bigrams_train)
trigram_dict, trigram_vocab = get_ngram_sorted_dictionary_vocab(trigrams_train)
fourgram_dict, fourgram_vocab = get_ngram_sorted_dictionary_vocab(fourgrams_train)

# ##s_as_ngrams(fourgrams_train_oh)

# ####################################################################################### APPROACH 2
#create an ngram representation vector with ngrams of interes -> then encode tweets wether they have given ngram 1 (or more if multiple appearances) or they dont have a given ngram 0
unigram_vector = unigram_vocab[0:300]
bigram_vector = bigram_vocab[0:300]
trigram_vector = trigram_vocab[0:300]
fourgram_vector = fourgram_vocab[0:300]

unigram_train_ohvector = get_onehots_given_vector(unigrams_train,unigram_vector)
unigram_test_ohvector = get_onehots_given_vector(unigrams_test,unigram_vector)

bigram_train_ohvector = get_onehots_given_vector(bigrams_train,bigram_vector)
bigram_test_ohvector = get_onehots_given_vector(bigrams_test,bigram_vector)

trigram_train_ohvector = get_onehots_given_vector(trigrams_train,trigram_vector)
trigram_test_ohvector = get_onehots_given_vector(trigrams_test,trigram_vector)

fourgram_train_ohvector = get_onehots_given_vector(fourgrams_train,fourgram_vector)
fourgram_test_ohvector = get_onehots_given_vector(fourgrams_test,fourgram_vector)


##################################################################### TAKING TARGETS INTO CONSIDERATION
unigrams_per_target_dictionary, unigrams_per_target_vocabulary = get_ngrams_for_targets(unigrams_train, unigram_vocab, targets_train)
unigrams_per_target_onehot_train = get_onehot_for_tweet_knowing_target_and_vector(unigrams_per_target_dictionary, unigrams_per_target_vocabulary,unigrams_train,targets_train)
unigrams_per_target_onehot_test = get_onehot_for_tweet_knowing_target_and_vector(unigrams_per_target_dictionary, unigrams_per_target_vocabulary,unigrams_test,targets_test)

bigrams_per_target_dictionary, bigrams_per_target_vocabulary = get_ngrams_for_targets(bigrams_train, bigram_vocab, targets_train)
bigrams_per_target_onehot_train = get_onehot_for_tweet_knowing_target_and_vector(bigrams_per_target_dictionary, bigrams_per_target_vocabulary,bigrams_train,targets_train)
bigrams_per_target_onehot_test = get_onehot_for_tweet_knowing_target_and_vector(bigrams_per_target_dictionary, bigrams_per_target_vocabulary,bigrams_test,targets_test)


trigrams_per_target_dictionary, trigrams_per_target_vocabulary = get_ngrams_for_targets(trigrams_train, trigram_vocab, targets_train)
trigrams_per_target_onehot_train = get_onehot_for_tweet_knowing_target_and_vector(trigrams_per_target_dictionary, trigrams_per_target_vocabulary,trigrams_train,targets_train)
trigrams_per_target_onehot_test = get_onehot_for_tweet_knowing_target_and_vector(trigrams_per_target_dictionary, trigrams_per_target_vocabulary,trigrams_test,targets_test)

fourgrams_per_target_dictionary, fourgrams_per_target_vocabulary = get_ngrams_for_targets(fourgrams_train, fourgram_vocab, targets_train)
fourgrams_per_target_onehot_train = get_onehot_for_tweet_knowing_target_and_vector(fourgrams_per_target_dictionary, fourgrams_per_target_vocabulary,fourgrams_train,targets_train)
fourgrams_per_target_onehot_test = get_onehot_for_tweet_knowing_target_and_vector(fourgrams_per_target_dictionary, fourgrams_per_target_vocabulary,fourgrams_test,targets_test)

########################################################################## ENCODE EACH WORD (in this context 1,2,3,4 gram as an onehot)

unigram_wordOhs_train = get_tweets_as_wordOhs(unigrams_train,unigram_vocab,size=500,padto=15)
unigram_wordOhs_test = get_tweets_as_wordOhs(unigrams_test,unigram_vocab,size=500,padto=15)

bigram_wordOhs_train = get_tweets_as_wordOhs(bigrams_train,bigram_vocab,size=500,padto=15)
bigram_wordOhs_test = get_tweets_as_wordOhs(bigrams_test,bigram_vocab,size=500,padto=15)

trigram_wordOhs_train = get_tweets_as_wordOhs(trigrams_train,trigram_vocab,size=500,padto=15)
trigram_wordOhs_test = get_tweets_as_wordOhs(trigrams_test,trigram_vocab,size=500,padto=15)

fourgram_wordOhs_train = get_tweets_as_wordOhs(fourgrams_train,fourgram_vocab,size=500,padto=15)
fourgram_wordOhs_test = get_tweets_as_wordOhs(fourgrams_test,fourgram_vocab,size=500,padto=15)


#***************************************************************************************************************************************************
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

# from model_builder import *
# print("______________________________NGRAMS NN_________________________________________")
# print("--------------------NGRAMS--------------------------------------------")
# print('---------NGRAMS targets not considered----------')
# build_train_test_models(unigram_wordOhs_train,unigram_wordOhs_test,"unigrams_wordOHs")
# build_train_test_models(bigram_wordOhs_train,bigram_wordOhs_test,"bigrams_wordOHs")
# build_train_test_models(trigram_wordOhs_train,trigram_wordOhs_test,"trigrams_wordOHs")
# build_train_test_models(fourgram_wordOhs_train,fourgram_wordOhs_test,"fourgrams_wordOHs")

# build_train_test_simpleDense_models(unigrams_tain_ohp,unigrams_tain_ohp,"unigrams ohp")
# build_train_test_simpleDense_models(bigrams_tain_ohp,bigrams_tain_ohp,"bigrams ohp")
# build_train_test_simpleDense_models(trigrams_tain_ohp,trigrams_tain_ohp,"trigrams ohp")
# build_train_test_simpleDense_models(fourgrams_tain_ohp,fourgrams_tain_ohp,"fourgrams ohp")

# build_train_test_simpleDense_models(unigram_train_ohvector,unigram_test_ohvector,"unigrams ohvec")
# build_train_test_simpleDense_models(bigram_train_ohvector,bigram_test_ohvector,"bigrams ohvec")
# build_train_test_simpleDense_models(trigram_train_ohvector,trigram_test_ohvector,"trigrams ohvec")
# build_train_test_simpleDense_models(fourgram_train_ohvector,fourgram_test_ohvector,"fourgrams ohvec")

# build_train_test_simpleDense_models(unigrams_per_target_onehot_train,unigrams_per_target_onehot_test,"unigrams oh_pertargetvect")
# build_train_test_simpleDense_models(bigrams_per_target_onehot_train,bigrams_per_target_onehot_test,"bigrams oh_pertargetvect")
# build_train_test_simpleDense_models(trigrams_per_target_onehot_train,trigrams_per_target_onehot_test,"trigrams oh_pertargetvect")
# build_train_test_simpleDense_models(fourgrams_per_target_onehot_train,fourgrams_per_target_onehot_test,"fourgrams oh_pertargetvect")

# print("--------------------NGRAMS Targets considered--------------------------------------------")
# build_train_test_models_targets(unigram_wordOhs_train,unigram_wordOhs_test,"unigrams_wordOHs TARGET")
# build_train_test_models_targets(bigram_wordOhs_train,bigram_wordOhs_test,"bigrams_wordOHs TARGET")
# build_train_test_models_targets(trigram_wordOhs_train,trigram_wordOhs_test,"trigrams_wordOHs TARGET")
# build_train_test_models_targets(fourgram_wordOhs_train,fourgram_wordOhs_test,"fourgrams_wordOHs TARGET")

# build_train_test_simpleDense_models_targets(unigrams_tain_ohp,unigrams_test_ohp,"unigrams ohp TARGET")
# build_train_test_simpleDense_models_targets(bigrams_tain_ohp,bigrams_test_ohp,"bigrams ohp TARGET")
# build_train_test_simpleDense_models_targets(trigrams_tain_ohp,trigrams_test_ohp,"trigrams ohp TARGET")
# build_train_test_simpleDense_models_targets(fourgrams_tain_ohp,fourgrams_test_ohp,"fourgrams ohp TARGET")

# build_train_test_simpleDense_models_targets(unigram_train_ohvector,unigram_test_ohvector,"unigrams ohvec TARGET")
# build_train_test_simpleDense_models_targets(bigram_train_ohvector,bigram_test_ohvector,"bigrams ohvec TARGET")
# build_train_test_simpleDense_models_targets(trigram_train_ohvector,trigram_test_ohvector,"trigrams ohvec TARGET")
# build_train_test_simpleDense_models_targets(fourgram_train_ohvector,fourgram_test_ohvector,"fourgrams ohvec TARGET")

# build_train_test_simpleDense_models_targets(unigrams_per_target_onehot_train,unigrams_per_target_onehot_test,"unigrams oh_pertargetvect TARGET")
# build_train_test_simpleDense_models_targets(bigrams_per_target_onehot_train,bigrams_per_target_onehot_test,"bigrams oh_pertargetvect TARGET")
# build_train_test_simpleDense_models_targets(trigrams_per_target_onehot_train,trigrams_per_target_onehot_test,"trigrams oh_pertargetvect TARGET")
# build_train_test_simpleDense_models_targets(fourgrams_per_target_onehot_train,fourgrams_per_target_onehot_test,"fourgrams oh_pertargetvect TARGET")

# from classifier_builder import *
# print("______________________________NGRAMS CLASSIFIERS_________________________________________")
# print("--------------------NGRAMS--------------------------------------------")
# print('---------NGRAMS targets not considered----------')

# build_and_evaluate_classifier(unigrams_tain_ohp,unigrams_test_ohp,name="unigrams ohp")
# build_and_evaluate_classifier(bigrams_tain_ohp,bigrams_test_ohp,name="bigrams ohp")
# build_and_evaluate_classifier(trigrams_tain_ohp,trigrams_test_ohp,name="trigrams ohp")
# build_and_evaluate_classifier(fourgrams_tain_ohp,fourgrams_test_ohp,name="fourgrams ohp")

# build_and_evaluate_classifier(unigram_train_ohvector,unigram_test_ohvector,name="unigrams ohvec")
# build_and_evaluate_classifier(bigram_train_ohvector,bigram_test_ohvector,name="bigrams ohvec")
# build_and_evaluate_classifier(trigram_train_ohvector,trigram_test_ohvector,name="trigrams ohvec")
# build_and_evaluate_classifier(fourgram_train_ohvector,fourgram_test_ohvector,name="fourgrams ohvec")

# build_and_evaluate_classifier(unigrams_per_target_onehot_train,unigrams_per_target_onehot_test,name="unigrams oh_pertargetvect")
# build_and_evaluate_classifier(bigrams_per_target_onehot_train,bigrams_per_target_onehot_test,name="bigrams oh_pertargetvect")
# build_and_evaluate_classifier(trigrams_per_target_onehot_train,trigrams_per_target_onehot_test,name="trigrams oh_pertargetvect")
# build_and_evaluate_classifier(fourgrams_per_target_onehot_train,fourgrams_per_target_onehot_test,name="fourgrams oh_pertargetvect")

# print("--------------------NGRAMS Targets considered--------------------------------------------")
# build_and_evaluate_classifiers_targets(unigrams_tain_ohp,unigrams_test_ohp,name="unigrams ohp TARGET")
# build_and_evaluate_classifiers_targets(bigrams_tain_ohp,bigrams_test_ohp,name="bigrams ohp TARGET")
# build_and_evaluate_classifiers_targets(trigrams_tain_ohp,trigrams_test_ohp,name="trigrams ohp TARGET")
# build_and_evaluate_classifiers_targets(fourgrams_tain_ohp,fourgrams_test_ohp,name="fourgrams ohp TARGET")

# build_and_evaluate_classifiers_targets(unigram_train_ohvector,unigram_test_ohvector,name="unigrams ohvec TARGET")
# build_and_evaluate_classifiers_targets(bigram_train_ohvector,bigram_test_ohvector,name="bigrams ohvec TARGET")
# build_and_evaluate_classifiers_targets(trigram_train_ohvector,trigram_test_ohvector,name="trigrams ohvec TARGET")
# build_and_evaluate_classifiers_targets(fourgram_train_ohvector,fourgram_test_ohvector,name="fourgrams ohvec TARGET")

# build_and_evaluate_classifiers_targets(unigrams_per_target_onehot_train,unigrams_per_target_onehot_test,name="unigrams oh_pertargetvect TARGET")
# build_and_evaluate_classifiers_targets(bigrams_per_target_onehot_train,bigrams_per_target_onehot_test,name="bigrams oh_pertargetvect TARGET")
# build_and_evaluate_classifiers_targets(trigrams_per_target_onehot_train,trigrams_per_target_onehot_test,name="trigrams oh_pertargetvect TARGET")
# build_and_evaluate_classifiers_targets(fourgrams_per_target_onehot_train,fourgrams_per_target_onehot_test,name="fourgrams oh_pertargetvect TARGET")