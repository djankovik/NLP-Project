from processData import *
import numpy as np
from nltk.corpus import stopwords
from utils import get_tweets_as_wordOhs

def load_embeddings(file_name, vocabulary):
    embeddings = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            parts = line.split(' ')
            vals = np.array(parts[1:], dtype=np.float)
            if parts[0] in vocabulary:
                embeddings[parts[0]] = vals
            line = doc.readline()
    return embeddings

def load_embedding_weights(vocabulary, glovepath="",savepath="",embedding_size=50):
    if os.path.exists(savepath):
        with open(savepath, 'rb') as f:
            embedding_matrix = pickle.load(f)
    else:
        #print('Creating embedding weights...')
        embeddings = load_embeddings(glovepath,vocabulary)
        embedding_matrix = np.zeros((len(vocabulary), embedding_size))
        for i in range(len(vocabulary)):
            if vocabulary[i] in embeddings.keys():
                embedding_matrix[i] = embeddings[vocabulary[i]]
            else:
                embedding_matrix[i] = np.random.standard_normal(embedding_size)
        with open(savepath, 'wb') as f:
            pickle.dump(embedding_matrix, f)
    return embedding_matrix

# embedding_matrix = load_embedding_weights(tweets_vocabulary,glovepath='data\glove.6B.50d.txt',savepath="data\embedding_matrix.pkl")
# embedding_matrix_twitter = load_embedding_weights(tweets_vocabulary,glovepath='data\glove.twitter.27B.50d.txt',savepath="data\embedding_matrix_twt.pkl")

# from processData import *
# from model_builder import *

# print("_____________________EMBEDDINGS NN_____________________________")
# print("--------------------EMBEDDINGS-------------------------------------")
# print("--------------------EMBEDDINGS targets not considered-------------------------------------")
# build_train_test_simpleDense_models(tweets_intarray_padded_train,tweets_intarray_padded_test,"tweets_intarrays")
# build_train_test_models(tweets_wordOhs_train,tweets_wordOhs_test,"tweet_wordOhs")
# build_train_test_embeddings_models(tweets_intarray_padded_train,tweets_intarray_padded_test,embedding_matrix,len(tweets_vocabulary),"tweets_embeddings")
# build_train_test_embeddings_models(tweets_intarray_padded_train,tweets_intarray_padded_test,embedding_matrix_twitter,len(tweets_vocabulary),"tweets_twitterembeddings")

# print("--------------------EMBEDDINGS targets considered-------------------------------------")
# build_train_test_simpleDense_models_targets(tweets_intarray_padded_train,tweets_intarray_padded_test,"tweets_intarrays TARGETS")
# build_train_test_models_targets(tweets_wordOhs_train,tweets_wordOhs_test,"tweet_wordOhs TARGETS")
# build_train_test_models_embeddings_targets(tweets_intarray_padded_train,tweets_intarray_padded_test,embedding_matrix,len(tweets_vocabulary),"tweets_embeddings TARGETS")
# build_train_test_models_embeddings_targets(tweets_intarray_padded_train,tweets_intarray_padded_test,embedding_matrix_twitter,len(tweets_vocabulary),"tweets_twitterembeddings TARGETS")

# from classifier_builder import *
# print("_____________________EMBEDDINGS CLASSIFIERS_____________________________")
# print("--------------------EMBEDDINGS-------------------------------------")
# print("--------------------EMBEDDINGS targets not considered-------------------------------------")
# build_and_evaluate_classifier(tweets_intarray_padded_train,tweets_intarray_padded_test,name="tweets_intarrays")

# print("--------------------EMBEDDINGS targets considered-------------------------------------")
# build_and_evaluate_classifiers_targets(tweets_intarray_padded_train,tweets_intarray_padded_test,name="tweets_intarrays TARGETS")
