from ngrams import *
from sentiment_lexicons import *

import zipfile

def concat_preprocess_lexicons(repr1_train,repr2_train,repr1_test,repr2_test,dim_1D = True):
    concat_train = []
    concat_test = []
    concat_summary_train = []
    concat_summary_test = []

    for (r1,r2) in zip(repr1_train,repr2_train):
        concat = []
        if dim_1D:
            concat.extend(r1)
            concat.extend(r2)
        else:
            for (v1,v2) in zip(r1,r2):
                cnct = []
                cnct.extend(v1)
                cnct.extend(v2)
                concat.append(cnct)
        concat_train.append(concat)
        concat_summary_train.append(np.average(concat,axis=0))
    for (r1,r2) in zip(repr1_test,repr2_test):
        concat = []
        if dim_1D:
            concat.extend(r1)
            concat.extend(r2)
        else:
            for (v1,v2) in zip(r1,r2):
                cnct = []
                cnct.extend(v1)
                cnct.extend(v2)
                concat.append(cnct)
        concat_test.append(concat)
        concat_summary_test.append(np.average(concat,axis=0))

    return (concat_train,concat_test,concat_summary_train,concat_summary_test)

def concat_preprocess_lexiconAFINN(repr1_train,repr2_train,repr1_test,repr2_test):
    concat_train = []
    concat_test = []
    concat_summary_train = []
    concat_summary_test = []

    for (r1,r2) in zip(repr1_train,repr2_train):
        concat = []
        concat.extend(r1)
        concat.append(r2)
        concat_train.append(concat)
        concat_summary_train.append(np.average(concat,axis=0))
    for (r1,r2) in zip(repr1_test,repr2_test):
        concat = []
        concat.extend(r1)
        concat.append(r2)
        concat_test.append(concat)
        concat_summary_test.append(np.average(concat,axis=0))

    return (concat_train,concat_test,concat_summary_train,concat_summary_test)

def concat_preprocess_lexicons_Empath_VADER(repr1_train,repr2_train,repr1_test,repr2_test):
    concat_train = []
    concat_test = []
    concat_summary_train = []
    concat_summary_test = []

    for (r1,r2) in zip(repr1_train,repr2_train):
        concat = []
        concat.extend(r1)
        concat.extend(r2)
        concat_train.append(concat)
        concat_summary_train.append(np.average(concat,axis=0))
    for (r1,r2) in zip(repr1_test,repr2_test):
        concat = []
        concat.extend(r1)
        concat.extend(r2)
        concat_test.append(concat)
        concat_summary_test.append(np.average(concat,axis=0))

    return (concat_train,concat_test,concat_summary_train,concat_summary_test)