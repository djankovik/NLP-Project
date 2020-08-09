from utils import *
from readDataFromFile import *
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from ann_visualizer.visualize import ann_viz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def simpleDense_model(train_in,test_in,train_out,test_out,inputshape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(inputshape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\simpleDense.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="simpleDense")
    return evaluated

def simpleDense_model_targets(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test,inputshape=10):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(inputshape,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam') 
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="simpleDense_targets")
    return evaluated

##########################################################################################################################
def simpleRNN_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len((train_in.tolist())[0])))
    model.add(SimpleRNN(units=50))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\simpleRNN_embeddings.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')    
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="simpleRNN")
    return evaluated

def simpleRNN_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(SimpleRNN(units=50, activation="relu"))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\simpleRNN.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="simpleRNN")
    return evaluated

def simpleRNN_model_targets(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(SimpleRNN(units=50, activation="relu"))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="simpleRNN_targets")
    return evaluated

def simpleRNN_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()
    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len(train_in[0])))
        model.add(SimpleRNN(units=50, activation="relu"))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="simpleRNN+embeddings TARGETS")
    return evaluated
##########################################################################################################################
def simpleGRU_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len((train_in.tolist())[0])))
    model.add(GRU(256, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\GRU_embeddings.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="gru")
    return evaluated

def simpleGRU_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(GRU(256, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\GRU.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="gru")
    return evaluated

def simpleGRU_model_targets(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(GRU(256, dropout_W=0.2, dropout_U=0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="simpleGRU_targets")
    return evaluated

def simpleGRU_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len(train_in[0])))
        model.add(GRU(256, dropout_W=0.2, dropout_U=0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="simpleGRU+embeddings TARGETS")
    return evaluated

##########################################################################################################################
def Conv1D_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len((train_in.tolist())[0])))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\conv1D_embeddings.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="conv1D")
    return evaluated

def Conv1D_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\conv1D.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="conv1D")
    return evaluated

def Conv1D_model_targets(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="Conv1D_targets")
    return evaluated

def Conv1D_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len(train_in[0])))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="Conv1D+embeddings TARGETS")
    return evaluated
##########################################################################################################################
def multiLSTM_TimeDistributed_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len((train_in.tolist())[0])))
    model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(TimeDistributed(Dense(32, activation='relu'))) 
    model.add(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
    model.add(TimeDistributed(Dense(128, activation='relu'))) 
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\LSTM_TD_LSTM_TD_LSTM_embeddings.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="lstm_timedist_lst..")
    return evaluated

def multiLSTM_TimeDistributed_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(TimeDistributed(Dense(32, activation='relu'))) 
    model.add(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
    model.add(TimeDistributed(Dense(128, activation='relu'))) 
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\LSTM_TD_LSTM_TD_LSTM_embeddings.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="lstm_timedist_lst..")
    return evaluated

def multiLSTM_TimeDistributed_model_targets(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
        model.add(TimeDistributed(Dense(32, activation='relu'))) 
        model.add(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
        model.add(TimeDistributed(Dense(128, activation='relu'))) 
        model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) 
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="multiLSTM_TimeDistributed_targets")
    return evaluated

def multiLSTM_TimeDistributed_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len(train_in[0])))
        model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
        model.add(TimeDistributed(Dense(32, activation='relu'))) 
        model.add(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
        model.add(TimeDistributed(Dense(128, activation='relu'))) 
        model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) 
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="multiLSTM_TimeDistributed+embeddings TARGETS")
    return evaluated

##########################################################################################################################
def simpleLSTM_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len((train_in.tolist())[0])))
    model.add(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\simpleLSTM_embeddings.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="simpleLSTM")
    return evaluated

def simpleLSTM_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\simpleLSTM.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="simpleLSTM")
    return evaluated

def simpleLSTM_model_targets(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="simpleLSTM_targets")
    return evaluated

def simpleLSTM_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len(train_in[0])))
        model.add(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="simpleLSTM+embeddings TARGETS")
    return evaluated
##########################################################################################################################
def BILSTM_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len((train_in.tolist())[0])))
    model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\BiLSTM_embeddings.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="bi->lstm")
    return evaluated

def BILSTM_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    plot_model(model, to_file='modelplots\BiLSTM.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),detailedreport=True,singleLine=True, modelName="bi->lstm")
    return evaluated

def BILSTM_model_targets(train_in,test_in,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="BILSTM_targets")
    return evaluated

def BILSTM_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out=stances_onehot_train,test_out=stances_onehot_test):
    train_in_per_targets = get_things_separated_by_targets(train_in,targets_train)
    train_out_per_targets = get_things_separated_by_targets(train_out,targets_train)
    models_per_targets = dict()

    for target in train_in_per_targets.keys():
        train_target_in = make_np_arrays(train_in_per_targets[target])
        train_target_out = np.array(train_out_per_targets[target])
        model = Sequential()
        model.add(Embedding(input_dim=vocabulary_length,
                    output_dim=50,
                    weights=[embedding_matrix],
                    input_length=len(train_in[0])))
        model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(train_target_in, train_target_out, batch_size=32, epochs=15, verbose=0) 
        models_per_targets[target] = model
    
    models_predictions = []

    for (tst_in,trg_in) in zip(test_in,targets_test):
        model = models_per_targets[trg_in]
        y_pred = model.predict(make_np_arrays([tst_in]))
        models_predictions.append(y_pred[0])
        
    evaluated = evaluate_model_predictions(models_predictions,test_out,detailedreport=True,singleLine=True, modelName="BILSTM+embeddings TARGETS")
    return evaluated

##########################################################################################################################
def build_train_test_simpleDense_models(train_in_raw,test_in_raw,name=""):
    train_out = np.array(stances_onehot_train)
    test_out = np.array(stances_onehot_test)
    train_in = make_np_arrays(train_in_raw)
    test_in = make_np_arrays(test_in_raw)

    print('______'+name+'______')
    simpleDense_model(train_in,test_in,train_out,test_out,inputshape=len(train_in_raw[0]))

def build_train_test_simpleDense_models_targets(train_in_raw,test_in_raw,name=""):
    print('______'+name+'______')
    train_in = train_in_raw
    test_in = test_in_raw
    train_out = stances_onehot_train
    test_out = stances_onehot_test
    simpleDense_model_targets(train_in_raw,test_in_raw,train_out,test_out,inputshape=len(train_in_raw[0]))

def build_train_test_models(train_in_raw,test_in_raw,name=""):
    train_out = np.array(stances_onehot_train)
    test_out = np.array(stances_onehot_test)
    train_in = make_np_arrays(train_in_raw)
    test_in = make_np_arrays(test_in_raw)

    print('______'+name+'______')
    simpleRNN_model(train_in,test_in,train_out,test_out)
    simpleGRU_model(train_in,test_in,train_out,test_out)
    Conv1D_model(train_in,test_in,train_out,test_out)
    multiLSTM_TimeDistributed_model(train_in,test_in,train_out,test_out)
    simpleLSTM_model(train_in,test_in,train_out,test_out)
    BILSTM_model(train_in,test_in,train_out,test_out)

def build_train_test_models_targets(train_in_raw,test_in_raw,name=""):
    print('______'+name+'______')
    train_in = train_in_raw
    test_in = test_in_raw
    train_out = stances_onehot_train
    test_out = stances_onehot_test
    simpleRNN_model_targets(train_in,test_in,train_out,test_out)
    simpleGRU_model_targets(train_in,test_in,train_out,test_out)
    Conv1D_model_targets(train_in,test_in,train_out,test_out)
    multiLSTM_TimeDistributed_model_targets(train_in,test_in,train_out,test_out)
    simpleLSTM_model_targets(train_in,test_in,train_out,test_out)
    BILSTM_model_targets(train_in,test_in,train_out,test_out)

def build_train_test_embeddings_models(train_in_raw,test_in_raw,embedding_matrix,vocabulary_length,name=""):
    train_out = np.array(stances_onehot_train)
    test_out = np.array(stances_onehot_test)
    train_in = make_np_arrays(train_in_raw)
    test_in = make_np_arrays(test_in_raw)

    print('______'+name+'______')
    simpleRNN_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length)
    simpleGRU_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length)
    Conv1D_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length)
    multiLSTM_TimeDistributed_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length)
    simpleLSTM_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length)
    BILSTM_model_embeddings(train_in,test_in,train_out,test_out,embedding_matrix,vocabulary_length)

def build_train_test_models_embeddings_targets(train_in_raw,test_in_raw,embedding_matrix,vocabulary_length,name=""):
    print('______'+name+'______')
    train_in = train_in_raw
    test_in = test_in_raw
    train_out = stances_onehot_train
    test_out = stances_onehot_test
    print(str(len(train_in[0])))
    simpleRNN_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out,test_out)
    simpleGRU_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out,test_out)
    Conv1D_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out,test_out)
    multiLSTM_TimeDistributed_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out,test_out)
    simpleLSTM_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out,test_out)
    BILSTM_model_embeddings_targets(train_in,test_in,embedding_matrix,vocabulary_length,train_out,test_out)