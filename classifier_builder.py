from sklearn import svm
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn import metrics
import numpy as np
from sklearn.base import clone
from  processData import *
from  utils import *

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
    print("---------------"+name+"-----------------------")
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

def run_classifier_with_target_knowledge(train_in,test_in,classifier,train_out=stances_asints_train,test_out=stances_asints_test,targets_in = targets_train, targets_out = targets_test,name=""):
    if name == "":
        name = type(classifier).__name__
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
    
    y_pred = predictions
    print(' '+name)
    print("\taccuracy:",metrics.accuracy_score(test_out, y_pred))
    print("\tpecision | pweighted: "+str(metrics.precision_score(test_out, y_pred,average="weighted"))+", pmicro: "+str(metrics.precision_score(test_out, y_pred,average="micro"))+", pmacro: "+str(metrics.precision_score(test_out, y_pred,average="macro")))
    print("\trecall | rweighted: "+str(metrics.recall_score(test_out, y_pred,average="weighted"))+", rmicro: "+str(metrics.recall_score(test_out, y_pred,average="micro"))+", rmacro: "+str(metrics.recall_score(test_out, y_pred,average="macro")))
    print("\tf1 | f1weighted: "+str(metrics.f1_score(test_out, y_pred,average="weighted"))+", f1micro: "+str(metrics.f1_score(test_out, y_pred,average="micro"))+", f1macro: "+str(metrics.f1_score(test_out, y_pred,average="macro")))
    evaluate_classifier(y_pred,test_out,name)

    classifier_name = type(classifier).__name__
    evaluate_classifier(predictions,test_out,name=classifier_name)

def build_and_evaluate_classifiers_targets(train_in,test_in,train_out=stances_asints_train,test_out=stances_asints_test,name=""):
    print("--------------------"+name+"---------------------")
    run_classifier_with_target_knowledge(train_in, test_in,BernoulliNB())
    run_classifier_with_target_knowledge(train_in, test_in,GaussianNB())
    run_classifier_with_target_knowledge(train_in, test_in,DecisionTreeClassifier())   
    run_classifier_with_target_knowledge(train_in, test_in,ExtraTreeClassifier(random_state=42))
    run_classifier_with_target_knowledge(train_in, test_in,NearestCentroid())
    run_classifier_with_target_knowledge(train_in, test_in,svm.SVC(kernel='rbf',decision_function_shape='ovo',degree=5,gamma='auto'),name="SVM ovo")
    run_classifier_with_target_knowledge(train_in, test_in,svm.SVC(kernel='rbf',decision_function_shape='ovr',degree=5,gamma='auto'),name="SVM ovr")
    run_classifier_with_target_knowledge(train_in, test_in,svm.LinearSVC(max_iter=10000),name="LinearSVM")
    run_classifier_with_target_knowledge(train_in, test_in,RandomForestClassifier(random_state=42))