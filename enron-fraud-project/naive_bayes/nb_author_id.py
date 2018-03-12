#!/usr/bin/python

    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    clf = GaussianNB()
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf.predict(features_test) 
    print "prediction time:", round(time()-t0, 3), "s"
    accuracy =  accuracy_score(pred,labels_test)
    return clf, accuracy

print(classify(features_train,labels_train))

#########################################################


