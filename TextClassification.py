"""
Written By: Amr Koura
In this script, I will solve the text classification
"""
import nltk
import random
import numpy as np
import pickle
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf 
    
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



user_reviews=[]
user_ratings=[]
with open ('ratings.txt', 'rb') as fp:
    user_ratings = np.array(pickle.load(fp))
    
with open ('reviews.txt', 'rb') as fp:
    user_reviews = np.array(pickle.load(fp))
        
documents=[]
all_words = []
for i in range(len(user_reviews)):
    label='neg'
    if user_ratings[i]>3:
        label='pos'
    documents.append((user_reviews[i].lower().split(),label))
    for word in user_reviews[i].split():
        all_words.append(word.lower())


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]            

featuresets = [(find_features(rev), category) for (rev, category) in documents]

     
# set that we'll train our classifier with
#take 90 % for development and 10 % as validation
# apply 5 cross validation.

val_sample_index = int(0.2 * float(len(featuresets)))
NB_accuracy=[]
MNB_accuracy=[]

BernoulliNB_accuracy=[]
LogisticRegression_accuracy=[]
SGD_accuracy=[]
SVC_accuracy=[]
LinearSVC_accuracy=[]
NuSVC_accuracy=[]
voted_accuracy=[]
for i in range(5):
    training_set=featuresets
    testing_set = featuresets[i*val_sample_index:i*val_sample_index+val_sample_index]
    #take all but not the indeicies in the vlaidation set
    training_set=featuresets[:i*val_sample_index]+featuresets[i*val_sample_index+val_sample_index:]
    
    # set that we'll test against.
    print("length of training_set=",len(training_set))
    print("length of testing_set=",len(testing_set))
    
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    acc=nltk.classify.accuracy(classifier, testing_set)
    NB_accuracy.append(acc)
    print((i+1),"NaiveBayes Classifier accuracy percent:",(acc)*100)
    
    from nltk.classify.scikitlearn import SklearnClassifier
    
    from sklearn.naive_bayes import MultinomialNB,BernoulliNB
    
        
    from sklearn.linear_model import LogisticRegression,SGDClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC
    
    #classifier.show_most_informative_features(15)
    
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    acc=nltk.classify.accuracy(MNB_classifier, testing_set)
    MNB_accuracy.append(acc)
    print((i+1),"MNB_classifier accuracy percent:", (acc)*100)
    
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    acc=nltk.classify.accuracy(BernoulliNB_classifier, testing_set)
    BernoulliNB_accuracy.append(acc)
    print((i+1),"BernoulliNB_classifier accuracy percent:", (acc)*100)
    
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    acc=nltk.classify.accuracy(LogisticRegression_classifier, testing_set)
    LogisticRegression_accuracy.append(acc)
    print((i+1),"LogisticRegression_classifier accuracy percent:", (acc)*100)
    
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    acc=nltk.classify.accuracy(SGDClassifier_classifier, testing_set)
    SGD_accuracy.append(acc)
    print((i+1),"SGDClassifier_classifier accuracy percent:", (acc)*100)
    
    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    acc=nltk.classify.accuracy(SVC_classifier, testing_set)
    SVC_accuracy.append(acc)
    print((i+1),"SVC_classifier accuracy percent:", (acc)*100)
    
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    acc=nltk.classify.accuracy(LinearSVC_classifier, testing_set)
    LinearSVC_accuracy.append(acc)
    print((i+1),"LinearSVC_classifier accuracy percent:", (acc)*100)
    
    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    acc=nltk.classify.accuracy(NuSVC_classifier, testing_set)
    NuSVC_accuracy.append(acc)
    print((i+1),"NuSVC_classifier accuracy percent:", (acc)*100)
    
    voted_classifier = VoteClassifier(classifier,
                                      NuSVC_classifier,
                                      LinearSVC_classifier,
                                      SGDClassifier_classifier,
                                      MNB_classifier,
                                      BernoulliNB_classifier,
                                      LogisticRegression_classifier)
    acc=nltk.classify.accuracy(voted_classifier, testing_set)
    voted_accuracy.append(acc)
    print((i+1),"voted_classifier accuracy percent:", (acc)*100)
    print ("---------------------------------------------------")

print("result from 5-cross validation!!")


print("Naive Bayes  accuracy percent:",np.mean(NB_accuracy)*100)
print("MultinomialNB  accuracy percent:",np.mean(MNB_accuracy)*100)
print("BernoulliNB  accuracy percent:",np.mean(BernoulliNB_accuracy)*100)

print("Logistic regression  accuracy percent:",np.mean(LogisticRegression_accuracy)*100)
print("SGD  accuracy percent:",np.mean(SGD_accuracy)*100)
print("SVC   accuracy percent:",np.mean(SVC_accuracy)*100)
print("LinearSVC  accuracy percent:",np.mean(LinearSVC_accuracy)*100)
print("NuSVC  accuracy percent:",np.mean(NuSVC_accuracy)*100)
print("voted  accuracy percent:",np.mean(voted_accuracy)*100)


               