# -*- mode: Python; coding: utf-8 -*-
from __future__ import unicode_literals
from classifier import Classifier
import numpy as np
from nltk import *
import operator, math, copy
from corpus import BlogsCorpus, Document, NamesCorpus
#all the information about a particular name from the file, including feature value

class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        """From NLTK's names_demo_features: first & last letters, how many
        of each letter, and which letters appear."""
        name = self.data
        return ([name[0].lower(), name[-1].lower()] +
                [name.lower().count(letter) for letter in letters] +
                [letter in name.lower() for letter in letters])
                
class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""
    def __init__(self, model=[{}, FreqDist()]):
        super(Classifier, self).__init__()

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)
    
    #get frequency counts for the features
    def train(self, instances):
        """Remember the labels associated with the features of instances."""
        vocab = set()
        [vocab.add(word) for n in instances for word in n.features()]
        
        #count number of occurrences per clas label
        labels = FreqDist() #freqdist for class label occurrences
        for n in instances:
            labels[n.label] += 1
        
        #populate dictionary with freqdists for each label
        vocab = FreqDist(list(vocab))
        for word in vocab.keys():
            vocab[word] = 0
        featurefreqs = defaultdict()
        for label in labels:
            featurefreqs[label]= copy.deepcopy(vocab)     
       
       #count occurrences of each word per label
        for n in instances:
            for word in n.features():
                featurefreqs[n.label][word] += 1

        self.model = [featurefreqs, labels]

    def classify(self, instance):
        """Classify an instance using the features seen during training."""
        logprob = {}
        total = sum(self.model[1].values()) #total instances
        for label in self.model[1].keys():
            logprob[label] = math.log(self.model[1][label] / float(total)) #prob of label
       
            #get logprob of given feature vector
            vocab = set(self.model[0][label].keys())
            index = 0
            for word in instance.features(): #if word never seen before or freq was 0 in training, give smoothed count
                if word not in vocab or self.model[0][label][word] is 0 :
                    logprob[label] += math.log(0.5/(float(self.model[1][label] + len(vocab) +1)))
                else:
                    logprob[label] += math.log((self.model[0][label][word])/(float(self.model[1][label])))
                index+=1
        
        return max(logprob.iteritems(), key=operator.itemgetter(1))[0]
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
