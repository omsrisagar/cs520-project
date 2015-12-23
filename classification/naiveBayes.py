# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
    #Viet added - Conditional Proabilities.
    self.P = util.Counter()
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    print "Begin define train and tune..."
    # print self.features        
    # print self.legalLabels

    c = util.Counter()
    k_res = util.Counter()
    for index in range(len(trainingData)):
      datum = trainingData[index];
      label = trainingLabels[index];
      for key in datum.sortedKeys():
        if(datum[key]==1):
          c[(1,key,label)] += 1
        # elif (datum[key]==2):
        #   c[(2,key,label)] += 1
        else: 
          c[(0,key,label)] += 1
    
    # Conditional Probabilities
    for k in kgrid:
      print "Set k = ", k
      for feature in self.features:
        for label in self.legalLabels:
          # S = c[(1, feature, label)] + k + c[(0, feature, label)] + k + c[(2, feature, label)] + k
          # self.P[(2, feature, label)] = (c[(2, feature, label)] + k) / (S * 1.0)
          # self.P[(1, feature, label)] = (c[(1, feature, label)] + k) / (S * 1.0)
          # self.P[(0, feature, label)] = (c[(0, feature, label)] + k) / (S * 1.0)
          S = c[(1, feature, label)] + k + c[(0, feature, label)] + k
          self.P[(1, feature, label)] = (c[(1, feature, label)] + k) / (S * 1.0)
          self.P[(0, feature, label)] = (c[(0, feature, label)] + k) / (S * 1.0)
      
      # calculate the accuracy 
      guesses = self.classify(validationData)
      correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
      k_res[k] = 100.0 * correct / len(validationLabels)
      print "Accuracy = ", k_res[k], "%"

    # Evalute and choose the otimum k 
    print k_res
    k = k_res.sortedKeys()[0]
    print "Set k = ", k 

    # Reassign conditional probabilities 
    for feature in self.features:
      for label in self.legalLabels:
        # S = c[(1, feature, label)] + k + c[(0, feature, label)] + k + c[(2, feature, label)] + k
        # self.P[(2, feature, label)] = (c[(2, feature, label)] + k) / S
        # self.P[(1, feature, label)] = (c[(1, feature, label)] + k) / S
        # self.P[(0, feature, label)] = (c[(0, feature, label)] + k) / S 
        S = c[(1, feature, label)] + k + c[(0, feature, label)] + k 
        self.P[(1, feature, label)] = (c[(1, feature, label)] + k) / (S * 1.0)
        self.P[(0, feature, label)] = (c[(0, feature, label)] + k) / (S * 1.0)

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    for feature in self.features:
      for label in self.legalLabels: 
        # if(datum[feature]==2):
        #   logJoint[label] += math.log(self.P[(2, feature, label)])
        # elif(datum[feature]==1):
        #   logJoint[label] += math.log(self.P[(1, feature, label)])
        # else:
        #   logJoint[label] += math.log(self.P[(0, feature, label)])   
        if(datum[feature]==1):
          logJoint[label] += math.log(self.P[(1, feature, label)])
        else:
          logJoint[label] += math.log(self.P[(0, feature, label)])   
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    temp = util.Counter()
    for feature in self.features:
      temp[feature] = self.P[(1,feature, label1)] / self.P[(1,feature, label2)]

    featuresOdds = temp.sortedKeys()[0:99]
    return featuresOdds
    

    
      


