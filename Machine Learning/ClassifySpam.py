#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:21:52 2020

@author: jasonmaloney
"""

'''
  This program shell reads email data for the spam classification problem.
  The input to the program is the path to the Email directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySPAM.py  <corpus directory path> <limit number>
'''

# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
import re

# define a stopword remover function
def stopword_remover(document):
    nltkstopwords = stopwords.words('english')
    addl_stopwords = ['subject', '']
    all_stopwords = nltkstopwords + addl_stopwords
    cleaned_list = []
    for word in document:
        if word not in all_stopwords:
            cleaned_list.append(word)
    return cleaned_list  

# define a feature definition function here
'MOST COMMON WORD FEATURE FUNCTION'
# feature labels will be V_keyword()
# value of the feature is Boolean (True, False) - if word is in the document
def most_common_feature(document, common_word_features):
    document_words = set(document) # get the unique words
    features = {} # define an empty dictionary for features
    # put each word in the dictionary
    for word in common_word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features

'BIGRAM FEATURE FUNCTION'
# feature labels will be B_bigram()
# value of the feature is Boolean (True, False) - if bigram is in the document
def bigram_document_features(document, common_word_features, bigram_features):
    #document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    #for word in common_word_features:
     #   features['V_{}'.format(word)] = (word in document_words)
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
    return features

'PUNCTUATION COUNT FUNCTION'  
# get the average amount of punctuation in ham
def punctuation_count(email):
  punctuation = []
  for word in email:
      punct = re.sub('[\w\s]','', word)
      punctuation.append(punct)
 # subtract 1 because each email will have '' instead of words/spaces - that is not punctuation
  return(len(set(punctuation))) 

'PUNCTUATION COUNT FEATURE FUNCTION'
# feature labels will be PC_count()
# value of the feature is Boolean (True, False) - 
# if count is greater than or equal to average spam punctuation count
def punct_count_feature(email, punct_count_list, avg_spam_punct):
  doc_count = punctuation_count(email)
  features = {}
  #features['PC_{}'.format(doc_count)] = (doc_count >= avg_ham_punct)
  features['PC_{}'.format(doc_count)] = (doc_count >= avg_spam_punct)
  return features

'LEXICAL SCORE FUNCTION'
# function takes a list as the argument
# counts the number of unique words and finds ratio of unique to total words
def lexical_score(email):
    email_length = len(email)
    unique_words = len(set(email))
    return(unique_words/email_length)

'LEXICAL SCORES - ratio of unique words to total words'
# feature labels will be LS_score()
# value of the feature is Boolean (True, False) - 
# if count is greater than or equal to average ham lexical score
def lex_score_feature(email, lex_score_list, avg_ham_lex):
    email_score = lexical_score(email)
    features = {}
    features['LS_{}'.format(email_score)] = (email_score >= avg_ham_lex)
    #features['LS_{}'.format(email_score)] = (email_score >= avg_spam_lex)
    return features

# define eval_measures to get recall, precision, and F1 scores
def eval_measures(gold, predicted, labels):
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []

    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
       
        # debugging pring statements
        #print(lab, "TP = ", TP)
        #print(lab, "FN = ", FN)
        #print(lab, "FP = ", FP)
        #print(lab, "TN = ", TN)
        
        # use these to compute recall, precision, F1
        # for small numbers, guard against dividing by zero in computing measures
        # original if statement below
        #if (TP == 0) or (FP == 0) or (FN == 0):
        # added additional conditions to if statement to ensure neither denominator can be 0
        if (TP + FP == 0) or (TP + FN == 0):
        #if ((TP == 0) and (FP == 0)) or ((TP == 0) and (FN == 0)): 
          recall_list.append(0)
          precision_list.append(0)
          F1_list.append(0)
        else:
          recall = TP / (TP + FP)
          precision = TP / (TP + FN)
          recall_list.append(recall)
          precision_list.append(precision)
          F1_list.append( 2 * (recall * precision) / (recall + precision))
    # the evaluation measures in a table with one row per label
    return (precision_list, recall_list, F1_list)


# define a cross validation function - takes number of folds, featuresets to use, and labels
def cross_validation_PRF(num_folds, featuresets, labels):
    subset_size = int(len(featuresets)/num_folds)
    print("Each fold size:", subset_size)
    print("Test Percentage: ", round(subset_size/len(featuresets),2))
    print("Train Percentage: ", round((1 - subset_size/len(featuresets)),2), "\n")
    # for the number of labels - start the totals lists with zeroes
    num_labels = len(labels)
    total_precision_list = [0] * num_labels
    total_recall_list = [0] * num_labels
    total_F1_list = [0] * num_labels 
    
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i * subset_size): ][ :subset_size]
        train_this_round = featuresets[ :(i * subset_size)] + featuresets[((i + 1) * subset_size): ]

        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        
        # evaluate agains the test_this_round to produce 
        # the gold and predicted labels
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))
            
        # computes evaluation measures for this fold and returns list 
        # of measures for each label 
        print("Fold", i, " - Accuracy:", round(accuracy_this_round, 3)* 100, "%\n")
        (precision_list, recall_list, F1_list) = eval_measures(goldlist, predictedlist, labels)

        # confusion matrix for each fold
        cm = nltk.ConfusionMatrix(goldlist, predictedlist)

        # confusion matrix with counts
        #print(cm.pretty_format(sort_by_count = True, show_percents = False, truncate = 9))
        # confusion matrix with percents
        print(cm.pretty_format(sort_by_count = True, show_percents = True, truncate = 9))

            # take off the triple string to print precision, recall, and F1 for each fold
            print('\tPrecision\tRecall\t\tF1')
            # print measures for each label
            for i, lab in enumerate(labels):
                print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
                      "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
        
        # for each label add to the sums in the total list
        for i in range(num_labels):
            # for each label, add the 3 measures to the 3 lists of totals
            total_precision_list[i] += precision_list[i]
            total_recall_list[i] += recall_list[i]
            total_F1_list[i] += F1_list[i]
            
            # debugging print statements
            #print("P - ", total_precision_list[i])
            #print("R - ", total_recall_list[i])
            #print("F - ", total_F1_list[i])
                     
    # find precision, recall, and F measure averaged over all rounds for all labels
    #compute the average from the total lists
    precision_list = [tot/num_folds for tot in total_precision_list]
    recall_list = [tot/num_folds for tot in total_recall_list]
    F1_list = [tot/num_folds for tot in total_F1_list]
      
    # debugging print statements
    #print("Total precision - ", total_precision_list, " - \n", "precision list - ", precision_list)
    #print("Total recall - ", total_recall_list, " - \n", "recall list - ", recall_list)
    #print("Total F1 - ", total_F1_list, " - \n", "F1 list - ", F1_list)

        # print evaluation measures in a table with one row per label
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
        
        # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
              "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
        
        # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list)/num_labels), \
          "{:10.3f}".format(sum(recall_list)/num_labels), \
          "{:10.3f}".format(sum(F1_list)/num_labels))
        
    # for micro averaging, weight the scores for each label by the number of items
    # this is a better result because we do not have balanced data - more ham than spam
    # initialize a dictionary for label counts and then count them
    label_counts = {}
    for lab in labels:
        label_counts[lab] = 0
            
            # count the labels
    for (doc, lab) in featuresets:
        label_counts[lab] += 1

            # make weights compared to the number of documents in featuresets
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab]/num_docs) for lab in labels]
    print('\nLabel Counts', label_counts)
            
            # print micro average over all labels
    print('\nMicro Average Precision\tRecall\t\tF1 \tOver All Labels')
            # zip combines corresponding elts from a list into a tuple            
    precision = sum([a * b for a, b in zip(precision_list, label_weights)])
    recall = sum([a * b for a, b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a, b in zip(F1_list, label_weights)])
    print('\t', "{:10.3f}".format(precision), \
          "{:10.3f}".format(recall), "{:10.3f}".format(F1))

# function to read spam and ham files, train and test a classifier 
def processspamham(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  # start lists for spam and ham email texts
  hamtexts = []
  spamtexts = []
  os.chdir(dirPath)
  # process all files in directory that end in .txt up to the limit
  #    assuming that the emails are sufficiently randomized
  for file in os.listdir("./spam"):
    if (file.endswith(".txt")) and (len(spamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./spam/"+file, 'r', encoding="latin-1")
      spamtexts.append (f.read())
      f.close()
  for file in os.listdir("./ham"):
    if (file.endswith(".txt")) and (len(hamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./ham/"+file, 'r', encoding="latin-1")
      hamtexts.append (f.read())
      f.close()
  
  # print number emails read
  print ("Number of spam files:",len(spamtexts))
  print ("Number of ham files:",len(hamtexts))
    
  # create list of mixed spam and ham email documents as (list of words, label)
  emaildocs = []
  hamdocs = []
  spamdocs = []
  
  # add all the spam
  for spam in spamtexts:
    tokens = nltk.word_tokenize(spam)
    # filter stop words
    #word = stopword_remover(tokens)
    emaildocs.append((tokens, 'spam'))
    #emaildocs.append((tokens, 'spam'))
    spamdocs.append(tokens)
  
  # add all the regular emails
  for ham in hamtexts:
    tokens = nltk.word_tokenize(ham)
    # filter stop words
    #word = stopword_remover(word)
    #emaildocs.append((tokens, 'ham'))
    emaildocs.append((tokens, 'ham'))
    hamdocs.append(tokens)
  
  # randomize the list
  random.shuffle(emaildocs)

  # create a list that has the labels  
  lab = [lab for email, lab in emaildocs]
  labels = list(set(lab))

  #lab = []
  #for email, lab in emaildocs:
   #   lab.append(lab)
  #labels = list(set(lab))
  
  'MOST COMMON WORD FEATURE'
  all_words_list = []
  for email, cat in emaildocs:
      for words in email:
          all_words_list.append(words)
  word_counts = nltk.FreqDist(all_words_list)

  # get the 10% most common words/symbols
  upper_freq = int(0.1 * len(word_counts))
  most_common_words = word_counts.most_common(upper_freq)

  # define the features
  common_word_features = []
  for (word, freq) in most_common_words:
      common_word_features.append(word)
  
  'BIGRAM PMI FEAURE'
  # define the bigram function from nltk
  bigram_measures = nltk.collocations.BigramAssocMeasures()
 
  # get all the bigrams 
  #finder = BigramCollocationFinder.from_words(all_words_list) 

  ham_words_list = []
  for email in hamdocs:
      for words in email:
          ham_words_list.append(words)
  
  spam_words_list = []
  for email in spamdocs:
      for words in email:
          spam_words_list.append(words)
          
  spam_limit = (int(0.1 * len(list(nltk.bigrams(spam_words_list)))))
  ham_limit = (int(0.1 * len(list(nltk.bigrams(ham_words_list)))))
  
  spam_finder = BigramCollocationFinder.from_words(spam_words_list)
  ham_finder = BigramCollocationFinder.from_words(ham_words_list)
  

  spam_features = spam_finder.nbest(bigram_measures.pmi, spam_limit)
  ham_features = ham_finder.nbest(bigram_measures.pmi, ham_limit)
  
  bigram_features = spam_features + ham_features
  '''
  # top 20% of bigrams
  upper_limit = int(0.2 * len(list(nltk.bigrams(all_words_list))))
  print(len(all_words_list))
  print("upper limit=", upper_limit)
  # get the 500 highest scoring bigrams
  bigram_features = finder.nbest(bigram_measures.pmi, upper_limit)
  '''
  
  'PUNCTUATION COUNT FEATURE'
  hp = []
  # get the average number of punctuations in ham
  for email in hamdocs:
      hp.append(punctuation_count(email))
  avg_ham_punct = sum(hp)/len(hp)
  
  sp = []
  # get the average number of punctuations in spam
  for email in spamdocs:
      sp.append(punctuation_count(email))
  avg_spam_punct = sum(sp)/len(sp)

  # get the punctuation count for each email
  punct_count_list = []
  for email, lab in emaildocs:
    punct_count_list.append(punctuation_count(email))

  'LEXICAL SCORE FEATURE'
  # get lexical score for all emails
  lex_list = []
  for email, lab in emaildocs:
      e_words = []
      for words in email:
          e_words.append(words)
      lex_list.append(lexical_score(e_words))

  # get lexical score for ham emails
  ham_lex = []
  for email in hamdocs:
      ham_lex.append(lexical_score(email))

  # get lexical score for spam emails
  spam_lex = []
  for email in spamdocs:
      spam_lex.append(lexical_score(email))
   
  # average lexical score of ham and spam
  avg_ham_lex = sum(ham_lex)/len(ham_lex)
  avg_spam_lex = sum(spam_lex)/len(spam_lex)

  # feature sets from each feature definition function 
  'MOST COMMON FEATURE SET'
  common_featuresets = []
  for (email, label) in emaildocs:
      common_featuresets.append((most_common_feature(email, common_word_features), label))
  
  'BIGRAM FEATURE SET'
  bigram_featuresets = []
  for(email, label) in emaildocs:
    bigram_featuresets.append((bigram_document_features(email, common_word_features, bigram_features), label))

  'PUNCTUATION COUNT FEATURE SET'
  punct_count_featuresets = []
  for (email, label) in emaildocs:
      punct_count_featuresets.append((punct_count_feature(email, punct_count_list, avg_spam_punct), label))    
  
  'LEXICAL SCORE FEATURE SET'
  lex_score_featuresets = []
  for (email,label) in emaildocs:
      lex_score_featuresets.append((lex_score_feature(email, lex_list, avg_ham_lex), label))
   
  # train classifier and show performance in cross-validation
  
  num_folds = 6
 
  'MOST COMMON WORD FEATURE CV'
  print("\n\n--------MOST COMMON WORD FEATURE CROSS VALIDATION--------\n\n")
  cross_validation_PRF(num_folds, common_featuresets, labels)
  
  'BIGRAM FEATURE CV'
  print("\n\n--------BIGRAM FEATURE CROSS VALIDATION--------\n\n")
  cross_validation_PRF(num_folds, bigram_featuresets, labels)
  
  'PUNCTUATION COUNT CV'
  print("\n\n--------PUNCTUATION COUNT FEATURE CROSS VALIDATION--------\n\n")
  cross_validation_PRF(num_folds, punct_count_featuresets, labels)
  
  'LEXICAL SCORE CROSS VALIDATION'
  print("\n\n--------LEXICAL SCORE FEATURE CROSS VALIDATION--------\n\n")
  cross_validation_PRF(num_folds, lex_score_featuresets, labels)
  
processspamham('/Users/jasonmaloney/Documents/Syracuse/IST 664 NLP/Final Project/FinalProjectData/EmailSpamCorpora 2/corpus', 5172)

"""
commandline interface takes a directory name with ham and spam subdirectories
   and a limit to the number of emails read each of ham and spam
It then processes the files and trains a spam detection classifier.

"""
'''
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: python classifySPAM.py <corpus-dir> <limit>')
        sys.exit(0)
    processspamham(sys.argv[1], sys.argv[2]
'''    
