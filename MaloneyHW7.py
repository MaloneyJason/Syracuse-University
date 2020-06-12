#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:17:57 2020

@author: jasonmaloney
"""

################################################################## 
#                                                                #
# This program will classify the sentiment of tweets to Airline  #
#                                                                #
##################################################################


##################################################################
# LOAD THE PACKAGES
##################################################################
import re  # reg ex for data cleaning
import pandas as pd # data frames
import numpy as np # arrays

from sklearn.model_selection import train_test_split # train/test sets
from sklearn.feature_extraction.text import CountVectorizer # boolean and frequncy vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer # weighted vectorizer

from sklearn.naive_bayes import MultinomialNB # multinomial naive bayes
from sklearn.svm import LinearSVC # SVM - one vs all models
from sklearn.metrics import confusion_matrix # prediction results
from sklearn.metrics import classification_report # performance metrics

import seaborn as sns # plots
import matplotlib.pyplot as plt # plot settings

import warnings # ignore warnings
warnings.filterwarnings('ignore')

##################################################################
# READ THE DATA
##################################################################

# change path to your local folder
path = '/Users/jasonmaloney/Documents/Syracuse/IST 736 Text Mining/HW 7 - SVM classification/'

# data is from https://www.kaggle.com/crowdflower/twitter-airline-sentiment

# read in the data
airdf = pd.read_csv(path + 'Tweets.csv')
airdf.shape
# only need to keep the sentiment and the text of the reviews
airdf.columns
# airline_sentiment, text
cols_to_keep = ['airline_sentiment', 'text']
air = airdf[cols_to_keep] 
air.airline_sentiment.value_counts()

##################################################################
# CLEAN THE DATA
##################################################################

# store the airline name - plots in the report
airline_names = air['text'].str.extract(pat = '(@\w+)')
air['name'] = airline_names
# strip the airline - each tweet begins with @airlinename
air['text'] = air['text'].str.replace('@\w+', '', regex = True)

# convert positive = 0, neutral = 1, negative = 2 for easy id of confusion matrix
air['sent'] = 0
air.loc[air['airline_sentiment'] == 'positive']['sent']
air['sent'][air['airline_sentiment'] == 'neutral'] = 1
air['sent'][air['airline_sentiment'] == 'negative']= 2
air['sent'].value_counts()

#################################################################
# EXPLORE THE DATA
##################################################################

# value counts of sentiment
freq_dist = air['airline_sentiment'].value_counts()
freq_dist
print('The balance of the labels:\n', freq_dist/sum(freq_dist))
print()

# use a subset so that the data is balanced 1000 each
pos = air.loc[air['airline_sentiment'] == 'positive'][:1000]
neg = air.loc[air['airline_sentiment'] == 'negative'][:1000]
neut = air.loc[air['airline_sentiment'] == 'neutral'][:1000]
todo = [pos, neg, neut]
# join them together
df = pd.concat(todo)
df.shape

## plot of airlines with negative reviews
negair = air.loc[air['airline_sentiment'] == 'negative']['name']
negair.index = range((len(negair)))
plt.figure()
negair.value_counts()[:10].plot(kind = 'bar')
plt.title('10 Airlines with the Most Negative Tweets')
plt.show()

## plot of airlines with positive reviews
posair = air.loc[air['airline_sentiment'] == 'positive']['name']
posair.index = range(len(posair))
plt.figure()
posair.value_counts()[:10].plot(kind = 'bar')
plt.title('10 Airlines with the Most Positive Tweets')
plt.show()

## plot airlines with neutral reviews
neutair = air.loc[air['airline_sentiment'] == 'neutral']['name']
neutair.index = range(len(neutair))
plt.figure()
neutair.value_counts()[:10].plot(kind = 'bar')
plt.title('10 Airlines with the Most Neutral Tweets')
plt.show()

## plot of distribution of sentiment
plt.figure()
air['airline_sentiment'].value_counts().plot(kind = 'bar')
plt.title('Distirbution of Sentiment')
plt.xlabel('Sentiment', rotation = 0)
for i, v in enumerate(air['airline_sentiment'].value_counts()):
    plt.text(i - 0.1, v + 50, str(v))
plt.show()

## df 5 airlines with most tweets by sentiment
mosttweets = air.loc[air['name'].isin(['@united', '@USAirways', '@AmericanAir', '@SouthwestAir', '@JetBlue'])][['name', 'airline_sentiment']]
mosttweets.index = range(len(mosttweets))
mosttweets['count'] = 1

# plot of 5 airlines with most tweets by sentiment
plt.figure()
sns.countplot(x = 'name', hue = 'airline_sentiment', data = mosttweets)
plt.xlabel('Airline Twitter Handle')
plt.ylabel('')
plt.title('Top 5 Airlines Tweets by Sentiment')
plt.show()

### separate the text and the labels for vectorization
### df is the balanced data frame
text = df['text']
labels = df['sent']

##################################################################
# VECTORIZE THE DATA
##################################################################

### define vectorizers
gen_vect = CountVectorizer(encoding = 'latin-1')

# boolean
bool_vect = CountVectorizer(encoding = 'latin-1', stop_words = 'english', min_df = 5,  binary = True)
# frequency counts
freq_vect = CountVectorizer(encoding = 'latin-1', stop_words = 'english', min_df = 5, binary = False)
# L1 normalized by document length
norm_vect = TfidfVectorizer(encoding = 'latin-1', stop_words = 'english', min_df = 5, norm = 'l1', use_idf = False)
# tf-idf vectorizer - min_df = 5
tfidf_vect = TfidfVectorizer(encoding = 'latin-1', stop_words = 'english', min_df = 5, use_idf = True)

### get the vocabulary
# all words
gen_vec = gen_vect.fit_transform(text)
# filtered vectorizers
bool_vec = bool_vect.fit_transform(text)
freq_vec = freq_vect.fit_transform(text)
norm_vec = norm_vect.fit_transform(text)
tfidf_vec = tfidf_vect.fit_transform(text)

### vectorize the test data - only use transform - don't add new tokens
#bool_test_vec = bool_vect.transform(text_test)
#freq_test_vec = freq_vect.transform(text_test)
#tfidf_test_vec = tfidf_vect.transform(text_test)

### How big is the vocabulary for each?
print('Total words: {}'.format(gen_vec.shape[1]))
print('Boolean vocabulary: {}'.format(bool_vec.shape[1]))
print('Frequency vocabulary: {}'.format(freq_vec.shape[1]))
print('Normalized vocabulary: {}'.format(norm_vec.shape[1]))
print('TFIDF vocabulary: {}'.format(tfidf_vec.shape[1]))


##################################################################
# TRAIN-TEST DATA
##################################################################

###### train/test data 60/40 - set random_state for reproducible results
bool_train_vec, bool_test_vec, label_train, label_test = train_test_split(bool_vec, labels, test_size = 0.4, random_state = 19)
freq_train_vec, freq_test_vec, label_train, label_test = train_test_split(freq_vec, labels, test_size = 0.4, random_state = 19)
norm_train_vec, norm_test_vec, label_train, label_test = train_test_split(norm_vec, labels, test_size = 0.4, random_state = 19)
tfidf_train_vec, tfidf_test_vec, label_train, label_test = train_test_split(tfidf_vec, labels, test_size = 0.4, random_state = 19)


##################################################################
# SUPPORT VECTOR MACHINES MODELS 
##################################################################
import time

t0 = time.time()
# initialize the LinearSVC model
svm_clf = LinearSVC(C = 1)
# use the training data to train the model
# need the vector and the sentiment label
print('\n ----------- SVM BOOLEAN MODEL ----------- ')

### BOOLEAN 
svm_clf.fit(bool_train_vec, label_train)

# predictions and confusion matrix
bool_pred = svm_clf.predict(bool_test_vec)
bool_cm = confusion_matrix(label_test, bool_pred, labels = [0, 1, 2])
print('Confusion Matrix of Predictions.\n Actual = rows Predicted = columns')
print(bool_cm)
sum(bool_cm).sum()
t1 = time.time()
print('elapsed seconds: ', t1 - t0)
## confusion matrix
plt.figure()
sns.heatmap(bool_cm/sum(bool_cm).sum(), annot = True, cmap = 'RdBu', fmt = '.2%')
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('SVM Boolean Confusion Matrix')
plt.show()

## performance metrics
print(' ----------- SVM Boolean Performance Metrics')
target_names = ['0','1','2']
print(classification_report(label_test, bool_pred, target_names=target_names))
print()

print('\n ----------- SVM FREQUENCY COUNT MODEL ----------- ')
### FREQUENCY COUNT
svm_clf.fit(freq_train_vec, label_train)
freq_pred = svm_clf.predict(freq_test_vec)
freq_cm = confusion_matrix(label_test, freq_pred, labels = [0, 1, 2])
print('Confusion Matrix of Predictions.\n Actual = rows Predicted = columns')
print(freq_cm)

## confusion matrix
plt.figure()
sns.heatmap(freq_cm/sum(freq_cm).sum(), annot = True, cmap = 'RdBu', fmt = '.2%')
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('SVM Frequency Count Confusion Matrix')
plt.show()

## performance metrics
print(' ----------- SVM Frequency Count Performance Metrics ')
target_names = ['0','1','2']
print(classification_report(label_test, freq_pred, target_names=target_names))

print('\n ----------- SVM NORMALIZED (L1) MODEL ----------- ')
### NORMALIZED BY DOCUMENT LENGTH
t0 = time.time()
svm_clf.fit(norm_train_vec, label_train)
norm_pred = svm_clf.predict(norm_test_vec)
norm_cm = confusion_matrix(label_test, norm_pred, labels = [0, 1, 2])
t1 = time.time()
print('Confusion Matrix of Predictions.\n Actual = rows Predicted = columns')
print(norm_cm)
print('elapsed seconds: ', t1 - t0)

## confusion matrix
plt.figure()
sns.heatmap(norm_cm/sum(norm_cm).sum(), annot = True, cmap = 'RdBu', fmt = '.2%')
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('SVM Normalized by Document Length Confusion Matrix')
plt.show()

## performance metrics
print(' ----------- SVM Normalized Performance Metrics')
target_names = ['0','1','2']
print(classification_report(label_test, norm_pred, target_names = target_names))


print('\n ----------- SVM TFIDF MODEL ----------- ')
### TFIDF
svm_clf.fit(tfidf_train_vec, label_train)
tfidf_pred = svm_clf.predict(tfidf_test_vec)
tfidf_cm = confusion_matrix(label_test, tfidf_pred, labels = [0, 1, 2])
print('Confusion Matrix of Predictions.\n Actual = rows Predicted = columns')
print(tfidf_cm)

## confusion matrix
plt.figure()
sns.heatmap(tfidf_cm/sum(tfidf_cm).sum(), annot = True, cmap = 'RdBu', fmt = '.2%')
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('SVM TFIDF Confusion Matrix')
plt.show()

## performance metrics
print(' ----------- SVM TFIDF Performance Metrics ')
target_names = ['0','1','2']
print(classification_report(label_test, tfidf_pred, target_names=target_names))
print('\n\n')

# all confusion matrices in a 2x2 grid
plt.figure()
fig, axes = plt.subplots(2, 2, sharex = True, sharey = True)
# boolean
sns.heatmap(bool_cm/sum(bool_cm).sum(), annot = True, ax = axes[0,0], cmap = 'RdBu', fmt = '.2%', cbar = False)
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
axes[0,0].set_title('Boolean')
# frequency
sns.heatmap(freq_cm/sum(freq_cm).sum(), annot = True, ax = axes[0,1], cmap = 'RdBu', fmt = '.2%', cbar = False)
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
axes[0,1].set_title('Frequency Count')
# normalized (L1)
sns.heatmap(norm_cm/sum(norm_cm).sum(), annot = True, ax = axes[1,0], cmap = 'RdBu', fmt = '.2%', cbar = False)
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
axes[1,0].set_title('Normalized by Document Length')
# tfidf
sns.heatmap(tfidf_cm/sum(tfidf_cm).sum(), annot = True, ax = axes[1,1], cmap = 'RdBu', fmt = '.2%', cbar = False)
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
axes[1,1].set_title('TFIDF')
# main title and main axes labels
plt.suptitle('Confusion Matrices of Prediction Results SVM Models')
fig.text(0.5, 0.04, 'Predicted Sentiment', ha ='center')
fig.text(0.04, 0.5, 'Actual Sentiment', va ='center', rotation = 'vertical')

plt.show()





##################################################################
# MULTINOMIAL NAIVE BAYES MODELS 
##################################################################

# instantiate the model
multiNB = MultinomialNB()

print(' ----------- NAIVE BAYES BOOLEAN MODEL ----------- ')
### BOOLEAN 
multiNB.fit(bool_train_vec, label_train)
bool_pred = multiNB.predict(bool_test_vec)
boolnb_cm = confusion_matrix(label_test, bool_pred)    
print('Confusion Matrix of Predictions.\n Actual = rows Predicted = columns')
print(boolnb_cm)

plt.figure()
sns.heatmap(boolnb_cm/sum(boolnb_cm).sum(), annot = True, cmap = 'RdBu', fmt = '.2%')
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Naive Bayes Boolean Confusion Matrix')
plt.show()

print(' ----------- Naive Bayes Boolean Performance Metrics')
target_names = ['0','1','2']
print(classification_report(label_test, bool_pred, target_names=target_names))

print('\n ----------- NAIVE BAYES FREQUENCY COUNT MODEL ----------- ')
### FREQUENCY COUNTS 
t0 = time.time()
multiNB.fit(freq_train_vec, label_train)
freqnb_pred = multiNB.predict(freq_test_vec)
freqnb_cm = confusion_matrix(label_test, freqnb_pred)    
print('Confusion Matrix of Predictions.\n Actual = rows Predicted = columns')
print(freqnb_cm)
t1 = time.time()
print('elapsed seconds: ', t1 - t0)

# confusion matrix
plt.figure()
sns.heatmap(freqnb_cm/sum(freqnb_cm).sum(), annot = True, cmap = 'RdBu', fmt = '.2%')
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Naive Bayes Frequency Count Confusion Matrix')
plt.show()

# performance metrics
print(' ----------- Naive Bayes Frequency Count Performance Metrics')
target_names = ['0','1','2']
print(classification_report(label_test, freqnb_pred, target_names=target_names))

print('\n ----------- NAIVE BAYES NORMALIZED (L1) MODEL ----------- ')
### NORMALIZED BY DOCUMENT LENGTH
multiNB.fit(norm_train_vec, label_train)
normnb_pred = multiNB.predict(norm_test_vec)
normnb_cm = confusion_matrix(label_test, normnb_pred)    
print('Confusion Matrix of Predictions.\n Actual = rows Predicted = columns')
print(normnb_cm)

# confusion matrix
plt.figure()
sns.heatmap(normnb_cm/sum(normnb_cm).sum(), annot = True, cmap = 'RdBu', fmt = '.2%')
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Naive Bayes Normalized by Document Length Confusion Matrix')
plt.show()

# performance metrics
print(' ----------- Naive Bayes Frequency Count Performance Metrics')
target_names = ['0','1','2']
print(classification_report(label_test, normnb_pred, target_names = target_names))

print('\n ----------- NAIVE BAYES TFIDF MODEL ----------- ')
### TFIDF
multiNB.fit(tfidf_train_vec, label_train)
tfidfnb_pred = multiNB.predict(tfidf_test_vec)
tfidfnb_cm = confusion_matrix(label_test, tfidfnb_pred)    
print('Confusion Matrix of Predictions.\n Actual = rows Predicted = columns')
print(tfidfnb_cm)

# confusion matrix
plt.figure()
sns.heatmap(tfidfnb_cm/sum(tfidfnb_cm).sum(), annot = True, cmap = 'RdBu', fmt = '.2%')
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Naive Bayes TFIDF Confusion Matrix')
plt.show()

# performance metrics
print(' ----------- Naive Bayes TFIDF Performance Metrics ')
target_names = ['0','1','2']
print(classification_report(label_test, tfidfnb_pred, target_names = target_names))

# all confusion matrices in a 2x2 grid
plt.figure()
fig, axes = plt.subplots(2, 2, sharex = True, sharey = True)
#boolean
sns.heatmap(boolnb_cm/sum(boolnb_cm).sum(), annot = True, ax = axes[0,0], cmap = 'RdBu', fmt = '.2%', cbar = False)
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
axes[0,0].set_title('Boolean')
# frequency
sns.heatmap(freqnb_cm/sum(freqnb_cm).sum(), annot = True, ax = axes[0,1], cmap = 'RdBu', fmt = '.2%', cbar = False)
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
axes[0,1].set_title('Frequency Count')
# normalized (L1)
sns.heatmap(normnb_cm/sum(normnb_cm).sum(), annot = True, ax = axes [1,0], cmap = 'RdBu', fmt = '.2%', cbar = False)
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
axes[1,0].set_title('Normalized by Document Length')
# tfidf
sns.heatmap(tfidfnb_cm/sum(tfidfnb_cm).sum(), annot = True, ax = axes[1,1], cmap = 'RdBu', fmt = '.2%', cbar = False)
plt.xticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
plt.yticks([0.5, 1.5, 2.5], labels = ['pos', 'neu', 'neg'])
axes[1,1].set_title('TFIDF')
# main title and axes labels
plt.suptitle('Confusion Matrices of Prediction Results Naive Bayes Models')
fig.text(0.5, 0.04, 'Predicted Sentiment', ha ='center')
fig.text(0.04, 0.5, 'Actual Sentiment', va ='center', rotation = 'vertical')

plt.show()


