#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:56:15 2020

@author: jasonmaloney
"""

############################################## 
#
# HW 8 - TOPIC MODELING
#
##############################################
# follows the walk through from https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

## import packages
import glob # read all files from a folder
import pandas as pd
import numpy as np
import re
import os

# preprocessing for LDA
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models

# NLTK processing
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem.porter import *
import nltk

# sklearn for LDA
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

# visualization of topics and main words
import pyLDAvis.sklearn as LDAvis
import pyLDAvis

##############################################
# these paths are unique to your machine

#################################
#
# READ IN DATA
#
#################################

# path to the main 110 folder with all files
path = '/Users/jasonmaloney/Documents/Syracuse/IST 736 Text Mining/HW 8 - Topic Modeling/110/'
FileNameList=os.listdir(path)
print(FileNameList)


# use these for the subset
fd_path = '/Users/jasonmaloney/Documents/Syracuse/IST 736 Text Mining/HW 8 - Topic Modeling/110/110-f-d'
fr_path = '/Users/jasonmaloney/Documents/Syracuse/IST 736 Text Mining/HW 8 - Topic Modeling/110/110-f-r'
md_path = '/Users/jasonmaloney/Documents/Syracuse/IST 736 Text Mining/HW 8 - Topic Modeling/110/110-m-d'
mr_path = '/Users/jasonmaloney/Documents/Syracuse/IST 736 Text Mining/HW 8 - Topic Modeling/110/110-m-r'


## store file names
all_filenames = glob.glob(path + '*/*.txt')


# use these for the subset
fd_filenames = glob.glob(fd_path + '/*.txt')
fr_filenames = glob.glob(fr_path + '/*.txt')
md_filenames = glob.glob(md_path + '/*.txt')
mr_filenames = glob.glob(mr_path + '/*.txt')


## read in the files
# define a function to read in files and append to a dictionary
def open_files(filename_list, file_dict):
    count = 1
    for filename in filename_list:
        with open(filename, 'r', encoding = 'latin-1') as file:
            if filename in file_dict:
                continue
            # read in the txt file
            current_file = file.read()
            # lowercase all the letters
            current_file = current_file.lower()    
            # prelim cleaning - remove <WORD> and line breaks
            current_file = re.sub(r'</?\w+>', '', current_file)
            current_file = re.sub(r'\n', '', current_file) 
            current_file = re.sub(r'[.!?,:;\\\'\(\)-]', '', current_file) # remove punctuation
            current_file = re.sub(r'(\d+)', '', current_file) # remove digits
            # append to the dictionary key = filename value = cleaned text
            file_dict[filename] = current_file
            count += 1
            # close this file
        file.close()
        # create a dataframe of the texts - each column is a text - variable is file name
    filedf = pd.DataFrame(file_dict, index = range(count))
    filedf.index = range(len(filedf))
    print(count)
    # return a transposed df so the texts are in rows
    return filedf.transpose()

## all files
main_dict = {}
df = open_files(all_filenames, main_dict)
df.index = range(len(df))
df.head()


## subset
# read in female democrat files
fd_dict = {}
fd_df = open_files(fd_filenames, fd_dict)
fd_df.index = range(len(fd_df))
fd_df.head()

# read in female republican files
fr_dict = {}
fr_df = open_files(fr_filenames, fr_dict)
fr_df.index = range(len(fr_df))
fr_df.head()

# read in male democrat files
md_dict = {}
md_df = open_files(md_filenames, md_dict)
md_df.index = range(len(md_df))
md_df.head()

# read in male republican files
mr_dict = {}
mr_df = open_files(mr_filenames, mr_dict)
mr_df.index = range(len(mr_df))
mr_df.head()

# shapes of each df
print('Female Democrats have {} files'.format(fd_df.shape[0]))
print('Female Republicans have {} files'.format(fr_df.shape[0]))
print('Male Democrats have {} files'.format(md_df.shape[0]))
print('Male Republicans have {} files'.format(mr_df.shape[0]))

# get a small sample of 50 each to work with - computer couldn't handle all files
small_fd = fd_df # there are only 50 files
small_fr = fr_df # there are only 18 files
small_md = md_df.sample(75, replace = False) # get a random sample of 100
small_mr = mr_df.sample(75, replace = False)

# group together the texts
small_data = pd.concat([small_fd, small_fr, small_md, small_mr])
small_data.shape


# reindex
small_data.index = range(len(small_data))
small_data


documents = small_data[0]
#documents = df[0]
documents[1]

#########################
## sklearn -  from Dr. G's LDA code
#########################
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 

more_stop = ['mr', 'mrs','ms', 'house', 'representative', 'representatives', 'house', 'speaker', 'chairman']

stop_words = text.ENGLISH_STOP_WORDS.union(more_stop) 


NUM_TOPICS = 5
# instatiate the vectorizer
vect_LDA = CountVectorizer(stop_words = stop_words, strip_accents = 'ascii')
# vectorize the documents - get dtm
data_vect = vect_LDA.fit_transform(documents)
data_vect.shape
# create a dataframe
corpus_df_LDA = pd.DataFrame(data_vect.toarray(), columns = vect_LDA.get_feature_names())
corpus_df_LDA.shape
# filter digits from column names
corpus_df_LDA = corpus_df_LDA[corpus_df_LDA.columns.drop(list(corpus_df_LDA.filter(regex = r'(\d+)')))]
corpus_df_LDA

# LDA MODEL n_components = number of topics
lda_model = LatentDirichletAllocation(n_components = NUM_TOPICS, max_iter = 10, learning_method = 'online')
# fit the model to the vectorized data (dtm)
lda_Z = lda_model.fit_transform(data_vect)
print(lda_Z.shape)  # 218 docs, 5 topics
# fit the model to the dataframe
lda_Z_DF = lda_model.fit_transform(corpus_df_LDA)


# Build a Non-Negative Matrix Factorization Model
nmf_model = NMF(n_components = NUM_TOPICS)
nmf_Z = nmf_model.fit_transform(corpus_df_LDA)
print(nmf_Z.shape)# 218 docs, 5 topics
 
# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components = NUM_TOPICS)
lsi_Z = lsi_model.fit_transform(corpus_df_LDA)
print(lsi_Z.shape)  # 40 docs, 10 topics
 
# Let's see how the first document in the corpus looks like in
## different topic spaces
print(lda_Z_DF[0])
print(nmf_Z[0])
print(lsi_Z[0])

# create a print function to print the top 10 words and weights of each topic
def print_topics(model, vectorizer, top_n = 10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

print("LDA Model:")
print_topics(lda_model, vect_LDA)
print("=" * 20)
 
print("NMF Model:")
print_topics(nmf_model, vect_LDA)
print("=" * 20)
 
print("LSI Model:")
print_topics(lsi_model, vect_LDA)
#print("=" * 20)

####################################################
##
## VISUALIZATION
##
####################################################
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
data_vect.shape

panel = LDAvis.prepare(lda_model, data_vect, vect_LDA, mds = 'tsne')
pyLDAvis.show(panel)


import matplotlib.pyplot as plt
#vocab is a vocabulary list
vocab = vect_LDA.get_feature_names()  # change to a list
len(vocab)


print('LDA MODEL')
word_topic = np.array(lda_model.components_)
word_topic = word_topic.transpose()
num_topics = 5
num_top_words = 10
vocab_array = np.asarray(vocab)
len(vocab)
#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 10
plt.figure()
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    plt.suptitle('LDA Model')
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()

print('NMF MODEL')
word_topic = np.array(nmf_model.components_)
word_topic = word_topic.transpose()
num_topics = 5
num_top_words = 10
vocab_array = np.asarray(vocab)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 10
plt.figure()
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    plt.suptitle('NMF Model')
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()

print('LSI MODEL')
word_topic = np.array(lsi_model.components_)
word_topic = word_topic.transpose()
num_topics = 5
num_top_words = 10
vocab_array = np.asarray(vocab)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 10
plt.figure()
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    plt.suptitle('LSI Model')
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()




#################################
#
# STEM AND LEMMATIZE DATA
#
#################################

# instantiate the stemmer
stemmer = PorterStemmer()

# function to lemmatize and stem preprocessing steps
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos = 'v'))

# function to preprocess text
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# select a document to preview the function
doc_sample = documents[105]
print(doc_sample)
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

# preprocess all documents
processed_docs = documents.map(preprocess)

# create a bag of words dictionary
# contains the number of times a word appears in training set
dictionary = gensim.corpora.Dictionary(processed_docs)

# take a look at the dictionary...can set count to whatever number
# of items you want to see
count = 0
for key, value in dictionary.iteritems():
    print(key, value)
    count += 1
    if count > 15:
        break

# Filter out tokens that appear in 
# 1. less than 15 documents or 
# 2. more than 0.5 documents
# After the first 2, keep only the first 100,000 most frequent tokens
dictionary.filter_extremes(no_below = 15, no_above = 0.5, keep_n = 100000)
print(dictionary)

###### BAG OF WORDS
# for each document - create a dictionary reporting how many words 
# and how many times they appear - save to bow_corpus

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
len(bow_corpus)

bow_corpus[35]
# each word is represented by a number

# check out the bag of words for one document
bow_doc5 = bow_corpus[5]
len(bow_doc5)
for i in range(len(bow_doc5)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc5[i][0], # get word number
                                                     dictionary[bow_doc5[i][0]], # actual word
                                                     bow_doc5[i][1])) # count of word

######################
# TF-IDF - I think this is unused
######################
from gensim import corpora, models

# create the model
tfidf = models.TfidfModel(bow_corpus)
# fit the model
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

#########################
# LDA with BAG OF WORDS
#########################
    
# train the LDA model 
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics = 10, id2word = dictionary, passes = 2, workers = 2)

# for each topic - explore the words occuring in that topic and their relative weight
for idx, topic in lda_model.print_topics(-1):
    print('Topic {}, \nWords: {}'.format(idx, topic))



word_topic = np.array(lda_model.components_)
word_topic = word_topic.transpose()
num_topics = 10
num_top_words = 10
vocab_array = np.asarray(vocab)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 10

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()




