import os
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier
train_path = "./aclImdb/train"
test_path = "./aclImdb/imdb_te.csv"

stopwords = open("./aclImdb/stopwords.txt", 'r').read().split()

def load_data(trainpath):
    traindf = pd.DataFrame(columns = ['row_number', 'text', 'polarity'])

    train_direc = os.listdir(trainpath)[0:2]
    for direc in train_direc:
        if direc=='neg':
            filelist = os.listdir(trainpath + '/neg')
            for i, file in enumerate(filelist):
                # frame = pd.read_csv(train_path+'/pos/'+file)
                f = open(trainpath + '/neg/' + file, "r")
                try:
                    txt = f.read()
                except UnicodeDecodeError:
                    txt = ''
                traindf = traindf.append(pd.DataFrame([[i, txt, 0]], columns=['row_number', 'text', 'polarity']))

        elif direc=='pos':
            filelist = os.listdir(trainpath + '/pos')
            for i, file in enumerate(filelist):
                # frame = pd.read_csv(train_path+'/pos/'+file)
                f = open(trainpath + '/pos/' + file, "r")
                try:
                    txt = f.read()
                except UnicodeDecodeError:
                    txt = ''
                traindf = traindf.append(pd.DataFrame([[i, txt, 1]], columns=['row_number', 'text', 'polarity']))

    traindf.to_csv('imdb_tr.csv', index = False)
    X = traindf['text'].values
    y = traindf['polarity'].values
    return X, y

def save(filename, predictions):

    output = open(filename, 'a')

    for prediction in predictions:
        output.write(str(prediction) + "\n")
    output.close()

def tokenizer(text):
    only_alnum_pattern = re.compile('([^\s\w]|_)+')
    no_tags_pattern = re.compile('<[^>]*>')
    text = no_tags_pattern.sub('', text.lower())
    text = only_alnum_pattern.sub('', text.lower())
    tokens = [w for w in text.split() if w not in stopwords]
    return tokens

load_data(trainpath=train_path, testpath=test_path)

def unigram(Xtrain, ytrain, testdf):


    Xtest = testdf['text'].values
    vect = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
    Xtrain_vectorized = vect.fit_transform(Xtrain)
    Xtest_vectorized = vect.transform(Xtest)
    clf = SGDClassifier(loss='hinge', penalty='l1')
    clf.fit(Xtrain_vectorized, ytrain)
    predictions = clf.predict(Xtest_vectorized)
    save('unigram.output.txt', predictions)

def unigramtfidf(Xtrain, ytrain, testdf):
    vect = TfidfVectorizer(stop_words=stopwords,decode_error='ignore', use_idf=True, ngram_range=(1, 1), preprocessor=None, tokenizer=tokenizer, analyzer='word')

    Xtest = testdf['text'].values
    Xtrain_vectorized = vect.fit_transform(Xtrain)
    Xtest_vectorized = vect.transform(Xtest)
    clf = SGDClassifier(loss='hinge', penalty='l1')
    clf.fit(Xtrain_vectorized, ytrain)
    predictions = clf.predict(Xtest_vectorized)
    save('unigramtfidf.output.txt', predictions)

def bigram(Xtrain, ytrain, testdf):
    vect = HashingVectorizer(decode_error='ignore', ngram_range=(1, 2), preprocessor=None, tokenizer=tokenizer, analyzer='word')

    Xtest = testdf['text'].values
    Xtrain_vectorized = vect.fit_transform(Xtrain)
    Xtest_vectorized = vect.transform(Xtest)
    clf = SGDClassifier(loss='hinge', penalty='l1')
    clf.fit(Xtrain_vectorized, ytrain)
    predictions = clf.predict(Xtest_vectorized)
    save('bigram.output.txt', predictions)

def bigramtfidf(Xtrain, ytrain, testdf):
    vect = TfidfVectorizer(decode_error='ignore', use_idf=True, ngram_range=(1, 2), preprocessor=None, tokenizer=tokenizer, analyzer='word')
    Xtest = testdf['text'].values
    Xtrain_vectorized = vect.fit_transform(Xtrain)
    Xtest_vectorized = vect.transform(Xtest)
    clf = SGDClassifier(loss='hinge', penalty='l1')
    clf.fit(Xtrain_vectorized, ytrain)
    predictions = clf.predict(Xtest_vectorized)
    save('bigramtfidf.output.txt', predictions)