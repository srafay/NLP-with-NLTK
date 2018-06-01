# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

data = pd.read_csv('Sentiment.csv')

# Keeping only the neccessary columns
data = data[['text','sentiment']]

# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)

# Split positive and negative words for making a wordcloud
train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']
train_neu = train[ train['sentiment'] == 'Neutral']
train_neu = train_neu['text']

def wordcloud_draw(data, color = 'white', filename='wordCloud.png'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(filename)
    plt.show()
    
print("Positive words")
wordcloud_draw(train_pos,'white', 'positiveWords_Cloud.png')
print("Negative words")
wordcloud_draw(train_neg, 'white', 'negativeWords_Cloud.png')
print("Neutral words")
wordcloud_draw(train_neu, 'white', 'neutralWords_Cloud.png')

