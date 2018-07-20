# -*- coding: utf-8 -*-

import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras import utils

data = pd.read_csv('input/Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
tweets = pd.DataFrame(columns=['text','sentiment'])
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words("english"))

for index, row in data.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    #tweets.append((words_without_stopwords,row.sentiment))
    tweets = tweets.append({'text':' '.join(words_without_stopwords), 'sentiment':''.join(row.sentiment)}, ignore_index=True)


train_size = int(len(tweets) * .8)

train_posts = tweets['text'][:train_size]
train_tags = tweets['sentiment'][:train_size]

test_posts = tweets['text'][train_size:]
test_tags = tweets['sentiment'][train_size:]

vocab_size = 500
tokenize = text.Tokenizer(num_words=vocab_size)
tokenize.fit_on_texts(train_posts)

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_labels = 3
batch_size = 128

from sklearn.utils import class_weight
y_ints = [y.argmax() for y in y_train]
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_ints),
                                                 y_ints)


model = Sequential()
model.add(Dense(32, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=512, 
                    verbose=1, 
                    validation_split=0.1,
                    class_weight=class_weights)

def plotLossnAcc(history):
    print(history.history.keys())
    #  "Accuracy"
    plt.figure(figsize=(7,5))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.figure(figsize=(7,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
plotLossnAcc(history)

score = model.evaluate(x_test, y_test, 
                       batch_size=batch_size, verbose=1)

print('\nTest score:', score[0])
print('Test accuracy:', score[1])

def getAccuracy(x_test, test_tags):
    correct = 0
    for i in range(len(x_test)):
        prediction = model.predict(np.array([x_test[i]]))
        predicted_label = text_labels[np.argmax(prediction)]
        if (test_tags.iloc[i] == predicted_label):
            correct += 1
    return (correct/len(x_test))

text_labels = encoder.classes_ 

print ("Accuracy is: " + str(getAccuracy(x_test, test_tags)))

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_posts.iloc[i][:50], "...")
    print('Actual label:' + test_tags.iloc[i])
    print("Predicted label: " + predicted_label + "\n")
    
y_softmax = model.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)
    
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
    
cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(14,10))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
plt.show()