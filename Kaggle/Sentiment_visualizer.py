# -*- coding: utf-8 -*-

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

data = pd.read_csv('input/Sentiment.csv')

# Keeping only the neccessary columns
data = data[['text','sentiment']]

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

def analyzeAndAnnotate(data):
    positiveWords = data[data['sentiment'] == 'Positive']
    positiveWords = positiveWords['text']
    negativeWords = data[data['sentiment'] == 'Negative']
    negativeWords = negativeWords['text']
    neutralWords = data[data['sentiment'] == 'Neutral']
    neutralWords = neutralWords['text']
    print("Positive words")
    wordcloud_draw(positiveWords,'white', 'visualizations/positiveWords_Cloud.png')
    print("Negative words")
    wordcloud_draw(negativeWords, 'white', 'visualizations/negativeWords_Cloud.png')
    print("Neutral words")
    wordcloud_draw(neutralWords, 'white', 'visualizations/neutralWords_Cloud.png')
    
# Call the function to make visualization figures

analyzeAndAnnotate(data)