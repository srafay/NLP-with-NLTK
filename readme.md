# Sentiment Analysis

*Sentiment analysis is becoming a popular area of research and social media analysis, especially around user reviews and tweets. It is a special case of text mining generally focused on identifying opinion polarity (positive, negative, neutral etc)*

### Simple Approach

* We can get the training data which is the sentences mapped to either positive, neutral or negative in a list
  * If we have database available in the form of simple sentences, then we can label these sentences as positive, negative or neutral
* We can perform preprocessing to clean our sentences
  * For removing the common and unimportant words (we, i, to, is etc)
  * We can split the sentence into words
  * And then remove the words with length lower than 2
* These cleaned sentences can now be converted to features
  * Features are needed so that the model could detected whether the sentence is positive, neg, or neutral
  * One simple approach is to break sentences into words
  * Then extract features by frequency of their appearance in a tweet
* After getting features from the sentences, we can train a simple model
* One such model is Naive Bayes Classifier
  * Available in NLTK (python)
  
#### Drawback
* This approach can have a drawback
* "My house is not great"
  * The word ‘great’ weights more on the positive side but the word ‘not’ is part of two negative tweets in our training set so the output from the classifier is ‘negative’
  * but 'The movie is not bad' would return ‘negative’ even if it is ‘positive’
  * thus even large training data might not help the model in this case
* "Your song is annoying"
  * The classifier thinks it is positive. The reason is that we don’t have any information on the feature name ‘annoying’
  * Larger the training sample tweets is, the better the classifier will be
