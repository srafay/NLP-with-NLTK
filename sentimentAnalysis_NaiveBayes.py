import nltk

pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

test_tweets = [('I feel happy this morning', 'positive'),
              ('Larry is my friend', 'positive'),
              ('I do not like that man', 'negative'),
              ('My house is not great', 'negative'),
              ('Your song is annoying', 'negative')]

tweets = []

# Remove words smaller than 2 characters like "we", "i", "to", "is"
for (words, sentiment) in pos_tweets + neg_tweets:
    # If a word has length > 2 chars, then lowercase it and append in words_filtered
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

# Extract the list of words from the tweet
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

# Words are extracted as features by frequency of their appearance in a tweet
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

# Classifier needs the features as a dictionary
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

training_set = nltk.classify.apply_features(extract_features, tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
tweet = 'Larry is my friend'
print classifier.classify(extract_features(tweet.split()))

def getAccuracy(classifier, test_set):
    total = len(test_set)
    correct = 0
    for i in range (total):
	if (test_set[i][1] == classifier.classify(extract_features(test_set[i][0].split())) ):
	    correct += 1
    return (correct*100/total)

print 'Accuracy is: ', getAccuracy(classifier, test_tweets)
