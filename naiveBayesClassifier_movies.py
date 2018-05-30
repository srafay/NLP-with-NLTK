import nltk.classify.util

# Download movie_reviews dataset if not available locally
nltk.download("movie_reviews")

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

# Simple Bag of Words model in which every word is a feature with value of True
def word_feats(words):
    return dict([(word, True) for word in words])

# Separate ids of neg and pos reviews
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

# Extract words (features) from the reviews and label them as neg and pos respectively
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

# 75% Train - Test split
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4

# Train = [start till 75%]
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]

# Test = [75% till end]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

# Training the simplest classifier
classifier = NaiveBayesClassifier.train(trainfeats)

# Using builtin function for checking the accuracy of the model
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)

# Shows which features are most effective for classifying the review as pos or neg
classifier.show_most_informative_features()
