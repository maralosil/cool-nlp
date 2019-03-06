import string
import random
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english') + list(string.punctuation))

def remove_stopwords(words):
  return [word for word in words if word not in stop_words]

freq_dist = nltk.FreqDist(remove_stopwords(
                            movie_reviews.words()))

common_words = [word for word,_ in freq_dist.most_common(2000)]

reviews = [(set(remove_stopwords(movie_reviews.words(fileid))),
            fileid, category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(reviews)

def make_features(words):
  features = {}
  for word in common_words:
    features[word] = (word in words)
  return features

features_set = [(make_features(words), category)
                for (words, _, category) in reviews]

index = int(len(reviews) * 0.7)
training_set = features_set[:index]
testing_set = features_set[index:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print('Accuracy: ', nltk.classify.accuracy(classifier, testing_set) * 100)

classifier.show_most_informative_features(10)

