import nltk
import random
import pickle
from nltk import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
	def __init__(self,*classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf




# Import the training data here #
train_data = []

with open("training.txt", "r") as train_file:
	for line in train_file:
		temp = word_tokenize(line)
		train_data.append((temp[1:],temp[0]))

all_words = []

for x in train_data:
	for val in x[0]:
		all_words.append(val)

random.shuffle(train_data)
random.shuffle(all_words)

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())
''' 
	# generally we discard some entries from all_words
	# (which are less common) but in this case,
	# selecting all entries gives better accuracy

'''

def find_features(reviews):
	words = set(reviews)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features


train_data_words = [i[0] for i in train_data]


featuresets = [(find_features(rev), category) for (rev,category) in train_data]

# total_featuresets = len(featuresets) 	# 7086

# Dividing the data into training and testing section
training_set = featuresets[:5000]
testing_set = featuresets[5000:]


########## Training on different classifiers ########
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes accuracy (percent):", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(20)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multonomial Naive Bayes classifier accuracy (percent):", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB classifier accuracy (percent):", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression classifier accuracy (percent):", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# Very bad accuracy (scrape it!)
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC classifier accuracy (percent):", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier classifier accuracy (percent):", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC classifier accuracy (percent):", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC classifier accuracy (percent):", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

best_classifier = VoteClassifier(classifier,
								MNB_classifier,
								BernoulliNB_classifier,
								LogisticRegression_classifier,
								SGDClassifier_classifier,
								LinearSVC_classifier,
								NuSVC_classifier)
print("VoteClassifier accuracy (percent):", (nltk.classify.accuracy(best_classifier, testing_set))*100)

print("Classification:", best_classifier.classify(testing_set[0][0]), "Confidence %:", best_classifier.confidence(testing_set[0][0])*100)
print("Classification:", best_classifier.classify(testing_set[1][0]), "Confidence %:", best_classifier.confidence(testing_set[1][0])*100)
print("Classification:", best_classifier.classify(testing_set[2][0]), "Confidence %:", best_classifier.confidence(testing_set[2][0])*100)
print("Classification:", best_classifier.classify(testing_set[3][0]), "Confidence %:", best_classifier.confidence(testing_set[3][0])*100)
print("Classification:", best_classifier.classify(testing_set[4][0]), "Confidence %:", best_classifier.confidence(testing_set[4][0])*100)
print("Classification:", best_classifier.classify(testing_set[5][0]), "Confidence %:", best_classifier.confidence(testing_set[5][0])*100)

'''
# Saving Trained Classifiers
with open("NBclassifiers.pickle","wb") as save_classifer:
	pickle.dump(classifier,save_classifer)
	# pickle.dump(MNB_classifier,save_classifer)
	# pickle.dump(BernoulliNB_classifier,save_classifer)
	# pickle.dump(LogisticRegression_classifier,save_classifer)
	# pickle.dump(SGDClassifier_classifier,save_classifer)
	# pickle.dump(LinearSVC_classifier,save_classifer)
	# pickle.dump(NuSVC_classifier,save_classifer)
'''

'''
##########	Un-comment this section to save the results in a file #############

with open("analyser_results.txt", "a") as file:

	file.write("\n\nTest Run")


	file.write("\nNaive Bayes accuracy (percent): {0}".format((nltk.classify.accuracy(classifier, testing_set))*100))
	file.write("\n{0}".format(classifier.show_most_informative_features(20)))

	file.write("\nMultonomial Naive Bayes classifier accuracy (percent): {0}".format((nltk.classify.accuracy(MNB_classifier, testing_set))*100))

	file.write("\nBernoulliNB classifier accuracy (percent): {0}".format((nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100))

	file.write("\nLogisticRegression classifier accuracy (percent): {0}".format((nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100))

	file.write("\nSGDClassifier classifier accuracy (percent): {0}".format((nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100))

	file.write("\nLinearSVC classifier accuracy (percent): {0}".format((nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100))

	file.write("\nNuSVC classifier accuracy (percent): {0}".format((nltk.classify.accuracy(NuSVC_classifier, testing_set))*100))

##############################################################################
'''


real_testing_set = []


with open("testdata.txt", "r") as test_file:
	for line in test_file:
		try:
			temp1 = word_tokenize(line)
			real_testing_set.append(temp1)
		except:
			print("Can't read data:",line)
	print("Exit")


testing_featuresets = [find_features(rev1) for (rev1) in real_testing_set]



# print("Classification:", best_classifier.classify(real_testing_set), "Confidence %:", best_classifier.confidence(real_testing_set)*100)
with open("results_testdata.txt","a") as file:
	for real_testing_set_element in testing_featuresets:
		file.write("Classification: {0}\tConfidence %: {1}\n".format(best_classifier.classify(real_testing_set_element), best_classifier.confidence(real_testing_set_element)*100))
		# print("Classification:",best_classifier.classify(real_testing_set_element),"Confidence %:\n", best_classifier.confidence(real_testing_set_element)*100)