
##########################################
#				 PART 2 				 #
##########################################
import time
from part1 import *

def get_count(word,txt_file):
	"""
	FUNCTION:
		This function takes as input the string 'word' and text file 'txt_file' and outputs
		the count of that word in that text file.
	INPUT:
		word (string)			: word to be searched for in txt_file
		txt_file (.txt file)	: a text file of real or fake news headlines
	OUTPUT:
		count (int)				: number of occurences of word in the text file txt_file
	"""
	count = 0
	with open(txt_file) as f:
		for line in f:
			words = line.split()
			count += words.count(word)
	return count

def create_count_dict(x,y):
	"""
	FUNCTION:
		This function takes as input the x, y dataset pair (headlines, classification) and outputs
		a dictionary containing the count of the number of headlines each word is in given a class.
	INPUT:
		x (list)				: list of headlines data
		y (list)				: list of headlines classification
	OUTPUT:
	word_counts (dict)			: a dictionary where the keys are "real" or "fake" and the values are
								  subdictionaries with keys being words and values being the number of
								  headlines that that word occurs in given the class.
	"""	
	word_counts = {"real": {}, "fake": {}} # contains the number of headlines a word occurs in given a class
	i = 0

	for headline in x:
		headline_class = y[i] # y[i] is the classification corresponding to the x[i] headline
		words = set(headline.split()) #
		for word in words:
			if word not in word_counts[headline_class]:
				word_counts[headline_class][word] = 0 # if word not in dictionary, initialize a (key,value) pair --> (word,count), where count = 0
			word_counts[headline_class][word] += 1
		i+=1
	return word_counts

def get_word_set(txt_file):
	"""
	FUNCTION:
		This function takes as input the text file 'txt_file' and outputs all the unique words
		(i.e., the set of words) within that text file.
	INPUT:
		txt_file (.txt file)	: a text file of real or fake news headlines
	OUTPUT:
		word_set (list)			: a list of all the unique words in the text file txt_file
	"""
	word_list = []
	with open(txt_file) as f:
		for line in f:
			words = line.split()
			word_list.extend(words)
	word_set = set(word_list)
	return word_set

def create_prob_dist(x,y,m=100,p=0.01):
	"""
	FUNCTION:
		This function takes as input the x, y dataset pair (headlines, classification) and outputs
		a probability dictionary containing the probabilities P(xi = 1|y = c) for all words.
	INPUT:
		x (list)				: list of headlines data
		y (list)				: list of headlines classification
	OUTPUT:
		P (dict)				: a dictionary where the keys are "real" or "fake" and the values are
								  subdictionaries with keys being words and values being their conditional
								  probabilities P(xi = 1 | c).
	"""	
	global tot_headlines_count
	global word_counts

	P = {"real": {}, "fake": {}} # initialize the dict which will be the discrete probability distribution of words
	i = 0
	for headline in x:
		classification = y[i]
		words = set(headline.split())
		for word in words:
			# IMPLEMENTING: P(xi = 1| y = c) = count(xi,c)/count(c)
			P[classification][word] = float(word_counts[classification][word] + m*p) / float(tot_headlines_count[classification] + m)
		i+=1
	return P

def get_accuracy(y_pred,y):
	"""
	FUNCTION:
		This function takes as input the y vector of correct classifications and the y_pred vector
		of predicted classifications and outputs the accuracy (% correct).
	INPUT:
		y_pred (list)			: list of predicted headline classifications
		y (list)				: list of actual headline classifications
	OUTPUT:
	accuracy (float)			: a float representing the percent number of correct classifications
	"""	
	correct = 0
	total = 0
 	i = 0
	while i < len(y_pred):
		if y_pred[i] == y[i]:
			correct += 1
		total += 1
		i += 1
	accuracy = float(correct)/float(total)

	return accuracy*100

def naive_bayes(P,x,y):
	"""
	FUNCTION:
		This function takes as input the probability dictionary containing the conditional probabilities
		of all the words in the training set, as well the x, y dataset pair (headlines, classifications)
		that we want to predict using naive bayes.
	INPUT:
		P (dict)				: a dictionary where the keys are "real" or "fake" and the values are
								  subdictionaries with keys being words and values being their conditional
								  probabilities P(xi = 1 | c).
		x (list)				: list of headlines data
		y (list)				: list of headlines classification
	OUTPUT:
		accuracy (float)		: the accuracy of the naive bayes implementation # MAY CHANGE
	"""	
	i = 0
	y_pred = []

	for headline in x:
		sent_prob = {"real": 1, "fake": 1}
		words = headline.split()
		for word in words:
			if word not in P["real"]:
				sent_prob["real"] *= 0
			else:
				sent_prob["real"] *= float(P["real"][word]) # MAKE THIS LOG (USE HINT FROM PROJECT PDF)
			if word not in P["fake"]:
				sent_prob["fake"] *= 0
			else:
				sent_prob["fake"] *= float(P["fake"][word])
		predicted_class = max(sent_prob, key=sent_prob.get) # returns the key corresponding to the max value in the dictionary
		y_pred.append(predicted_class)
		i += 1
	accuracy = get_accuracy(y_pred,y)

	return accuracy

def part2():
	word_set = get_word_set("clean_real.txt")

	# GET DATASETS: HEADLINES (x) AND CLASSIFICATIONS (y)
	x_train, y_train, x_val, y_val, x_test, y_test = get_datasets()	# NOTE: Ratio of real to fake news headlines in the datasets is ~ 1.5.

	# GET TOTAL COUNTS (USED AS NORMALIZATION FACTORS)
	global tot_headlines_count
	tot_headlines_count = {"real": 0, "fake": 0}
	for classification in y_train:
		tot_headlines_count[classification] += 1

	# GET DICTIONARY CONTAINING NUMBER OF HEADLINES A WORD OCCURS IN GIVEN THE CLASS
	global word_counts
	word_counts = create_count_dict(x_train,y_train)

	# TRAIN (i.e., BUILD THE DISCRETE PROBABILITY DISTRIBUTION)
	P = create_prob_dist(x_train,y_train)
	accuracy = naive_bayes(P,x_val,y_val)
	print(accuracy)

#________________________ RUN PART2 ________________________
# part2()






