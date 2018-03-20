
##########################################
#			 PART 2 AND PART 3			 #
##########################################
import time
from part1 import *
from math import log as log
from math import exp as exp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
								  probabilities P(xi = 1 | c)
	"""	
	global tot_headlines_count
	global word_counts

	P = {"real": {}, "fake": {}} # initialize the dict which will be the discrete probability distribution of words
	i = 0

	for headline in x:
		classification = y[i]
		words = set(headline.split())

		if classification == "real":
			not_classification = "fake"
		else:
			not_classification = "real"

		for word in words:
			# IMPLEMENTING: P(xi = 1| y = c) = count(xi,c)/count(c)
			P[classification][word] = float(word_counts[classification][word] + m*p) / float(tot_headlines_count[classification] + m)
			if word not in P[not_classification]:
				if word not in word_counts[not_classification]:
					P[not_classification][word] =float(m*p) / float(tot_headlines_count[not_classification] + m)
				else:
					P[not_classification][word] =float(word_counts[not_classification][word] + m*p) / float(tot_headlines_count[not_classification] + m)
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

def test_naive_bayes(P,x,y):
	"""
	FUNCTION:
		This function takes as input the probability dictionary containing the conditional probabilities
		of all the words in the training set, as well the x, y dataset pair (headlines, classifications)
		that we want to predict using naive bayes.
	INPUT:
		P (dict)				: a dictionary where the keys are "real" or "fake" and the values are
								  subdictionaries with keys being words and values being their conditional
								  probabilities P(xi = 1 | c)
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
				# sent_prob["real"] *= 0
				sent_prob["real"] += log(0.000000000000001) #log(0)
			else:
				# sent_prob["real"] *= float(P["real"][word]) # MAKE THIS LOG (USE HINT FROM PROJECT PDF)
				sent_prob["real"] += log(float(P["real"][word]))
			if word not in P["fake"]:
				# sent_prob["fake"] *= 0
				sent_prob["fake"] += log(0.000000000000001) #log(0)
			else:
				# sent_prob["fake"] *= float(P["fake"][word])
				sent_prob["fake"] += log(float(P["fake"][word]))
		sent_prob["real"] = exp(sent_prob["real"])
		sent_prob["fake"] = exp(sent_prob["fake"])
		predicted_class = max(sent_prob, key=sent_prob.get) # returns the key corresponding to the max value in the dictionary
		y_pred.append(predicted_class)
		i += 1
	accuracy = get_accuracy(y_pred,y)

	return accuracy

def get_optimal_parameters(x_train,y_train,x_val, y_val):
	"""
	FUNCTION:
		This function takes as input the training dataset pair (x_train, y_train) and the validation
		dataset pair (x_val, y_val). The former is used to build the conditional probability distribution
		dictionary and the latter is used to test the naive bayes' algorithm with varying parameter values.
		This function sweeps through the a wide range of m and p combinations, and returns the parameters
		m and p that yield the highest accuracy on the validation set.
	INPUT:
		x_train (list)		: a list of headlines containing ~70% of the total data
		y_train (list)		: a list of classifications corresponding to the train_set headlines
		x_val (list)		: a list of headlines containing ~15% of the total data
		y_val (list)		: a list of classifications corresponding to the val_set headlines
	OUTPUT:
		m (int)				: naive bayes' parameter
		p (float)			: naive bayes' paremeter
	"""
	best_accuracy = 0
	best_parameters = [None,None]
	for m in range(10,1010,10):
		for p in [0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.10,0.50]: # for p in range(0.01,0.5,0.05): # didn't work
			P = create_prob_dist(x_train,y_train,m,p)
			accuracy = test_naive_bayes(P,x_val,y_val)
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_parameters = [m,p]
	m = best_parameters[0]
	p = best_parameters[1]
	return m, p

def get_prior_probabilities(y):
	"""
	FUNCTION:
		This function takes as input the y vector of headline classifications and returns the prior
		probabilities of the classification being "real" or "fake" based on the proportion of the
		training data in each class.
	OUTPUT:
		P_priors (dict)		: a dictionary with keys "real" and "fake" where the values are the prior
							  probabilities of these classifications (i.e., P(c = real) and P(c = fake))
	"""

	P_priors = {"real": 0, "fake": 0} # initialize prior probabilities dict to zero
	count_real, count_fake = 0, 0 # initialize counts to zero

	for classification in y:
		if classification == "real":
			count_real += 1
		else:
			count_fake += 1

	tot_count = count_real+count_fake
	P_priors["real"] = float(count_real)/float(tot_count)
	P_priors["fake"] = float(count_fake)/float(tot_count)
	return P_priors

def create_class_prob_dict(P,P_priors,m,p):
	"""
	FUNCTION:
		This function takes as input two probability dictionaries P and P_priors and outputs a dictionary
		P_classes containing the conditional probabilities of the classes given a word (i.e., P(c|w))
	INPUT:
		P (dict)			: a dictionary where the keys are "real" or "fake" and the values are
							  subdictionaries with keys being words and values being their conditional
							  probabilities P(xi = 1 | c)
		P_priors (dict)		: a dictionary with keys "real" and "fake" where the values are the prior
							  probabilities of these classifications (i.e., P(c = real) and P(c = fake))
		m (int)				: naive bayes' parameter
		p (float)			: naive bayes' paremeter
	OUTPUT:
		P_classes (dict)	: a dictionary where the keys are "real" or "fake" and the values are
							  subdictionaries with keys being classifications and values being their 
							  conditional probabilities P(c|w)
	"""
	P_classes = {}
	for classification in P:
		
		if classification == "real":
			not_classification = "fake"
		else:
			not_classification = "real"

		for word in P[classification]:
			bayes_num = P[classification][word]*P_priors[classification]
			bayes_denom = P[classification][word]*P_priors[classification] + P[not_classification][word]*P_priors[not_classification]
			bayes_res = float(bayes_num)/float(bayes_denom)
			
			if word not in P_classes:
				P_classes[word] = {classification: bayes_res}
			else:
				P_classes[word][classification] = bayes_res

	return P_classes

def create_negated_class_prob_dict(P,P_priors,m,p):
	"""
	FUNCTION:
		This function takes as input two probability dictionaries P and P_priors and outputs a dictionary
		P_classes containing the conditional probabilities of the classes given a word (i.e., P(c|not w))
	INPUT:
		P (dict)			: a dictionary where the keys are "real" or "fake" and the values are
							  subdictionaries with keys being words and values being their conditional
							  probabilities P(xi = 1 | c)
		P_priors (dict)		: a dictionary with keys "real" and "fake" where the values are the prior
							  probabilities of these classifications (i.e., P(c = real) and P(c = fake))
		m (int)				: naive bayes' parameter
		p (float)			: naive bayes' paremeter
	OUTPUT:
		P_classes_negated (dict)	: a dictionary where the keys are "real" or "fake" and the values are
							  	      subdictionaries with keys being classifications and values being their 
							  		  conditional probabilities P(c|not w)
	"""
	P_classes_negated = {}
	for classification in P:

		if classification == "real":
			not_classification = "fake"
		else:
			not_classification = "real"

		for word in P[classification]:
			bayes_num = (1-P[classification][word])*P_priors[classification]
			bayes_denom = (1-P[classification][word])*P_priors[classification] + (1-P[not_classification][word])*P_priors[not_classification]
			bayes_res = float(bayes_num)/float(bayes_denom)
			
			if word not in P_classes_negated:
				P_classes_negated[word] = {classification: bayes_res}
			else:
				P_classes_negated[word][classification] = bayes_res

	return P_classes_negated

def get_top10s(P_classes, P_classes_negated,no_stopwords=False):
	"""
	FUNCTION:
		This function takes as input two probability dictionaries P_classes and P_classes_negated 
		and outputs two dictionaries containing the top 10 words most useful in classifiying 
		headlines when those words are present or absent, for both "real" and "fake" headlines.
	INPUT:
		P_classes (dict)			: a dictionary where the keys are "real" or "fake" and the values are
							 		 subdictionaries with keys being classifications and values being their 
							 		 conditional probabilities P(c|w)
		P_classes_negated (dict)	: a dictionary where the keys are "real" or "fake" and the values are
							  	      subdictionaries with keys being classifications and values being their 
							  		  conditional probabilities P(c|not w)
		no_stopwords (boolean)		: an optional input that, when True, prevents stopwords from being returned
									  in the top 10 dictionaries
	OUTPUT:
		top_10_present (dict)		: a dictionary where the keys are "real" or "fake" and the values are
							  	      subdictionaries with keys being words and values being their conditional
							  		  probabilities P(c|w)
		top_10_absent (dict)		: a dictionary where the keys are "real" or "fake" and the values are
							  	      subdictionaries with keys being words and values being their conditional
							  		  probabilities P(c|not w)				  		  
	"""
	# Probabilistically:
	# ______________________________________________________________________________________________________________________
	# P(c = real | w = word) 
	# = P(c = real, w = word) / P(w = word)
	# = P(w = word | c = real) * P(c = real) / [P(w = word | c = real) * P(c = real) + P(w = word | c = fake) * P(c = fake)]
	#_______________________________________________________________________________________________________________________
	
	# 10 WORDS WHOSE PRESENCE MOST STRONGLY PREDICT THE NEWS IS REAL AND FAKE
	top_10_present = {"real": {}, "fake": {}}
	for classification in top_10_present:
		for word in P_classes:
			if no_stopwords and word in ENGLISH_STOP_WORDS: continue
			if len(top_10_present[classification].keys()) < 10:
				top_10_present[classification][word] = P_classes[word][classification]
				continue
			if classification in P_classes[word]: 
				if P_classes[word][classification] > min(top_10_present[classification].values()):
					word_of_min_prob = min(top_10_present[classification], key=top_10_present[classification].get)
					top_10_present[classification].pop(word_of_min_prob)
					top_10_present[classification][word] = P_classes[word][classification]

	# 10 WORDS WHOSE ABSENCE MOST STRONGLY PREDICT THE NEWS IS REAL AND FAKE
	top_10_absent = {"real": {}, "fake": {}}
	for classification in top_10_absent:
		for word in P_classes_negated:
			if no_stopwords and word in ENGLISH_STOP_WORDS: continue
			if len(top_10_absent[classification].keys()) < 10:
				top_10_absent[classification][word] = P_classes_negated[word][classification]
				continue
			if classification in P_classes_negated[word]: 
				if P_classes_negated[word][classification] > min(top_10_absent[classification].values()):
					word_of_min_prob = min(top_10_absent[classification], key=top_10_absent[classification].get)
					top_10_absent[classification].pop(word_of_min_prob)
					top_10_absent[classification][word] = P_classes_negated[word][classification]

	return top_10_present, top_10_absent

def part2():

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

	# OBTAIN OPTIMIZED NAIVE BAYES PARAMETERS
	m, p = get_optimal_parameters(x_train,y_train,x_val,y_val)
	# m, p = 750, 0.00005 # what is obtained by running the parameter optimzation function

	# TRAIN (i.e., BUILD THE DISCRETE PROBABILITY DISTRIBUTION)
	P = create_prob_dist(x_train,y_train,m,p)

	# GET THE ACCURACY OF THE NAIVE BAYES MODEL
	train_accuracy = test_naive_bayes(P,x_train,y_train)
	val_accuracy = test_naive_bayes(P,x_val,y_val)
	test_accuracy = test_naive_bayes(P,x_test,y_test)
	print("The training set accuracy is {}%, achieved with parameters (m = {}, p = {}).".format(train_accuracy,m,p))
	print("The validation set accuracy is {}%, achieved with parameters (m = {}, p = {}).".format(val_accuracy,m,p))
	print("The test set accuracy is {}%, achieved with parameters (m = {}, p = {}).".format(test_accuracy,m,p))

	# GET THE PRIOR PROBABILITY DISTRIBUTION DICTIONARY
	P_priors = get_prior_probabilities(y_train)

	# GET THE CONDITIONAL PROBABILITY DISTRIBUTION DICTIONARY CONTAINING P(c|w) FOR ALL WORDS w
	P_classes = create_class_prob_dict(P,P_priors,m,p)

	# GET THE CONDITIONAL PROBABILITY DISTRIBUTION DICTIONARY CONTAINING P(c|not w) FOR ALL WORDS w
	P_classes_negated = create_negated_class_prob_dict(P,P_priors,m,p)
	return P_classes, P_classes_negated

def part3():

	# GET THE CONDITIONAL PROBABILITY DICTIONARIES (P(c|w) and P(c|not w)) FROM PART2
	P_classes, P_classes_negated = part2()
	
	# UNCOMMENT TO SEE TOP 10 WORDS DISPLAYED FOR EACH SUBSECTION

	# print("____________________ TOP 10 WITH STOPWORDS ____________________")
	# # GET THE TOP 10 WORDS WITH MOST INFLUENCE WHEN PRESENT AND ABSENT IN CLASSIFYING HEADLINES
	# top_10_present, top_10_absent = get_top10s(P_classes,P_classes_negated)
	# print("The top 10 words whcose presence most strongly predict the news is real are:")
	# for word in top_10_present["real"]: print("{}: {}".format(word,top_10_present["real"][word]))
	# print("The top 10 words whose presence most strongly predict the news is fake are:")
	# for word in top_10_present["fake"]: print("{}: {}".format(word,top_10_present["fake"][word]))
	# print("The top 10 words whose absence most strongly predict the news is real are:")
	# for word in top_10_absent["real"]: print("{}: {}".format(word,top_10_absent["real"][word]))
	# print("The top 10 words whose absence most strongly predict the news is fake are:")
	# for word in top_10_absent["fake"]: print("{}: {}".format(word,top_10_absent["fake"][word]))

	# print("____________________ TOP 10 WITHOUT STOPWORDS ____________________")
	# # GET THE TOP 10 WORDS WITH MOST INFLUENCE WHEN PRESENT AND ABSENT IN CLASSIFYING HEADLINES (WITHOUT STOPWORDS)
	# top_10_present, top_10_absent = get_top10s(P_classes,P_classes_negated,no_stopwords=True)
	# print("The top 10 words (excluding stopwords) whose presence most strongly predict the news is real are:")
	# for word in top_10_present["real"]: print("{}: {}".format(word,top_10_present["real"][word]))
	# print("The top 10 words (excluding stopwords) whose presence most strongly predict the news is fake are:")
	# for word in top_10_present["fake"]: print("{}: {}".format(word,top_10_present["fake"][word]))
	# print("The top 10 words (excluding stopwords) whose absence most strongly predict the news is real are:")
	# for word in top_10_absent["real"]: print("{}: {}".format(word,top_10_absent["real"][word]))
	# print("The top 10 words (excluding stopwords) whose absence most strongly predict the news is fake are:")
	# for word in top_10_absent["fake"]: print("{}: {}".format(word,top_10_absent["fake"][word]))

#________________________ RUN PART2 ________________________
if __name__ == "__main__":
	part2()
	part3()




