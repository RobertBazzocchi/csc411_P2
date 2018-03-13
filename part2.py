
##########################################
#				 PART 2 				 #
##########################################
import time

def get_count(word,txt_file):
	"""
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

def get_word_set(txt_file):
	"""
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

def naive_bayes():
	word_set = get_word_set("clean_real.txt")
	tot_count = 0
	for word in word_set:
		count = get_count(word,"clean_real.txt")
		tot_count += count

	

#________________________ RUN PART2 ________________________
naive_bayes()