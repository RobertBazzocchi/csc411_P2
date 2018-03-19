
##########################################
#				 PART 1					 #
##########################################

def get_dataset_info(**keyword_counts):
	"""
	FUNCTION: 
		This function recieves a dictionary and outputs three things: (i) the number of lines in
		the real headlines dataset, (ii) the number of lines in the fake headlines dataset, and 
		(iii) a dictionary with words and their number of occurences in each of the datasets.

	INPUT:
		**keyword_counts (dict)	: a dictionary where the keys are the keywords to be counted
								  and the values are subdictionaries with values "real" or 
								  "fake" and values being the counts of the keyword in the
								  respective dataset
	OUTPUT:
		num_real_lines (int)	: number of lines in the real headlines dataset
		num_fake_lines (int)	: number of lines in the fake headlines dataset
		keywords_counts (dict)	: same dictionary as input but with completed count data
	"""
	num_real_lines, num_fake_lines = 0, 0

	with open("clean_real.txt") as f:
		for line in f:
			num_real_lines += 1
			for keyword in keyword_counts:
				keyword_counts[keyword]["real"] += line.count(keyword)

	with open("clean_fake.txt") as f:
		for line in f:
			num_fake_lines += 1
			for keyword in keyword_counts:
				keyword_counts[keyword]["fake"] += line.count(keyword)				

	# print("Number of times {} is in real news headlines: {}".format(keyword,count_real))
	# print("Number of times {} is in fake news headlines: {}".format(keyword,count_fake))

	return num_real_lines, num_fake_lines, keyword_counts

def partition_data(txt_file):
	"""
	FUNCTION:
		This function receives a .txt text file and partitions the data into a training, validation, 
		and test set containing only the headlines data (i.e., without classifications).
	INPUT:
		txt_file (.txt file)	: a text file of real or fake news headlines
	OUTPUT:
		train_set (list)		: a list of headlines
	"""
	tot_lines = sum(1 for line in open(txt_file)) # TOTAL NUMBER OF LINES IN txt_file

	train_set,val_set,test_set = [],[],[]
	train_not_full, val_not_full, test_not_full = True, True, True
	i = 1
	with open(txt_file) as f:
		for line in f:

			# FILL TRAINING SET
			if train_not_full:
				train_set.append(line)
				if i/float(tot_lines)> 0.7:
					train_not_full = False
					i = 0
				i+=1
				continue

			# FILL VALIDATION SET
			if val_not_full:
				val_set.append(line)
				if i/float(tot_lines) > 0.15:
					val_not_full = False
					i = 0
				i+=1
				continue

			# FILL TEST SET
			if test_not_full:
				test_set.append(line)
				if i/float(tot_lines) > 0.15:
					test_not_full = False
					i = 0
				i+=1
				continue

	return train_set, val_set, test_set

def get_datasets():
	"""
	FUNCTION:
		This function calls partition_data to obtain the training, validation, and test sets of the
		headlines to be used in training. It then creates their corresponding "classification" sets
		that classify each headline as either "fake" or "real".

	OUTPUT:
		train_set (list)		: a list of headlines containing ~70% of the total data
		val_set (list)			: a list of headlines containing ~15% of the total data
		test_set (list)			: a list of headlines containing ~15% of the total data

		train_class (list)		: a list of classifications corresponding to the train_set headlines
		val_class (list)		: a list of classifications corresponding to the val_set headlines
		test_class (list)		: a list of classifications corresponding to the test_set headlines
	"""
	train_real_lines, val_real_lines, test_real_lines = partition_data("clean_real.txt")
	train_class = ["real"]*len(train_real_lines)
	val_class = ["real"]*len(val_real_lines)
	test_class = ["real"]*len(test_real_lines)


	train_fake_lines, val_fake_lines, test_fake_lines = partition_data("clean_fake.txt")
	train_class.extend(["fake"]*len(train_fake_lines))
	val_class.extend(["fake"]*len(val_fake_lines))
	test_class.extend(["fake"]*len(test_fake_lines))

	train_set = train_real_lines + train_fake_lines
	val_set = val_real_lines + val_fake_lines
	test_set = test_real_lines + test_fake_lines

	return train_set, train_class, val_set, val_class, test_set, test_class

def part1():
	keywords = ["reason","hillary","new"]

	keyword_counts = {}

	# INITIALIZE THE KEYWORD COUNTS TO ZERO
	for keyword in keywords:
		keyword_counts[keyword] = {"real": 0, "fake": 0}

	# GET THE NUMBER OF REAL AND FAKE HEADLINES AS WELL AS THE KEYWORD COUNTS
	num_real_lines, num_fake_lines, keyword_counts = get_dataset_info(**keyword_counts)

	print("\n----------------------------------------------------------------")
	print("        	          Dataset Summary")
	print("----------------------------------------------------------------")

	print("There are two datasets provided for this project - a set of real news headlines and set of fake news headlines. The real news dataset and the fake news dataset consist of {} and {} headlines, respectively. Thus, the real news dataset contains nearly 700 more headlines than that of the fake news.\n".format(num_real_lines,num_fake_lines))
	print("It is difficult to predict the credibility of these news headlines simply by observation. However, there may be some keywords that indicate with a higher probability that a headline is fake or real.\n")

	print("For example, take the words, \"{}\", \"{}\", and \"{}\". These strings were counted in both the real news dataset and the fake news dataset, although the counts vary drastically. Thus, seeing one of these words in a news headlines may cause us to believe it is within the category that has the larger count.".format(keywords[0],keywords[1],keywords[2]))
	print("\nThese counts are summarized below:")

	for keyword in keyword_counts:
		print("___________________________")
		print("Word: {}".format(keyword))
		print("Occurences in Real News: {}".format(keyword_counts[keyword]["real"]))
		print("Occurences in Fake News: {}".format(keyword_counts[keyword]["fake"]))

#________________________ RUN PART1 ________________________
# part1()

# SET THE KEYWORDS WE WANT TO COUNT
# Some examples with the tuple: (real count, fake count)
# everyone:		(0, 5)
# hate:			(5, 13)
# sex:			(8, 16)
# hell:			(3, 12)
# bull:			(59, 5)



