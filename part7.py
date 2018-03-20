
##########################################
#				 PART 7				 #
##########################################
from sklearn import tree
from part4 import get_sets
import matplotlib.pyplot as plt

sets = get_sets()
x_train = sets[0]
y_train = sets[1]
x_val = sets[2]
y_val = sets[3]
x_test = sets[4]
y_test = sets[5]
column_list = sets[6]	

def vary_max_depth(X,Y,x_val,y_val):
	"""
	FUNCTION:
		This function takes as input the (X,Y) and (x_val, y_val) training sets, sweeps through 12
		values of max_depth to be used in the decision tree definition, and outputs the depth that
		yields the highest accuracy on the validation set.
	INPUT:
		X (list)			: a list of headlines containing ~70% of the total data
		Y (list)			: a list of classifications corresponding to the train_set headlines
		x_val (list)		: a list of headlines containing ~15% of the total data
		y_val (list)		: a list of classifications corresponding to the val_set headlines
	OUTPUT:
		best_depth (int)	: the depth value that yields the highest accuracy on the validation
	"""
	plot_info = [[],[]] # FIRST ITEM IS LIST OF DEPTHS, SECOND ELEMENT IS A LIST OF ACCURACIES
	best_accuracy = 0
	best_depth = None
	for depth in [50, 60, 70, 78, 79, 80, 81, 82, 83, 85, 88, 90, 100, 110, 120]:
		clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
		clf = clf.fit(X, Y)
		i = 0
		correct = 0
		for headline in x_val:
			y_pred = clf.predict([headline])
			if y_pred == y_val[i]:
				correct += 1
			i+=1
		accuracy = correct/i
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_depth = depth

		plot_info[0].append(depth)
		plot_info[1].append(accuracy)

	return best_depth, best_accuracy*100, plot_info

def plot_depth_vs_accuracy(plot_info):
	"""
	FUNCTION:
		This function takes as input plot info returned from vary_max_depth containing the history of
		depth values and accuracy values and plots this relationship on a graph.
	"""
	plt.title("Max_depth vs. Accuracy")
	plt.xlabel('max_depth')
	plt.ylabel('accuracy')
	plt.plot(plot_info[0], plot_info[1])
	plt.show()

def vary_max_depth_and_features(X,Y,x_val,y_val):
	"""
	FUNCTION:
		This function takes as input the (X,Y) and (x_val, y_val) training sets, sweeps through 10
		values of max_depth and 10 values of max_features to be used in the decision tree definition,
		and outputs the depth and feature that yields the highest accuracy on the validation set.
	INPUT:
		X (list)			: a list of headlines containing ~70% of the total data
		Y (list)			: a list of classifications corresponding to the train_set headlines
		x_val (list)		: a list of headlines containing ~15% of the total data
		y_val (list)		: a list of classifications corresponding to the val_set headlines
	OUTPUT:
		best_depth (int)	: the depth value that yields the highest accuracy on the validation
		best_feature (int)	: the feature value that yields the highest accuracy on the validation
	"""
	best_accuracy = 0
	best_feature = None
	best_depth = None
	for feature in [55, 60, 65, 68, 70, 73, 75, 78, 80, 90]:
		print("Processing feature: {}".format(feature))
		for depth in [70, 78, 79, 80, 81, 82, 83, 85, 88, 90]:
			clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=feature, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
			clf = clf.fit(X, Y)
			i = 0
			correct = 0
			for headline in x_val:
				y_pred = clf.predict([headline])
				if y_pred == y_val[i]:
					correct += 1
				i+=1
			accuracy = correct/i
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_feature = feature
				best_depth = depth

	return best_feature, best_depth, best_accuracy*100

def get_accuracy(X,Y,x_set,y_set,best_max_depth,best_max_features):
	"""
	FUNCTION: Outputs the accuracy of the decision tree built from the X,Y dataset pair when compared to
	the x_set, y_set data set pair.
	INPUT:
		X (list)			: a list of headlines containing ~70% of the total data
		Y (list)			: a list of classifications corresponding to the train_set headlines
		x_val (list)		: a list of headlines containing ~15% of the total data
		y_val (list)		: a list of classifications corresponding to the val_set headlines
	"""
	best_accuracy = 0
	clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=best_max_depth, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=best_max_features, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
	clf = clf.fit(X, Y)

	i = 0
	correct = 0
	for headline in x_set:
		y_pred = clf.predict([headline])
		if y_pred == y_set[i]:
			correct += 1
		i+=1
	accuracy = correct/i
	if accuracy > best_accuracy:
		best_accuracy = accuracy

	return best_accuracy*100

def part7():

	# FIND BEST MAX_DEPTH PARAMETER
	# best_max_depth, best_accuracy, plot_info = vary_max_depth(x_train,y_train,x_val,y_val)
	# plot_depth_vs_accuracy(plot_info)
	# print("Best accuracy: {}%".format(best_accuracy))
	# print("Best depth: {}".format(best_max_depth))

	# FIND BEST MAX_FEATURES / MAX_DEPTH COMBINATION
	best_max_features, best_max_depth, best_accuracy = vary_max_depth_and_features(x_train,y_train,x_val,y_val)
	print("Best accuracy: {}%".format(best_accuracy))
	print("Best depth: {}".format(best_max_depth))
	print("Best feature: {}".format(best_max_features))

	# OBTAIN RESULTS FOR ALL SETS
	# train_accuracy = get_accuracy(x_train,y_train,x_train,y_train,best_max_depth,best_max_features)
	val_accuracy = get_accuracy(x_train,y_train,x_val,y_val,best_max_depth,best_max_features)
	# test_accuracy = get_accuracy(x_train,y_train,x_test,y_test,best_max_depth,best_max_features)
	# print("The training set accuracy is {}%.".format(train_accuracy))
	print("The validation set accuracy is {}%.".format(val_accuracy))
	# print("The test set accuracy is {}%.".format(test_accuracy))

def part8():
	# FROM PART7, WE HAVE BEST max_depth = 90, BEST max_features = 80
	clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy",
 	max_depth=90, max_features=80)
	clf_entropy.fit(x_train, y_train)

	tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')

if __name__ == "__main__":
	part7()
	part8()