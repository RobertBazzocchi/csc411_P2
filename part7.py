
##########################################
#				 PART 7				 #
##########################################
from sklearn import tree
from part4 import get_sets

sets = get_sets()
x_train = sets[0]
y_train = sets[1]
x_val = sets[2]
y_val = sets[3]
x_test = sets[4]
y_test = sets[5]
column_list = sets[6]	

print(x_train.shape)


X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)