##########################################
#				 PART 2 				 #
##########################################
import time
from part1 import *
from part2and3 import *
from math import log as log
from math import exp as exp
import torch
import numpy as np
import os.path
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def make_column_numbers(datasets):
	word_index = []

	for dataset in datasets:
		for line in dataset:
			word_list = line.split()
			for word in word_list:
				word = word.replace('\n', '')

				if word in word_index:
					continue
				else:
					word_index.append(word)
	return word_index


def make_x_set(x_set, column_list, type):
	data = np.zeros((1,(len(column_list)+1)))
	data[-1][-1] = 1
	
	num = 0
	for line in x_set:
		if num%100 == 0:
			print(str(int(num/len(x_set)*100)) + '%' + ' complete')


		word_list = line.split()
		for word in word_list:
			index = column_list.index(word)
			data[-1][index] = 1

		
		data = np.vstack((data,np.zeros((1,len(column_list)+1))))
		data[-1][-1] = 1

		num += 1

	data = data[:-1]
	np.save('x_' + type, data)
	return data

def make_y_set(y_set, type):
	data = np.zeros((1,1))

	for line in y_set:
		if line == 'real':
			data[-1][0] = 1
		data = np.vstack((data,np.zeros((1,1))))

	data = data[:-1]
	np.save('y_' + type, data)
	return data


def get_sets():
	x_train, y_train, x_val, y_val, x_test, y_test = get_datasets()
	column_list = make_column_numbers([x_train, x_val, x_test])

	if os.path.isfile('x_train.npy'):
		print('Loading x_train from memory')
		x_train = np.load('x_train.npy')
	else:
		print('Building x_train from data')
		x_train = make_x_set(x_train, column_list, 'train')

	if os.path.isfile('x_val.npy'):
		print('Loading x_val from memory')
		x_val = np.load('x_val.npy')
	else:
		print('Building x_val from data')
		x_val = make_x_set(x_val, column_list, 'val')

	if os.path.isfile('x_test.npy'):
		print('Loading x_test from memory')
		x_test = np.load('x_test.npy')
	else:
		print('Building x_test from data')
		x_test = make_x_set(x_test, column_list, 'test')

	if os.path.isfile('y_train.npy'):
		print('Loading y_train from memory')
		y_train = np.load('y_train.npy')
	else:
		print('Building y_train from data')
		y_train = make_y_set(y_train, 'train')

	if os.path.isfile('y_val.npy'):
		print('Loading y_val from memory')
		y_val = np.load('y_val.npy')
	else:
		print('Building y_val from data')
		y_val = make_y_set(y_val, 'val')

	if os.path.isfile('y_test.npy'):
		print('Loading y_test from memory')
		y_test = np.load('y_test.npy')
	else:
		print('Building y_test from data')
		y_test = make_y_set(y_test, 'test')

	return [x_train, y_train, x_val, y_val, x_test, y_test]



class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(5833, 1)  #5832 words + a bias

    def forward(self, x):
        """
        Using the sigmoid function
        """
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

def test_model(model, x_set, y_set):
   
    y_pred = model(x_set).data.numpy()
    y_set = y_set.data.numpy()
    correct = 0
    total = 0

    #print(np.argmax(y_pred, 1))
    # LOOK AT THE PERFORMANCE
    i = 0
    for n in y_pred:
    	if n[0] > 0.5:

    		if y_set[i] > 0.5:
    			correct += 1

    	else:
    		if y_set[i] <= 0.5:
    			correct += 1

    	i += 1

    return correct/i



def part4():
	#print(torch.__version__)
	#x_train, y_train, x_val, y_val, x_test, y_test = get_datasets()
	#column_list = make_column_numbers([x_train, x_val, x_test])
	
	#x_train = make_x_set(x_train, column_list, 'train')
	#y_train = make_y_set(y_train,'train')
	#print(y_train.shape)

	sets = get_sets()
	x_train = sets[0]
	y_train = sets[1]
	x_val = sets[2]
	y_val = sets[3]	

	x_train = Variable(torch.from_numpy(x_train)).float()
	y_train = Variable(torch.from_numpy(y_train)).float()
	x_val = Variable(torch.from_numpy(x_val)).float()
	y_val = Variable(torch.from_numpy(y_val)).float()


	model = Model()


	# Construct our loss function and an Optimizer. The call to model.parameters()
	# in the SGD constructor will contain the learnable parameters of the two
	# nn.Linear modules which are members of the model.
	criterion = torch.nn.BCELoss(size_average=True)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

	# Training loop
	for epoch in range(10000):
    		y_pred = model(x_train)
    		# Compute and print loss
    		loss = criterion(y_pred, y_train)
    		#print(epoch, loss.data[0])

    		# Zero gradients, perform a backward pass, and update the weights.
    		optimizer.zero_grad()
    		loss.backward()
    		optimizer.step()
			#print('train')
    		n = test_model(model,x_train,y_train)
    		print(n)

    		
    
	



part4()