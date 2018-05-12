import os
import numpy as np
from utils import *

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
	"""
	X is list of integers, where each integer is a number that maps to a character in the vocabulary.
	Y is list of integers, exactly the same as X but shifted one index to the left.
	"""

	##Forward propagate through time
	##rnn_forward performs forward pass on every time step and evaluates the losses
	loss, cache = rnn_forward(X, Y, a_prev, parameters)

	##Backpropagate through time
	gradients, a = rnn_backward(X, Y, parameters, cache)

	##Clip your gradients between -5 (min) and 5 (max)
	gradients = clip(gradients, 5)

	##Update parameters is normal updating of parameters
	parameters = update_parameters(parameters, gradients, learning_rate)

	return loss, gradients, a[len(X)-1]


##clipping exploding gradients
def clip(gradients, maxValue):
	
	dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

	##clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby].
	for gradient in [dWax, dWaa, dWya, db, dby]:
		np.clip(gradient, -10, 10, out = gradient)

	gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

	return gradients

def sample(parameters, char_to_ix, seed):
	"""
	Sample a sequence of characters according to a sequence of probability distributions output of the RNN

	Returns:
	indices - a list of length n containing the indices of the sampled characters.
	"""

	##Retrieve parameters and relevant shapes from "parameters" dictionary
	Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
	vocab_size = by.shape[0]
	n_a = Waa.shape[1]

	##Create the one-hot vector x for the first character (initializing the sequence generation).
	x = np.zeros((vocab_size, 1))
	##Initialize a_prev as zeros
	a_prev = np.zeros((n_a, 1))

	##Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate
	indices = []

	##Idx is a flag to detect a newline character, we initialize it to -1
	idx = -1 

	##Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
	##its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well 
	##trained model), which helps debugging and prevents entering an infinite loop. 
	counter = 0
	newline_character = char_to_ix['\n']

	while (idx != newline_character and counter != 50):

		##Forward propagate x using the equations
		a = np.tanh(np.matmul(Wax, x) + np.matmul(Waa, a_prev) + b)
		z = np.matmul(Wya, a) + by
		y = softmax(z)

		np.random.seed(counter + seed) 

		##Sample the index of a character within the vocabulary from the probability distribution y
		##p is the probability associated with each entery in the provided array
		idx = np.random.choice(np.arange(vocab_size), p = y.ravel())

		##Append the index to "indices"
		indices.append(idx)

		##Overwrite the input character as the one corresponding to the sampled index.
		x = np.zeros((vocab_size, 1))
		x[idx] = 1

		##Update "a_prev" to be "a"
		a_prev = a

		seed += 1
		counter +=1


		if (counter == 50):
			indices.append(char_to_ix['\n'])

	return indices



def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
	"""
	Generates dinosaur names. 
	data - text corpus
	ix_to_char - dictionary that maps the index to a character
	char_to_ix - dictionary that maps a character to an index
	n_a - number of units of the RNN cell
	dino_names - number of dinosaur names you want to sample at each iteration. 
	vocab_size - number of unique characters found in the text, size of the vocabulary

	"""

	##Retrieve n_x and n_y from vocab_size
	n_x, n_y = vocab_size, vocab_size

	##Initialize parameters
	parameters = initialize_parameters(n_a, n_x, n_y)

	##Initialize loss (this is required because we want to smooth our loss)
	loss = get_initial_loss(vocab_size, dino_names)

	##Build list of all dinosaur names (training examples).
	with open("dinos.txt") as f:
		examples = f.readlines()
		examples = [x.lower().strip() for x in examples]

	##Shuffle list of all dinosaur names
	np.random.seed(0)
	np.random.shuffle(examples)

	##Initialize the hidden state of the RNN
	a_prev = np.zeros((n_a, 1))

	##Optimization loop
	for j in range(num_iterations):
		##Below line is written so that index is always less than number of examples 		
		index = j % len(examples)
		X = [None] + [char_to_ix[ch] for ch in examples[index]]
		Y = X[1 : ] + [char_to_ix["\n"]]

		##Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
		##Choose a learning rate of 0.01
		curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)

		##Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
		loss = smooth(loss, curr_loss)

		##Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
		if j % 2000 == 0:

			print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

			##The number of dinosaur names to print
			seed = 0
			for name in range(dino_names):
				##Sample indices and print them
				sampled_indices = sample(parameters, char_to_ix, seed)
				print_sample(sampled_indices, ix_to_char)

				seed += 1  # To get the same result for grading purposed, increment the seed by one. 
				print('\n')
		

	return parameters



##Main code starts from here

data = open('dinos.txt', 'r').read()

##Makes the upper case string lowercase
data= data.lower()

##using a set data structure we store only unique elements, below line finds unique characters in the whole of dataset
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))


char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

parameters = model(data, ix_to_char, char_to_ix)


