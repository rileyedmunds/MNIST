import numpy as np
import random


class Net(object):

    def __init__(self, sizes):
        """constructor to define the dimensions are initial weights and biases for the net"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def sgd(self, training_data, epochs, mini_batch_size, alpha, test_data=None):
        # note that once this is called, mnist_loader has already loaded in the data for the network:
        # -training data, test_data and validation_data
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        # for each epoch, update the weights and biases accoding to the ouput from backpropagation
        for j in xrange(epochs):
            # shuffle training data to draw random sample (for mini-batch)
            random.shuffle(training_data)
            # break up into mini-batches
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            # draw that random mini-batch
            for mini_batch in mini_batches:
                # for that one minibatch we just chose, update the mini batch by calling update_mini_batch
                self.update_mini_batch(mini_batch, alpha)
            # print the ouput data to track progress
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, alpha):
        """Update whole network's weigths & biases with
         SGD using backpropagation on one mini batch"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [b + addb for b, addb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [w + addw for w, addw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (alpha / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (alpha / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
    	"""return the gradient from the cost function"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        ###backpropagation steps:
        """ 1) Compute activations for the current network (feedforward)
            2) Compute errors for the current network by back-propagating from the output later
            3) Compute gradients for each neuron via Chain rule on each neuron (back-propagating again)
            4) Update the weights on all neurons using SGD with the gradients computed with back-propagation"""
        #feedforward:
        activation = x
        activations = [x] #list to store all the activations
        zs = [] #stores all the activation ('z') vectors (current neuron formulae)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward error-calculation pass:
            #for the last layer:
        """delta is the error in the j-th later"""
        delta = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #moving back across the rest:
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def feedforward(self, a):
        """returns the output from the network if the input is 'a' """
        for b, w in zip(self.biases, self.weights):
            #note we update a:
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        """yields the number of inputs for which the NN gives the correct outcome"""
        outcomes = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in outcomes)

    def cost_prime(self, output_activations, y):
        """return partial a C_x vector/
        For squared error, this is the derivative of RMSE:
         squared term and 1/2 cancel each other out
        """

        return (output_activations - y)


#### Utility functions:
def sigmoid(z):
    """used in feedforward to get the outcome for some input and backprop to calculate the descent"""
    d = (1.0 + np.exp(-z))
    return 1.0 / d

def sigmoid_prime(z):
    """sigmoid devirative
        usage: backprop()"""
    n = 1 - sigmoid(z)
    return sigmoid(z) * n