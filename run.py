#train and test the network:

import mnist_loader
import rileynet

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = rileynet.Net([784, 30, 10])

#run the network:
net.sgd(training_data, 30, 10, 1.0, test_data=test_data) #optimal alpha is between 1.0 and 3.0