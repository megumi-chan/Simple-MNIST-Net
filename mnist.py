import gzip
import _pickle as cPickle
import neural_nets as nn
import numpy as np

mlp = None
def train():
    global train_data, train_labels
    with gzip.open('mnist.pkl.gz') as f:
        train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    minibatch_size = 100
    print("Creating data...")
    train_data, train_labels = nn.create_minibatches(train_set[0], train_set[1],
                                                  minibatch_size, one_hot = True, hot_size = 10)
    valid_data, valid_labels = nn.create_minibatches(valid_set[0], valid_set[1],
                                                  minibatch_size, one_hot = True, hot_size = 10)
    print("Done!")
    
    global mlp
    mlp = nn.MLP(layer_config=[784, 100, 100, 10], minibatch_size=minibatch_size)
    mlp.evaluate(train_data, train_labels, valid_data, valid_labels,
                 eval_train=True)
    
    
def recognize(mat):
    output = mlp.get_output(mat)
    return np.argmax(output)