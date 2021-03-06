import numpy as np

def f_sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return f_sigmoid(X)*(1 - f_sigmoid(X))
 
def f_softmax(X):
    if X.ndim == 2:
        Z = np.sum(np.exp(X), axis=1)
        Z = Z.reshape(Z.shape[0], 1)
    else:
        Z = np.sum(np.exp(X))
    return np.exp(X) / Z

def create_minibatches(data_matrix, target_matrix, size, one_hot, hot_size):
    count_batches = int(np.shape(data_matrix)[0] / size)
    if not one_hot:
        return (np.array_split(data_matrix, count_batches), np.array_split(target_matrix, count_batches))
    else:
        ohc_matrix = np.zeros((target_matrix.shape[0], hot_size))
        for i in range(0, target_matrix.shape[0]):
            current_vec = np.zeros(hot_size)
            current_vec[target_matrix[i]] = 1
            ohc_matrix[i] = current_vec
        return (np.array_split(data_matrix, count_batches), np.array_split(ohc_matrix, count_batches))

class Layer:
    def __init__(self, size, minibatch_size, is_input = False, is_output = False, activation = f_sigmoid):
        self.is_input = is_input
        self.is_output = is_output
        self.output_matrix = np.zeros((minibatch_size, size[0]))
        self.activation = activation
        self.weight_matrix = None
        self.input_matrix = None
        self.delta_matrix = None
        self.activation_derivative = None
        
        if not is_input:
            self.input_matrix = np.zeros((minibatch_size, size[0]))
            self.delta_matrix = np.zeros((minibatch_size, size[0]))
        
        if not is_output:
            self.weight_matrix = np.random.normal(size = size, scale = 1E-4)
            
        if not is_input and not is_output:
            self.activation_derivative = np.zeros((size[0], minibatch_size))
    
    def forward_propagate(self):
        if self.is_input:
            return self.output_matrix.dot(self.weight_matrix)
        
        self.output_matrix = self.activation(self.input_matrix)
        
        if self.is_output:
            return self.output_matrix
        else:
            # Append psuedo output = 1 for the bias
            self.output_matrix = np.append(self.output_matrix, np.ones((self.output_matrix.shape[0], 1)), axis = 1)
            self.activation_derivative = self.activation(self.input_matrix, deriv = True).T
            return self.output_matrix.dot(self.weight_matrix)
        
    def forward_vector(self):
        if self.is_input:
            return self.output_vector.dot(self.weight_matrix)
        self.output_vector = self.activation(self.input_vector)
        
        if self.is_output:
            return self.output_vector
        else:
            self.output_vector = np.append(self.output_vector, 1)
            return self.output_vector.dot(self.weight_matrix)

class MLP:
    def __init__(self, layer_config, minibatch_size):
        self.layers = []
        self.num_layers = len(layer_config)
        self.minibatch_size = minibatch_size
        
        for i in range(0, self.num_layers - 1):
            if i == 0:
                print("Initializing input layer with size {0}".format(layer_config[i]))
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]], minibatch_size, is_input = True))
            else:
                print("Initializing hidden layer with size {0}.".format(layer_config[i]))
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         minibatch_size,
                                         activation=f_sigmoid))
        print("Initializing output layer with size {0}.".format(layer_config[-1]))
        self.layers.append(Layer([layer_config[-1], None],
                                 minibatch_size,
                                 is_output=True,
                                 activation=f_softmax))
        print("Done!")
        
    def forward_propagate(self, data):
        self.layers[0].output_matrix = np.append(data, np.ones((data.shape[0], 1)), axis=1)

        for i in range(self.num_layers-1):
            self.layers[i+1].input_matrix = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()
    
    def backpropagate(self, yhat, labels):
        self.layers[-1].delta_matrix = (yhat - labels).T
        for i in range(self.num_layers - 2, 0, -1):
            weight_nobias = self.layers[i].weight_matrix[0:-1, :]
            self.layers[i].delta_matrix = weight_nobias.dot(self.layers[i+1].delta_matrix) * self.layers[i].activation_derivative
            
    def update_weights(self, eta):
        for i in range(0, self.num_layers - 1):
            weight_gradient = -eta*(self.layers[i+1].delta_matrix.dot(self.layers[i].output_matrix)).T
            self.layers[i].weight_matrix += weight_gradient
    
    def get_output(self, input):
        self.layers[0].output_vector = np.append(input, 1)
        for i in range(self.num_layers - 1):
            self.layers[i+1].input_vector = self.layers[i].forward_vector()
        return self.layers[-1].forward_vector()
    
    def evaluate(self, train_data, train_labels, test_data, test_labels,
                 num_epochs = 100, eta = 0.05, eval_train = False, eval_test = True):
        N_train = len(train_labels)*len(train_labels[0])
        N_test = len(test_labels)*len(test_labels[0])

        print("Training for {0} epochs...".format(num_epochs))
        for t in range(0, num_epochs):
            out_str = "[{0:4d}] ".format(t)

            for b_data, b_labels in zip(train_data, train_labels):
                output = self.forward_propagate(b_data)
                self.backpropagate(output, b_labels)
                self.update_weights(eta=eta)

            if eval_train:
                errs = 0
                for b_data, b_labels in zip(train_data, train_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = "{0} Training error: {1:.5f}".format(out_str,
                                                           float(errs)/N_train)

            if eval_test:
                errs = 0
                for b_data, b_labels in zip(test_data, test_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = "{0} Test error: {1:.5f}".format(out_str,
                                                       float(errs)/N_test)

            print(out_str)