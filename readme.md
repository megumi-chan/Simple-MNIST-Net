Usage
-------------
Run draw.py, then click train. After the training has finished, close popup, draw a number on the white zone of the window, then click Go to see the result. Click Clear to clear the draw zone, or click Show to display 28x28 pixels image corresponding to the drawn number.

Note :D
-------------------
- It makes many errors. 

Backpropagation Review
-------------------

For each layer: 

+ input_matrix: batch_size x current_layer_size ( For the first layer, its equal to the dimension of the input, 784)
+ delta_matrix:  current_layer_size x batch_size
+ output_matrix: batch_size x current_layer_size
+ weight_matrix: (current_layer_size + 1) x (next_layer_size) ( + 1 bias row )
+ activation_derivative :  current_layer_size x batch_size

Forward Propagate:
- For each layer:

- Calculate output_matrix = activation(input_matrix)
- Append bias column to output_matrix
- Calculate activation_derivative matrix = diff(activation(input_matrix)) transpose 
- Assign next layer input_matrix = current layer output_matrix dot weight_matrix

Return the last layer output_matrix

Backward Propagate:

- Calculate last layer delta_matrix = (calculated_labels - real_labels) transpose ( note that labels are in vector encoding )
- Remove bias row in the weight_matrix
- Calculate previous layers delta_matrix = current weight_matrix dot next layer delta_matrix then element-wise product with current activation_derivative matrix

Update Weight:
-For all layers:

- weight_gradient = learning_rate * (next layer delta_matrix dot current output_matrix) transpose
- weight_matrix -= weight_gradient

To Do
--------------
Create a convolutional neural network to do the task


