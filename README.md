# Machine_Learning_Project

## Neural Networks for Handwritten Digit Recognition - V0.2

Tensorflow is a machine learning package developed by Google. In 2019, Google integrated Keras into Tensorflow and released Tensorflow 2.0. Keras is a framework developed independently by François Chollet that creates a simple, layer-centric interface to Tensorflow
Here, a neural network is used to recognize two handwritten digits, zero and one. Actually, this is a binary classification task. Automatic handwritten digit recognition is widely used today, from recognizing postal codes (zip codes) on postal envelopes to recognizing written amounts on bank checks.

The dataset contains 1000 training examples of the handwritten 1 digits, bounded here by zero and one. Each training example is a 20-pixel x 20-pixel grayscale image of the digit
Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
Each training example becomes a single row in our data matrix X.
This gives us a 1000 x 400 matrix X where every row is a training example of a handwritten digit image.
The second part of the training set is a 1000 x 1 dimensional vector y that contains labels for the training set
y = 0 if the image is of the digit 0, y = 1 if the image is of the digit 1.

![image](https://user-images.githubusercontent.com/115104812/197747233-589ba34c-81db-4172-92f0-1f25b3789e7c.png)

layer1: The shape of W1 is (400, 25) and the shape of b1 is (25,)
layer2: The shape of W2 is (25, 15) and the shape of b2 is: (15,)
layer3: The shape of W3 is (15, 1) and the shape of b3 is: (1,)

Note: The bias vector b could be represented as a 1-D (n,) or 2-D (1,n) array. Tensorflow utilizes a 1-D representation and this lab will maintain that convention.

Tensorflow models are built layer by layer. A layer's input dimensions are calculated for you. You specify a layer's output dimensions and this determines the next layer's input dimension. The input dimension of the first layer is derived from the size of the input data specified in the model.fit statement below.


The following code will define a loss function and run gradient descent to fit the weights of the model to the training data.

** model.compile( loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(0.001) )
** model.fit(X, y, epochs=20 )

Keras prediction was used to run the model on a sample to make a prediction. The input to be estimated is an array, so the single instance is reshaped to be two-dimensional.

The output of the model is interpreted as a probability. As in the case of logistic regression, the probability is compared with a threshold to make a final estimate.


## Multiclass - V0.3

This code use a 2-layer network as shown. Unlike the binary classification networks, this network has four outputs, one for each class. Given an input example, the output with the highest value is the predicted class of the input.

inside is an example of how to construct this network in Tensorflow. Notice the output layer uses a linear rather than a softmax activation. While it is possible to include the softmax in the output layer, it is more numerically stable if linear outputs are passed to the loss function during training. If the model is used to predict probabilities, the softmax can be applied at that point.

he statements below compile and train the network. Setting from_logits=True as an argument to the loss function specifies that the output activation was linear rather than a softmax.
