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

## Multiclass Classification - V0.4

In this code, I used a neural network to recognize ten handwritten digits, 0-9. This is a multiclass classification task where one of n choices is selected. Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks.

The neural network I used in this code is shown in the figure below.
![image](https://user-images.githubusercontent.com/115104812/198883700-5055a1cb-4c28-4266-a514-e42eadf5e0d8.png)

This has two dense layers with ReLU activations followed by an output layer with a linear activation.
Since the images are of size  20×20 , this gives us  400  input
The parameters have dimensions that are sized for a neural network with  25  units in layer 1,  15  units in layer 2 and  10  output units in layer 3, one for each digit

Tensorflow models are built layer by layer. A layer's input dimensions  are calculated for you. You specify a layer's output dimensions and this determines the next layer's input dimension. The input dimension of the first layer is derived from the size of the input data specified in the model.fit statement below.

defines a loss function, SparseCategoricalCrossentropy and indicates the softmax should be included with the loss calculation by adding from_logits=True)

## Advice for Applying Machine Learning - V0.5


It has been observed that as the rank gets too large, the cross validation performance begins to degrade relative to the training performance.


I trained the model over and over, increasing the degree of the polynomial with each iteration. Here, I used the scikit-learn linear regression model for speed and simplicity. My goal here is to see the difference between overfitting and underfitting.
As the model complexity increases, the error in the trained data (blue) decreases.
the error of cross validation data decreases initially when the model starts to fit the data but then increases as the model starts to overfit the training data (fails to generalize)

As the regularization (lambda) increased, I saw that the model shifted from a high variance (over-fitting) model to a high-bias (under-fitting) model.
I have found that when a model has high variance and is overfitting, adding more samples improves performance, but also when it has a high bias (underfit) adding more samples improves performance.

I did three model tutorials with tensorflow; <p>
      * complex model <p>
      * simple model <p>
      * regularization model 
  
The complex model worked hard to catch the outliers of each category. As a result, it miscategorized some of the cross-validation data.
simple model has a little higher classification error on training data but does better on cross-validation data than the more complex model.
The regularization model is slightly worse on the training set than the simple model, but better on the cross validation set.

As regularization is increased, the performance of the model on the training and cross-validation data sets converge.

 






