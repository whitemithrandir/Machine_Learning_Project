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

 






