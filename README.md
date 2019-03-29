#machine learningL multi class logistic regression

This is a machine learning algorithm for Octave / Matlab to read hand written numbers. 

sigmoid.m takes X * theta as an input and returns the probability that the input should be classified as 1

displayData.m selects 100 random numbers and displays them

costFunction.m takes as an input:
    - theta vector
    - X features matrix
    - y results vector
    - lambda regularization parameter
and returns:
    - J cost of using theta and lambda
    - grad the gradient of the cost function

oneVsAll.m trains the model for the 10 different classes and returns the corresponding theta matrix

predictOneVsAll.m predicts the X values using the algorithm and returns a vector containing the redictions

training.m is a script that loads the data and runs the different function to display the dataset, train the algorithm, predict the values of y using the algorithm to process X and compute the accuracy of the classifier by comparing the result to thr y vector.
