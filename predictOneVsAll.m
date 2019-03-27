function p = predictOneVsAll(all_theta, X)

%   PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%   are in the range 1..K, where K = size(all_theta, 1). 
%   p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%   for each example in the matrix X. Note that X contains the examples in
%   rows. all_theta is a matrix where the i-th row is a trained logistic
%   regression theta vector for the i-th class.


%=========== set variables ==========

m = size(X, 1);
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

%=========== predictions =============

%returns a matrix (m x num_labels) with each column the probability that the case is the number corresponding to the index (and 0 if index = 10)
probabilities = sigmoid(X * all_theta');

%for each row, select highest probability and store it's index in p
[val,idx] = max(probabilities,[],2) ;
p = idx;

end





