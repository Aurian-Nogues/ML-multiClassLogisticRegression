function [all_theta] = oneVsAll(X, y, num_labels, lambda)

%   Trains multiple logistic regression classifiers and returns all
%   the classifiers in a matrix all_theta, where the i-th row of all_theta 
%   corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% set some variables

m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);

%add intercept column to X matrix
X = [ones(m, 1) X];

% ========== training ============

%set initial theta matrix and fmincg function parameters
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for i = 1:num_labels

    %c is the number we train the logistic regression for, 10 represents 0
    c = i

    %get theta vector that minimises cost function  for number i        
    [theta] = ...
        fmincg (@(t)(costFunction(t, X, (y == c), lambda)), ...
        initial_theta, options);

    %export theta vector to all_theta matrix
    all_theta(i, :) = theta';
end

