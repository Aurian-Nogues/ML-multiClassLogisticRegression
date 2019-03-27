
function [J, grad] = costFunction(theta, X, y, lambda)
%computes cost J and gradient grad for regularized logistic regression

%=========initialize some value =====

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%========= compute ============

%h(x) = hypothesis_vector = g(theta'X) %'
hyp_vct = sigmoid(X * theta); %

temp_theta = theta;
temp_theta(1) = 0; % regularization sums theta square ignoring intercept theta (theta(1))

J = 1/m * sum(-y'*log(hyp_vct) - (1-y')*log(1-hyp_vct)) + (lambda/(2*m)) * sum((temp_theta.^2)) ;


% prodcut of vectors already takes care of summation so no need to add
grad = 1/m * X'*(hyp_vct - y) + (lambda/m) * (temp_theta); %'

grad = grad(:);

end


%========================= test cases =====================================

% theta = [-2; -1; 1; 2];
% X = [ones(5,1) reshape(1:15,5,3)/10];
% y = ([1;0;1;0;1] >= 0.5);
% lambda = 3;
% [J grad] = costFunction(theta, X, y, lambda);

% J = 2.534819
% grad =  0.146561 / -0.548558 / 0.724722 / 1.398003
