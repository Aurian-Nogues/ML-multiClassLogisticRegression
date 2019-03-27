function g = sigmoid(z)
%returns the sigmoid value or logistic regression

g = zeros(size(z));
g = 1./(1+exp(-z));

end

%==================== test cases ===================

% sigmoid(-5)
%ans =  0.0066929

% sigmoid(0)
%ans =  0.50000

% sigmoid(5)
%ans =  0.99331

% sigmoid([4 5 6])
%ans =

 %  0.98201   0.99331   0.99753

% sigmoid([-1;0;1])
%ans =

 %  0.26894
 %  0.50000
 %  0.73106

 %=====================================================