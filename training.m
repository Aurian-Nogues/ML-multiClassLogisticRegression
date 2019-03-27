% Initialization
clear ; close all; clc

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data.mat'); % training data stored in arrays X, y
m = size(X, 1);
num_labels = 10;

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

% ============ One-vs-All Training ============

fprintf('Training One-vs-All Logistic Regression ... \n');

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

% ============= assess accuracy of training model ==================

fprintf('Assessing accuracy of training ... \n');

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);






