function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_array = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_array = [0.01;0.03;0.1;0.3;1;3;10;30];
errors = zeros(size(C_array),size(sigma_array));

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

%training process
for i = 1:size(C_array)
  for j = 1:size(sigma_array)
    sigma_tmp = sigma_array(j);
    model= svmTrain(X, y, C_array(i,1), @(x1, x2) gaussianKernel(x1, x2, sigma_tmp));
    predictions = svmPredict(model,Xval);
    error = mean(double(predictions ~= yval));
    errors(i,j) = error; %i,j not i*j
  end
end
[min_err,idx] = min(errors(:));
[row,col] = ind2sub(size(errors),idx);
fprintf("min_row:%f\n", row);
fprintf("min_col:%f\n", col);
C = C_array(row);
sigma = sigma_array(col);

%result
%C = 1
%sigma = 0.1

%tried:
%1. data type error: if the variable cannot handle double
%2. sigma_array(j) cannot be passed into gaussianKernel
% =========================================================================

end
