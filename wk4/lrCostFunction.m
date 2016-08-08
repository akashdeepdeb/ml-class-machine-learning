function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)

%newTheta, newX are required for the regularized form since the bias unit (x_0) is not used in gradient descent and cost computation

n = size(theta);

%modified theta starting from 2nd feature [size: n x 1]
newTheta = theta(2:n);

regularizationTerm = lambda*newTheta'*newTheta/(2*m);

% n = 400 since X (the design matrix) = 5000 x 400
prediction = sigmoid(X * theta);

%log gives the accuracy; and this helps calculating the cost function
J = -(y' * log(prediction) + (1-y)' * log(1 - prediction))/m + regularizationTerm;

%because x_j = 1 (when j=0)
grad0 = sum(prediction-y)/m;

%newX is a design matrix without first feature where j=0
newX = X(:,2:n);

%gradient for gradient descent.
grad = (newX' * (prediction - y))/m + lambda*newTheta/m;
        
%grad adjusted with first grad and the regularized gradient
grad = [grad0; grad];

% =============================================================

grad = grad(:);

end
