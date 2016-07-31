function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

n = size(theta);

%modified theta starting from 2nd feature [size: n x 1]
newTheta = theta(2:n);

%size: m x 1 vector (z) for all m
z = X*theta;
%newZ = X*newTheta;

%use z to generate hypothesis h(x) = g(z) (aka sigmoid);
hyp = sigmoid(z);
%newHyp = sigmoid(newZ);

%vectorized cost function for logistic regression using regularization
J = (-y'*log(hyp) - (1-y')*log(1-hyp))/m + lambda*newTheta'*newTheta/(2*m);

%because x_j = 1 (when j=0)
grad0 = sum(hyp-y)/m;

%newX is a design matrix without first feature where j=0
newX = X(:,2:n);

%results in (n+1) vector
grad = (newX' * (hyp-y))/m + lambda*newTheta/m;

%grad adjusted with first grad and the regularized gradient
grad = [grad0; grad];


% =============================================================

end
