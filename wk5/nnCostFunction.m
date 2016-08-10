function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 3 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing m values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%------------------------------------
% This is the feedforward propogation
X = [ones(m,1), X]; % adding of bias units to all examples.
a_2 = sigmoid(X * Theta1'); %this is a m x 25 matrix in our case
a_2 = [ones(m,1), a_2]; %adding bias unit in the front (m x 26 matrix in our case)
hyp = sigmoid(a_2 * Theta2'); %this creates an m x 10 matrix giving us hyp for all m

%--------------------------------------
% This is the cost function computation
%For all i from 1 to m, we get a 10x1 matrix with hyp values,
%and we already have values from y.
J = 0;
              
for i = 1:m,
    yNN = zeros(num_labels, 1); %converting the numerical y, to vector form
    yNN(y(i)) = 1;
    hypNN = hyp(i,:); %this creates a 1 x 10 vector
    J += (log(hypNN)*yNN + log(1-hypNN)*(1-yNN)); %unregularized cost function
end

J = -J/m; %unregularized
              
%----------------------------------------------------------------------
%Regularizing to prevent overfitting of terms; good practice in general
regTheta1 = Theta1(:,2:end);
regTheta1 = regTheta1(:);
regTheta2 = Theta2(:,2:end);
regTheta2 = regTheta2(:);
              
J += lambda * (sum(regTheta1 .^ 2) + sum(regTheta2 .^ 2))/(2*m);

%-------------------------
%backpropagation algorithm
%I could have implemented this while computing the cost function
%but this is a cleaner, modular structure that looks neater

for i = 1:m,
    a1 = X(i,:)'; %401 x 1 vector
              
    z2 = Theta1 * a1; %25 x 1 vector
    a2 = sigmoid(z2); a2 = [1; a2]; %gives 26x1 vector including the bias unit
              
    z3 = Theta2 * a2; %10 x 1 vector
    a3 = hyp = sigmoid(z3); %gives 10x1 vector
              
    ynn = zeros(1, num_labels); %converting numerical y to 1x10 vector
    ynn(y(i)) = 1; %y at position i should be 1 and rest should be 0
              
    d3 = hyp - ynn'; %error in last layer
    newTheta1 = Theta1(:, 2:end); %25 x 400 matrix for regularization
    newTheta2 = Theta2(:, 2:end); %10 x 25 matrix ignore bias and for regularization
    d2 = newTheta2' * d3 .* sigmoidGradient(z2); %error in second layer
              
    Theta1_grad = Theta1_grad + d2 * a1';
    Theta2_grad = Theta2_grad + d3 * a2';
end

% implementing regularization
Theta1_grad = (Theta1_grad + lambda * [zeros(hidden_layer_size,1), Theta1(:, 2:end)])/m;
Theta2_grad = (Theta2_grad + lambda * [zeros(num_labels,1), Theta2(:, 2:end)])/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
