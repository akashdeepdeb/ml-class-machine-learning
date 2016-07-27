% I used this code to generate a good-looking mesh plot

data = load('ex1data1.txt');
y = data(:,2);
m = length(y);
X = [ones(m,1), data(:,1)]; %design matrix
theta0 = -10:0.1:10;        %theta0 array to sample cost function values
theta1 = -10:0.1:10;        %theta1 array to sample cost function values
Jv = zeros(length(theta0), length(theta1));     %cost function matrix

%code for creating Jv (cost function) sampled by various (theta0, theta1) values
for i = 1:length(theta0),
  for j = 1:length(theta1),
    t = [theta0(i); theta1(j)];
    Jv(i,j) = computeCost(X,y,t);
  end;
end;

%create mesh plot
mesh(Jv);

%plot descriptions
xlabel('theta0'); ylabel('theta1'); zlabel('cost function J');
title('Surface plot of cost function J');

print -dpng 'costFuncPlot.png';
