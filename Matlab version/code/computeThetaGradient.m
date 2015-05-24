function grad = computeThetaGradient(xtrain, net, ttrain, A)

% Calculates gradient with respect to the components of theta
% INPUT:
%   X       - object-feature matrix with size m x n
%   theta   - n dim. parameters vector from previous optimization step
%   A       - fixed precision matrix
%   y       - answers vector from {0,1}^m
% OUTPUT:
%   grad    - gradient vector
w=netpack_nobias(net);
theta=w(net.nin*net.nhidden+1:end);
prediction = mlpfwd(NET,xtrain);
X=tanh(net.w1*xtrain);
grad = X'* sum((prediction - ttrain),2) + A * theta;

end
