function grad = computeThetaGradient(xtrain,NET, ttrain, A)

% Calculates gradient with respect to the components of theta
% INPUT:
%   X       - object-feature matrix with size m x n
%   theta   - n dim. parameters vector from previous optimization step
%   A       - fixed precision matrix
%   y       - answers vector from {0,1}^m
% OUTPUT:
%   grad    - gradient vector
theta=netpak(NET)';
grad = mlpgrad(NET, xtrain, ttrain)' + A*theta ;

end
