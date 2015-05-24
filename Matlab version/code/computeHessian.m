function H = computeHessian(xtrain,ttrain,net, A)

% Calculates hessian with respect to the components of theta
% INPUT:
%   X       - object-feature matrix with size m x n
%   thetaMP - n dim. parameters vector from previous optimization step
%   A       - fixed precision matrix
% OUTPUT:
%   H       - second derivatives matrix (hessian) 
H=mlphess(net,xtrain,ttrain);

H=H(a:b);
end
