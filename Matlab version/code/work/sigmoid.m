function sigma = sigmoid(X, theta)

% Calculates sigmoid function
% INPUT:
%   X       - object-feature matrix with size m x n
%   theta   - parameters vector with size n
% OUTPUT:
%   sigma   - m-dimensional vector, each element is sigma(X(i:, ) * theta)

sigma = ones(size(X, 1), 1) ./ (1 + exp(-X * theta)); 

end
