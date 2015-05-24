function R = computeWeightingMatrix(X, theta)

% Function computes weighting matrix with diagonal y_i*(1 - y_i)
% INPUT:
%   X       - object-feature matrix with size m x n
%   theta   - parameters vector with size n
% OUTPUT:
%   R       - m x m dimensional diagonal matrix with elements y_i*(1-y_i)

sigma = sigmoid(X, theta);
R = diag(sigma .* (1 - sigma));

end