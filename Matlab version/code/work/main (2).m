% Script illustrates covariance matrix estimation algorithm
% Date: April 2015

enableEstimationDemo = false;
enableRedundantFeatureDemo = true;

if enableEstimationDemo

m = 1000;
n = 3;

%
% Estimation demo 
%

A_type = 'scalar';

% generate object-feature matrix
X = ones(m, n + 1);
for i = 2:n+1
    r = rand();
    if r < 1/3
        X(:, i) = rand(m, 1);          % U[0, 1]
    elseif (r > 1/3) && (r < 2/3)
        X(:, i) = randn(m, 1);         % N(0, 1)
    else
        X(:, i) = -1 + 2 * rand(m, 1); % U[-1, 1]
    end
end

% set covariance matrix diagonal and generate parameters
if strcmp(A_type, 'scalar')
    alpha = ones(n + 1, 1) * 3;
    A = diag(alpha);
else
    alpha = 1 + 9 * rand(n + 1, 1); % ~ U[1, 10]
    A = diag(alpha);
end

theta = mvnrnd(zeros(numel(alpha), 1), inv(A))';
p = sigmoid(X, theta);
y = (rand(m, 1) < p) + 0.1 - 0.1;

[alphaEstimated, thetaMP] = estimateCovarianceLaplace(X, y, A_type, ...
                            'roundsNum', 10);

% Print results
if strcmp(A_type, 'scalar')                      
    abs(alpha(1) - alphaEstimated(1)) / abs(alpha(1))
else
    norm(alpha - alphaEstimated) / norm(alpha)
    max(abs(alpha - alphaEstimated))
end
norm(theta - thetaMP)
max(abs(theta - thetaMP))

end

if enableRedundantFeatureDemo

m = 1000;
n = 3;

%
% Redundant feature demo 
%

X = rand(m, n) - 0.5;

% set covariance matrix diagonal and generate parameters
alpha = [ones(n - 1, 1) / 4; 1e10];
A = diag(alpha);
theta = mvnrnd(zeros(numel(alpha), 1), inv(A))';

% generate answers
p = sigmoid(X, theta);
y = (rand(m, 1) < p) + 0.1 - 0.1;

[alphaEstimated, thetaMP] = estimateCovarianceLaplace(X, y, 'diag', ...,
                                                      'roundsNum', 15);
alphaEstimated
thetaMP

end
