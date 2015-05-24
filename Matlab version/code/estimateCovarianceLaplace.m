function [alpha, thetaMP] = estimateCovarianceLaplace(net,xtrain,ttrain, A_type, varargin)

% Function estimates precision matrix for parameters vector
% INPUT:
%   X          - object-feature matrix with m x n dimensions
%   y          - answers vector in {0, 1}^m
%   roundsNum  - rounds count for precision matrix estimation (default: 100)
%   thetaIterationsCount - iterations count to find thetaMP in each round
%   alphaIterationsCount - iterations count to find A diagonal with 
%       fixed thetaMP in each round
% OUTPUT:
%   alpha - estimated precision matrix diagonal

% default values for arguments
roundsNum = 30;
wIterNum = 30;
alphaIterNum = 1;
plotFlag = true;

procArgs(varargin);

% initialization

sz=net.nout*net.nhidden; %120
thetaMP = zeros(net.nhidden*net.nout,1); 
%alpha = 1e-10 * ones(size(X, 2), 1); 
alpha = ones(sz, 1); 
eps = 1e-10;
upper_bound = 1e10;

alphas = [alpha zeros(sz, roundsNum)];
a=(net.nin+1)*net.nhidden+1;
b=a+net.nhidden*(net.nout)-1;
for i = 1:roundsNum
    A = diag(alpha);
    
    for j = 1:wIterNum
        H = mlphess(net,xtrain,ttrain);
        H=H(a:b,a:b)+A;
        grad=mlpgrad(net,xtrain,ttrain)';
        grad=grad(a:b,1);
        thetaMP = thetaMP - (H + eps * eye(size(grad,1))) \ grad;
    end
    %thetaMP
    w=netpak(net)';
    w(a:b,1)=thetaMP;
    NETMP=netunpak(net,w);
    for j = 1:alphaIterNum
        alphaOld = alpha;
        H=mlphess(NETMP,xtrain,ttrain);
        H=H(a:b,a:b);
        sigma = inv(H+ A + eps * eye(size(grad,1)));
    
        for k = 1:size(grad,1)
            t = alphaOld(k) * sigma(k, k);
            if (t >= 1)
                alphaOld
                sigma
                thetaMP
            end
            alpha(k) = (1 - t) / (thetaMP(k) ^ 2);
        end
        alpha = min(alpha, upper_bound);
    end
    alphas(:, i + 1) = alpha;
    %alpha
end

if plotFlag
    if strcmp(A_type, 'scalar')
        plot(0:roundsNum, alphas(1, :));
        ylabel('$\mathbf{A}^{-1} = \alpha \mathbf{I}$', 'FontSize', 16, ...
               'FontName', 'Times', 'Interpreter','latex');
    else
        plot(0:roundsNum, alphas');
        ylabel('$\mathbf{A}^{-1} = \{\alpha_1, \dots, \alpha_n\}$', ...
               'FontSize', 16, 'FontName', 'Times', 'Interpreter','latex');
        for i = 1:numel(alpha)
            text(roundsNum - 1, alphas(i, roundsNum), ...
                 strcat('$\alpha_{', num2str(i), '}$') , 'Interpreter', 'latex');
        end
    end
    xlabel('Rounds', 'FontSize', 16, 'FontName', 'Times', 'Interpreter','latex');
end
