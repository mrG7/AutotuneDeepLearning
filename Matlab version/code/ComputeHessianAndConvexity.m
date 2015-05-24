%COMPUTE HESSIAN AND CONVEXITY FUNCTION FOR A NEURAL NET
%hessian is a vector having size equals quantity of weights.
%an element of hessian is the second derivative of E - function of quality
%with respect to the weight.
% s1 and s2 are matrixes of Convexity. s1(i,j) corresponds to w1(i,j),
% s2(i,j) corresponds to w2(i,j)

function [s1, s2] = ComputeHessianAndConvexity(net, xlearn, ylearn)

%A1,A2 - outputs of neurons on layers 1 and 2
w1 = net.w1;
w2 = net.w2;
nhidden = net.nhidden;
nout = net.nout;
nin = net.nin;
A1 = xlearn * w1;
A2 = tanh (A1) * w2;
[N, nin] = size(xlearn);



phi1 = tanh(A1);
phi2 = logsig(A2);
%F - vector of results computing by neural net
F = phi2;
d_phi1 = phi1 .* (ones(size(phi1)) - phi1);
d_phi2 = phi2 .* (ones(size(phi2)) - phi2);
d2_phi1 = d_phi1 .* (ones(size(phi1)) - phi1);
d2_phi2 = d_phi2 .* (ones(size(phi2)) - phi2);

% o = (sum(phi2'))' / nout;
hessIndex = 0;

Hessian = zeros(nhidden*(nin + nout),1);

%Convexity is a vector with size which is equal to size of hessian.
%Its element Convexity(i) = weight_i^2 * hessian(i) / 2
%we do need Convexity to use algorithm "optimal brain damage"
Convexity = zeros(nhidden*(nin + nout),1);

for j = 1:nhidden
    for i = 1:nin
        hessIndex = hessIndex + 1;
        for n = 1:N
            for p = 1:nout
                Hessian(hessIndex) = Hessian(hessIndex) + ...
                    + (1 / N) * ( (d_phi2(n,p) * w2(j,p) * d_phi1(n,j) * xlearn(n,i))^2 - ...
                    - (d2_phi2(n,p) * (w2(j,p) * d_phi1(n,j) * xlearn(n,i))^2 + ...
                        + d_phi2(n,p) * w2(j,p) * d2_phi1(n,j) * (xlearn(n,i))^2) * (ylearn(n) - F(n,p)));
            end;
        end;
        Convexity(hessIndex) = (w1(i,j))^2 * Hessian(hessIndex) / 2; 
    end;
end;

for k = 1:nout
    for j = 1:nhidden
        hessIndex = hessIndex + 1;
        for n = 1:N
                Hessian(hessIndex) = Hessian(hessIndex) + ...
                    + (1 / N) * ( (d_phi2(n,k))^2 * (phi1(n,j))^2 - ...
                    - d2_phi2(n,k)*(phi1(n,j))^2* (ylearn(n) - F(n,k)));
        end;
        Convexity(hessIndex) = (w2(j,k))^2 * Hessian(hessIndex) / 2;
    end;
end;

mark1 = nin*nhidden;
mark2 = mark1 + nhidden*nout;
s1 = reshape(Convexity(1:mark1), nin, nhidden);
s2 = reshape(Convexity(mark1+1: mark2), nhidden, nout);