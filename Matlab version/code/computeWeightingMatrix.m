function R = computeWeightingMatrix(net,xtrain)

% Function computes weighting matrix with diagonal y_i*(1 - y_i)
% INPUT:
%   X       - object-feature matrix with size m x n
%   theta   - parameters vector with size n
% OUTPUT:
%   R       - m x m dimensional diagonal matrix with elements y_i*(1-y_i)
sz=net.nin;
R=zeros(sz);
y = mlpfrw(net,xtrain);
for i=1:sz
    for j=1:sz
        for n=1:net.nout
        R(i,j)=R(i,j)+y(i,n)*(kron(i,j)-y(j,n));
        end
    end
end
 

end