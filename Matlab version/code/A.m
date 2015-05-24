% return cov matrix and Expected value w_0 of W
function [A,w_0] = A(net,options)
load('data.mat');
[xtrain,~,ttrain,~] = data_preparation(data,net.nin,net.nout);
%choosing size of samples to estimate the matrix W of parameters implementation
sample_size = round(2*length(xtrain)/net.nps);
if sample_size < 150
    sample_size = 150;
end

W = zeros(net.nps,net.nps);
k = sum(sum(net.strw1)) + sum(sum(net.strw2));%K is a number of active parameters

%estimating the matrix W of parameters implementation
 for i=1:k
  [xl, tl] = sample_generate(xtrain, ttrain, sample_size);  
  net = netopt(net,options,xl,tl,'scg');
  net=ActivateNet(net);
  W(:,i) = netpack_nobias(net);
 end
N = size(W,1);
w_0=(1/N)*sum(W,2)';
W_m=repmat(w_0,N,1);
A=(1/N-1)*(W-W_m)*(W-W_m)';
w_0=w_0';

