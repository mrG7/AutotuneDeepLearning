function [str,dataerr,error] = error_structure(net,xtrain, ttrain,options)    
%w=netpack_nobias(net);
%w=w(net.nin*net.nhidden+1:(net.nout+net.nin)*net.nhidden,1); %коэфф. W2 в столбец 120 штук
%[A_cov,w_0]=estimateCovarianceLaplace(net,xtrain,ttrain, 'diag');
%(w-w_0)'*diag(A_cov)*(w-w_0)+
%error=mlperr(net,xtrain, ttrain);
[sal1,~]=ComputeHessianAndConvexity(net,xtrain,ttrain);
str=sum(sum(sal1));
dataerr=mlperr(net,xtrain,ttrain);
error=mlperr(net,xtrain,ttrain)+str;

