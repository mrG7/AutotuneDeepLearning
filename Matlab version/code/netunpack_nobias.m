function [net] =netunpack_nobias(u,net)
net.w1=reshape(u(1:net.nin*net.nhidden),net.nin,net.nhidden);
net.w2=reshape(u(net.nin*net.nhidden+1:net.nin*net.nhidden+net.nout*net.nhidden),net.nhidden,net.nout);