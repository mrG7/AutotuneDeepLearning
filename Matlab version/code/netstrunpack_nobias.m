function [net] =netstrunpack_nobias(u,net)
net.strw1=reshape(u(1:net.nin*net.nhidden),net.nin,net.nhidden);
net.strw2=reshape(u(net.nin*net.nhidden+1:net.nin*net.nhidden+net.nout*net.nhidden),net.nhidden,net.nout);