function [w1,w2] =netpack_nobias(u,NET)
w1=reshape(u(1:NET.nin*NET.nhidden),NET.nin,NET.nhidden);
w2=reshape(u(NET.nin*NET.nhidden+1:NET.nin*NET.nhidden+NET.nout*NET.nhidden),NET.nhidden,NET.nout);