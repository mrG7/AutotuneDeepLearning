function [u]=netpack_nobias(net)
u=vertcat(reshape(net.w1, net.nin*net.nhidden, 1),reshape(net.w2, net.nout*net.nhidden, 1));