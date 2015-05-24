function [u]=netstrpack_nobias(net)
u=vertcat(reshape(net.strw1, net.nin*net.nhidden, 1),reshape(net.strw2, net.nout*net.nhidden, 1));