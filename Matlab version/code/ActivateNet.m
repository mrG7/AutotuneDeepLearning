function [net]= ActivateNet(net)
net.w1=(net.w1).*(net.strw1);
net.w2=(net.w2).*(net.strw2);
net.b1=(net.b1).*(net.strb1);
