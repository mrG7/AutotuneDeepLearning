function [net]=DeleteNeuron(net,N)
net.strw1(:,N)=zeros;
net.strw2(N,:)=zeros;
net.strb1(N)=zeros;
net=ActivateNet(net); 

