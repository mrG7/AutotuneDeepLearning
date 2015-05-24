function [x]=sctucture_of_net(net)
NNeuron=net.nhidden;
x=ones(NNeuron,1);
for i=1:NNeuron
    if(net.strw1(:,i)==zeros(net.nin,1)) 
        x(i)=0; 
    end
end