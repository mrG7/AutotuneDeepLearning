function [population]=CreatePopulation(net)
N_neuron=net.nhidden %кол-во нейронов
w=netpack_nobias(net);
netarray(N_neuron,1)=struct(net); 
population(N_neuron,size(w,1))=zeros;
for i=1:N_neuron %переделать без цикла ?
netarray(i)=DeleteNeuron(net,i);
population(i,:)=netstrpack_nobias(netarray(i));
end