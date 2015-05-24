% use distribution prob_of_mutation and give you individual to mutate, 
%delete random neuron from this individual, and add him to population
function  [populationunit]= scostic_mutation(net,populationunit)
%Ziros of random neuron 
del_neuron=randperm(net.nhidden,1);
net=netstrunpack_nobias(populationunit,net);
net=DeleteNeuron(net,del_neuron);
populationunit=netstrpack_nobias(net);




