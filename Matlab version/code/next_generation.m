
function [new_population]=next_generation(P,net,options,xtrain, ttrain,~,~, population)
N=size(population,1);
net_array(N,1)=struct(net);
error(N,1)=zeros;
str(N,1)=zeros;
dataerr(N,1)=zeros;

for i=1:N
   net_array(i)=net;
   net_array(i)=netstrunpack_nobias(population (i,:),net_array(i));
   net_array(i)=train(net_array(i), xtrain, ttrain);
   [str(i),dataerr(i),error(i)]=error_structure(net_array(i),xtrain,ttrain, options);
   
end 
prob_of_cross=exp(-error/max(error))/sum(exp(-(error)/max(error)));

%Cross+mutation
k=zeros(P,1); %number of units from population that will cross and mutation
for i=1:P
k(i)=Randn_by_our_distribution(prob_of_cross);    
end
new_population=population(k,:); %units from population
rindex=randperm(P);
for i=1:2:P-1
   n=randperm(net.nhidden,1);
   net1=netstrunpack_nobias(new_population(rindex(i),:),net);
   net2=netstrunpack_nobias(new_population(rindex(i+1),:),net);
   net1.strw1(:,1:n)=net2.strw1(:,1:n);
   net1.strw2(1:n,:)=net2.strw2(1:n,:);
   net2.strw1(:,n+1:net.nhidden)=net1.strw1(:,n+1:net.nhidden);
   net2.strw2(n+1:net.nhidden,:)=net1.strw2(n+1:net.nhidden,:);
   new_population(rindex(i),:)=netstrpack_nobias(net1);
   new_population(rindex(i+1),:)=netstrpack_nobias(net2);
   new_population(rindex(i+1),:)=scostic_mutation(net,new_population(rindex(i+1),:));
   new_population(rindex(i),:)=scostic_mutation(net,new_population(rindex(i),:));
end
    population(k,:)=new_population;
    new_population=population;
