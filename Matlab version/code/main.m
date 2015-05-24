tic 
addpath('D:/Mathnb/MLAlgorithms/Group874/Tokmakova2012HyperPar/netlab/netlab')
load('data.mat');
nin=43;
nhidden=30;
nout=6;

[xtrain, xcontrol, ttrain, tcontrol] = data_preparation(data,nout); 
%divide to control and train; transform integer t=1..6 to t={1, 0}^6

cycles=2000; %First number of train cycles
[NET,options]=instal_net(nin,nhidden,nout,cycles);
% 
NET=train(NET,xtrain,ttrain);

NET.ncycles=200; %Usual number of train cycles

population=CreatePopulation(NET); 
%Create population by del ith neuron in net CreatePopulation([111111]) = {[011111];[101111]...[111110]}
pause;

P=12;%Number of changing networks in population
N=50;%Number of iteration

new_poputation=next_generation(P,NET,options,xtrain,ttrain,xcontrol,tcontrol,population);
 
precision_mat=zeros(NET.nout,N);
recall_mat=zeros(NET.nout,N);
 
x=zeros(N,NET.nhidden);
k=zeros(P,1);
Error=zeros(1,N);
NofNeuroun=zeros(N,1);
 net=struc(NET);
for l=1:N 
new_poputation=next_generation(P,NET,options,xtrain,ttrain,xcontrol,tcontrol,new_poputation);
s=zeros(P,1);
         for t=1:P
            net(t)=netstrunpack_nobias(new_poputation(t,:),NET);
            net(t)=train(net(t),xtrain,ttrain);     
            s(t)=mlperr(net(t), xcontrol, tcontrol);
         end
         
         [Error(l),k]=min(s);
         NET=net(k);
         Error(l)
         precision_mat(:,l)=precision(NET,xcontrol,tcontrol);
         Presicion=precision_mat(:,l);
         Recall=recall_mat(:,l);
         recall_mat(:,l)=recall(NET,xcontrol,tcontrol);
         ClassName = {'1';'2';'3';'4';'5';'6'};
         T = table(Presicion,Recall,'RowNames',ClassName);
         NofNeuroun(l)=N_of_neuron(netstrpack_nobias(NET),NET);
         x(l,:)=sctucture_of_net(NET);
end


%ptot the results 
num=1:N;
for l=1:NET.nout
hold on;
h=figure;
plot(num,recall_mat(l,:),'-g',num,precision_mat(l,:),'-r');
legend('Recall','Precision');
set(0,'DefaultAxesFontSize',14,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontSize',14,'DefaultTextFontName','Times New Roman'); 
set(gca, 'FontSize', 14, 'FontName', 'Times');
set(legend,'FontSize',16,'FontName','Times');
axis('tight');
name=['fig/real_data/Class' num2str(l) 'PresicionAndReCall.jpg'];
saveas(h,name);
name=['fig/real_data/Class' num2str(l) 'PresicionAndReCall.eps'];
saveas(h,name);
hold off;
end
plot(num,Error,'-y');
set(0,'DefaultAxesFontSize',14,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontSize',14,'DefaultTextFontName','Times New Roman'); 
set(gca, 'FontSize', 14, 'FontName', 'Times');
set(legend,'FontSize',16,'FontName','Times');
axis('tight');
name='fig/real_data/Error.jpg';
saveas(h,name);
name='fig/real_data/Error.eps';
saveas(h,name);
save('GENWORKSPACE.mat');
toc






