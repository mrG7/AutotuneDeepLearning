addpath('D:/Mathnb/MLAlgorithms/Group874/Tokmakova2012HyperPar/netlab/netlab')

% Create simple data 4 classes 
N=100;

x1=repmat([0,0],N,1)+(0.4*randn(N,2))+0.1*rand(N,2);
x2=repmat([1,0],N,1)+0.4*randn(N,2)+0.1*rand(N,2);
x3=repmat([0,1],N,1)+0.4*randn(N,2)+0.1*rand(N,2);
x4=repmat([1,1],N,1)+0.4*randn(N,2)+0.1*rand(N,2);
%plot(x1(:,1),x1(:,2),'.r' ,x2(:,1),x2(:,2), '.g',x3(:,1),x3(:,2), '.b',x4(:,1),x4(:,2), '.y')
class(1:N,:)=repmat([0 0 0 1], N,1 );
class(N+1:2*N,:)=repmat([0 0 1 0], N,1 );
class(2*N+1:3*N,:)=repmat([0 1 0 0], N,1 );
class(3*N+1:4*N,:)=repmat([1 0 0 0], N,1 );
data=[x1;x2;x3;x4];
data=[data, class];

nin=2;
nhidden=20;
nout=4;

x=zeros(N,1);
y=zeros(3*N,1);
for i=1:N
x(i)=4*(i-1)+1;
y(3*i-2)=x(i)+1;
y(3*i-1)=x(i)+2;
y(3*i)=x(i)+3;
end;

xcontrol=data(x,[1 2]);
xtrain=data(y,[1 2]);
tcontrol=data(x,3:6);
ttrain=data(y,3:6);
cycles = 2000;
[NET,options]=instal_net(nin,nhidden,nout,cycles);
NET=train(NET, xtrain, ttrain);

population=CreatePopulation(NET);
P=10;
N=20;
NET.ncycles=100;
new_poputation=next_generation(P,NET,options,xtrain,ttrain,xcontrol,tcontrol,population);
 presicion_mat=zeros(NET.nout,N);
 recall_mat=zeros(NET.nout,N);
 x=zeros(N,NET.nhidden);
 k=zeros(P,1);
 L=zeros(1,N);
 N_of_N=zeros(N,1);
 net=struc(NET);
rate=zeros(1,N);
for l=1:N 
new_poputation=next_generation(P,NET,options,xtrain,ttrain,xcontrol,tcontrol,new_poputation);
s=zeros(P,1);
         for t=1:P
            net(t)=netstrunpack_nobias(new_poputation(t,:),NET);
            net(t)=train(net(t),xtrain,ttrain);
            s(t)=mlperr(net(t), xcontrol, tcontrol);
         end
         
         [L(l),k]=min(s);
         NET=net(k);
         L(l)
          rate(l)=sumpresicion(NET,xcontrol,tcontrol);
         presicion_mat(:,l)=precision(NET,xcontrol,tcontrol);
         l
         Presicion=presicion_mat(:,l); 
         recall_mat(:,l)=recall(NET,xcontrol,tcontrol);

         Recall=recall_mat(:,l);
         ClassName = {'1';'2';'3';'4'};
         T = table(Presicion,Recall,'RowNames',ClassName)
         N_of_N(l)=N_of_neuron(netstrpack_nobias(NET),NET);
         N_of_N(l)
         x(l,:)=sctucture_of_net(NET);
         
end
% for k=1:NET.nout
% plot(recall_mat(k,:));
% plot(accuracy(k,:));
% end

h=figure;

for l=1:NET.nout
hold on;
h=figure;
plot(num,recall_mat(l,:),'-g',num,presicion_mat(l,:),'-r');
legend('Recall','Precision');
set(0,'DefaultAxesFontSize',14,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontSize',14,'DefaultTextFontName','Times New Roman'); 
set(gca, 'FontSize', 14, 'FontName', 'Times');
set(legend,'FontSize',16,'FontName','Times');
axis('tight');
name=['fig/test/Class' num2str(l) num2str(cycles) num2str(NET.nhidden) 'PresicionAndReCall.jpg'];
saveas(h,name);
name=['fig/test/Class' num2str(l) num2str(cycles) num2str(NET.nhidden) 'PresicionAndReCall.eps'];
saveas(h,name);
hold off;
end


%Почему одинаковая точность при большом кол-ве циклов, хотя одна должна
%падать, а другая возрастать. 
%  precision(NET,xcontrol,tcontrol)
%  precision(NET,xtrain,ttrain)
 
%We can use function nrtopt
% error_d(NET,options,xcontrol,tcontrol)
% mlperr(NET,xcontrol,tcontrol)


% for i=1:5
% NET=Sum_OBD(NET,xtrain,ttrain,options);
% x(i)=mlperr(NET,xcontrol,tcontrol);
% end;
% plot(x)
 


         %opions=zeros(14,1);
         %[X,options]=scg(error_d,options,grad_error_d); 















