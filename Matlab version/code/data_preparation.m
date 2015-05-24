function [xtrain, xcontrol,ttrain,tcontrol]=data_preparation(data,nout)

size(data)
max(size(data))
n=randperm(max(size(data)));
size(n)
pause

train=n(1:round(2*max(size(data))/3));
control=n(round(2*max(size(data))/3):max(size(data)));
xtrain=data(train,2:end-1);
xcontrol=data(control,2:end-1);
y=data(:,45);
t=zeros(length(y),nout);

for i=1:length(y)
t(i,y(i))=1;
end;
ttrain=t(train,:);
tcontrol=t(control,:);