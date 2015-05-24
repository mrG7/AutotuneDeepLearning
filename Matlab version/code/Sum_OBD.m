function NET=Sum_OBD(NET,xtrain,ttrain,options)
[sal1,~]=ComputeHessianAndConvexity(NET,xtrain,ttrain);
[~,NNmin]=min(Nsal1(Nsal1~=0));
NET.strw1(:,NNmin)=zeros(size(NET.strw1,1),1);
NET.strw2(NNmin,:)=zeros(1,size(NET.strw2,2));
