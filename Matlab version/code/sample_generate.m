function [xsample,  tsample]=sample_generate(xtrain, ttrain, size_sample)
size_train=size(ttrain,1);
rvec=randperm(size_train,size_sample);
xsample=xtrain(rvec,:);
tsample=ttrain(rvec,:);
return