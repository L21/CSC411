load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 
min_variance = 0.01;
cluster_number = 20;
ite = 30;
[p,mu,vary,logProbtr]=mogEM(x,cluster_number,ite,min_variance,1);
logProbtr

