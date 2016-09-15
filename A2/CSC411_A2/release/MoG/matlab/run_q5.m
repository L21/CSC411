% Choose the best mixture of Gaussian classifier you have, compare this
% mixture of Gaussian classifier with the neural network you implemented in
% the last assignment. 


% Train neural network classifier. The number of hidden units should be
% equal to the number of mixture components. 

% Show the error rate comparison.

%-------------------- Add your code here --------------------------------
load digits;
min_variance = 0.01;
ite = 20;
k = 15;
[p_2,mu_2,vary_2,logProbtr_2]=mogEM(train2,K,ite,min_variance,1);
[p_3,mu_3,vary_3,logProbtr_3]=mogEM(train3,K,ite,min_variance,1);
[logProbtr_train2_2] = mogLogProb(p_2,mu_2,vary_2,train2);
[logProbtr_train2_3] = mogLogProb(p_3,mu_3,vary_3,train2);
[logProbtr_train3_2] = mogLogProb(p_2,mu_2,vary_2,train3);
[logProbtr_train3_3] = mogLogProb(p_3,mu_3,vary_3,train3);
[logProbtr_valid2_2] = mogLogProb(p_2,mu_2,vary_2,valid2);
[logProbtr_valid2_3] = mogLogProb(p_3,mu_3,vary_3,valid2);
[logProbtr_valid3_2] = mogLogProb(p_2,mu_2,vary_2,valid3);
[logProbtr_valid3_3] = mogLogProb(p_3,mu_3,vary_3,valid3);
[logProbtr_test2_2] = mogLogProb(p_2,mu_2,vary_2,test2);
[logProbtr_test2_3] = mogLogProb(p_3,mu_3,vary_3,test2);
[logProbtr_test3_2] = mogLogProb(p_2,mu_2,vary_2,test3);
[logProbtr_test3_3] = mogLogProb(p_3,mu_3,vary_3,test3);
train_error_mean = (mean(logProbtr_train2_2 < logProbtr_train2_3)...
    + mean(logProbtr_train3_2 > logProbtr_train3_3)) / 2;
valid_error_mean = (mean(logProbtr_valid2_2 < logProbtr_valid2_3)...
    + mean(logProbtr_valid3_2 > logProbtr_valid3_3)) / 2;
test_error_mean = (mean(logProbtr_test2_2 < logProbtr_test2_3)...
    + mean(logProbtr_test3_2 > logProbtr_test3_3)) / 2;
train_error_mean
valid_error_mean
test_error_mean


