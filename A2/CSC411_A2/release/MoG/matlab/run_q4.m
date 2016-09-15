load digits;

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];
min_variance = 0.01;
ite = 20;
for i = 1 : 4
    K = numComponent(i);
% Train a MoG model with K components for digit 2
%-------------------- Add your code here --------------------------------
    [p_2,mu_2,vary_2,logProbtr_2]=mogEM(train2,K,ite,min_variance,1);

% Train a MoG model with K components for digit 3
%-------------------- Add your code here --------------------------------
    [p_3,mu_3,vary_3,logProbtr_3]=mogEM(train3,K,ite,min_variance,1);

% Caculate the probability P(d=1|x) and P(d=2|x), 
% classify examples, and compute the error rate
% Hints: you may want to use mogLogProb function
%-------------------- Add your code here --------------------------------
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
    errorTrain(i) = train_error_mean;
    errorValidation(i) = valid_error_mean;
    errorTest(i) = test_error_mean;
    
end

% Plot the error rate
%-------------------- Add your code here --------------------------------
figure;
clf;
hold on; ...
for i=1:4
    plot(numComponent(1:i),errorTrain(1:i),'r');
    plot(numComponent(1:i),errorValidation(1:i),'g');
    plot(numComponent(1:i),errorTest(1:i),'b');
end
legend('Train', 'validation','Test'),...
title('Average Error Rate'), ...
xlabel('Cluster number'), ...
ylabel('Error');
hold off;