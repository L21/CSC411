%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_valid;
load mnist_test;

load mnist_train_small;
% train_inputs = train_inputs_small;
% train_targets = train_targets_small;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate =0.1;
% Weight regularization parameter
hyperparameters.weight_regularization =0.001;
% Number of iterations
hyperparameters.num_iterations =239;
% Logistics regression weights
% TODO: Set random weights.
weights = zeros(size(train_inputs,2)+1,1);


%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                        % other hyperparameters

N = size(train_inputs, 1);
training_matrix = zeros(hyperparameters.num_iterations,2);
validation_matrix = zeros(hyperparameters.num_iterations,2);
%% Begin learning with gradient descent.
for t = 1:hyperparameters.num_iterations

	%% TODO: You will need to modify this loop to create plots etc.

	% Find the negative log likelihood and derivative w.r.t. weights.
	[f, df, predictions] = logistic(weights, ...
                                           train_inputs_small, ...
                                           train_targets_small, ...
                                           hyperparameters);

  [cross_entropy_train, frac_correct_train] = evaluate(train_targets_small, predictions);


    if isnan(f) || isinf(f)
		error('nan/inf error');
	end

	%% Update parameters.
	weights = weights - hyperparameters.learning_rate .* df / N;

  predictions_valid = logistic_predict(weights, valid_inputs);
  [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
  training_matrix(t,1) = t;
  training_matrix(t,2) = cross_entropy_train;     
  validation_matrix(t,1) = t;
  validation_matrix(t,2) =cross_entropy_valid;  
	%% Print some stats.
	fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
			t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);

end
figure;
sub_fighre_1 = subplot(2,1,1);
title('CE(training data)');
hold(sub_fighre_1,'on');
sub_fighre_2 = subplot(2,1,2);
title('CE(validation data)');
hold(sub_fighre_2,'on');
for i = 1:t
    plot(sub_fighre_1,i,training_matrix(i,2),'o');
    plot(sub_fighre_2,i,validation_matrix(i,2),'o');
end
predictions_test = logistic_predict(weights, test_inputs);
[cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
fprintf('cross entropy for test: %d ,classification error: %d\n',cross_entropy_test,1-frac_correct_test);
