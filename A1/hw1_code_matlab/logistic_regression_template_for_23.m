%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_valid;
load mnist_test;
load mnist_train_small;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate =0.1;
% Weight regularization parameter
hyperparameters.weight_regularization =1;
% Number of iterations
hyperparameters.num_iterations =239;
% Logistics regression weights
% TODO: Set random weights.
weights = zeros(size(train_inputs,2)+1,1);

lambda = [0.001,0.01,0.1,1];

%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic_pen', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                        % other hyperparameters

N = size(train_inputs, 1);
training_matrix = zeros(4,2);
validation_matrix = zeros(4,2);
training_matrix_error = zeros(4,2);
validation_matrix_error = zeros(4,2);
%% Begin learning with gradient descent.
for i = 1:4
    cross_entropy_train_sum = 0;
    cross_entropy_valid_sum = 0;
    classfication_error_train_sum = 0;
    classfication_error_validate_sum = 0;
    for k = 1:1
        hyperparameters.weight_regularization = lambda(i);
        weights = zeros(size(train_inputs,2)+1,1);
        for t = 1:hyperparameters.num_iterations

            %% TODO: You will need to modify this loop to create plots etc.

            % Find the negative log likelihood and derivative w.r.t. weights.
            [f, df, predictions] = logistic_pen(weights, ...
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
            
            %% Print some stats.
            fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
                    t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);

        end

        cross_entropy_train_sum = cross_entropy_train_sum + cross_entropy_train;
        cross_entropy_valid_sum = cross_entropy_valid_sum + cross_entropy_valid;
        classfication_error_train_sum = classfication_error_train_sum + 1-frac_correct_train;
        classfication_error_validate_sum = classfication_error_validate_sum + 1-frac_correct_valid;
    end
    training_matrix(i,1) = lambda(i);
    training_matrix(i,2) = cross_entropy_train_sum/1;     
    validation_matrix(i,1) = lambda(i);
    validation_matrix(i,2) =cross_entropy_valid_sum/1;
    training_matrix_error(i,1) = lambda(i);
    training_matrix_error(i,2) = classfication_error_train_sum/1;     
    validation_matrix_error(i,1) = lambda(i);
    validation_matrix_error(i,2) =classfication_error_validate_sum/1; 
    predictions_test = logistic_predict(weights, test_inputs);
    [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
    fprintf('cross entropy for test: %d ,classification error: %d\n',cross_entropy_test,1-frac_correct_test);
end
figure;
sub_fighre_1 = subplot(2,2,1);
title('Cross Entropy(training data)');
hold(sub_fighre_1,'on');
sub_fighre_2 = subplot(2,2,2);
title('Cross Entropy(validation data)');
hold(sub_fighre_2,'on');
sub_fighre_3 = subplot(2,2,3);
title('Classfication Error(training data)');
hold(sub_fighre_3,'on');
sub_fighre_4 = subplot(2,2,4);
title('Classfication Error(validation data)');
hold(sub_fighre_4,'on');
for i = 1:4
    plot(sub_fighre_1,lambda(i),training_matrix(i,2),'*');
    plot(sub_fighre_2,lambda(i),validation_matrix(i,2),'*');
    plot(sub_fighre_3,lambda(i),training_matrix_error(i,2),'*');
    plot(sub_fighre_4,lambda(i),validation_matrix_error(i,2),'*');
end

