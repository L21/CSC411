function [f, df, y] = logistic(weights, data, targets, hyperparameters)
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%	  targets:    N x 1 vector of targets class probabilities.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value.
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
% y:             N x 1 vector of p  robabilities. This is the output of the classifier.
%

%TODO: finish this function
    f = 0;
    w = size(weights,1);
    df = zeros(size(weights));
    y = zeros(size(data,1));
    for i = 1:size(data,1)
        z = weights(1:(w-1)).'*data(i,:).' + weights(w);
        y(i) = sigmoid(z);
        f = f + (1-targets(i))*z + log(1+exp(-z));
        for j = 1:(w-1)
            df(j) = df(j) + data(i,j)*((1-targets(i))-(1-sigmoid(z)));
        end
        df(w) = df(w) + (1-targets(i)-(1-sigmoid(z)));
    end
end
