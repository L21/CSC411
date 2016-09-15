% The main function
% If test_images is provided, it will predict the results for those too, otherwise predicts 0 for the test cases.

load labeled_images.mat;
load public_test_images.mat;
load hidden_test_images.mat;

h = size(tr_images,1);
w = size(tr_images,2);

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end

% Cross validation
for K=[3:10 15 20 35 50]
  nfold = 10;
  acc(K) = cross_validate(K, tr_images, tr_labels, nfold);
  fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
end
[maxacc bestK] = max(acc);
fprintf('K is selected to be %d.\n', bestK);
% I get a bestK of 5

% Run the classifier
prediction_of_knn = knn_classifier(bestK, tr_images, tr_labels, test_images);

test_images = gabor_features(test_images);
tr_images = gabor_features(tr_images);
trained_models = struct();

[prediction_of_svm,svm_model] = different_classifier('SVM', double(tr_images)', double(tr_labels),...
    double(test_images)');
trained_models.svm_model = svm_model;

[prediction_of_nn,nn_model] = different_classifier('NN', double(tr_images)', double(tr_labels),...
    double(test_images)');
trained_models.nn_model = nn_model;

[prediction_of_knn,knn_model] = different_classifier('KNN', double(tr_images)', double(tr_labels),...
    double(test_images)');
trained_models.knn_model = knn_model; 

[prediction_of_log, log_model] = different_classifier('Logistic', double(tr_images)', double(tr_labels),...
    double(test_images)');
trained_models.log_model = log_model;

% [prediction_of_nb, nb_model] = different_classifier('NB', double(tr_images)', double(tr_labels),...
%     double(test_images)');
% trained_models.nb_model = nb_model;

% prediction_of_dis = different_classifier('dis', double(tr_images)', double(tr_labels),...
%    double(test_images)');

save('trainedModel.mat', 'trained_models', '-mat');

data_size = size(prediction_of_log,1);
prediction_combine = zeros(data_size,4);
prediction_combine(:,1) = prediction_of_svm;
prediction_combine(:,2) = prediction_of_log;
prediction_combine(:,3) = prediction_of_nn;
prediction_combine(:,4) = prediction_of_knn;
%prediction_combine(:,4) = prediction_of_nb;

row_number = size(prediction_combine, 1);
prediction = zeros(row_number,1);

for n = 1:row_number
    prediction(n) = most_frequent(prediction_combine(n,:));
end   

% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction; zeros(1253-length(prediction), 1)];
end

save('test.mat', 'prediction', '-mat');

% Print the predictions to file
fprintf('writing the output to prediction.csv\n');
fid = fopen('prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
  fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);

clear tr_images hidden_test_images public_test_images

