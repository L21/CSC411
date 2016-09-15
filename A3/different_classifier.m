function [label,train_model] = different_classifier(classifier, trian_input,...
    train_label, test_input)
if strcmp(classifier, 'NN')
    net = patternnet(60);
    net = train(net, trian_input', full(ind2vec(train_label')));
    %view(net)
    y = net(test_input');
    label = vec2ind(y);
    train_model = net;
    
else
    if  strcmp(classifier, 'SVM')
        template = templateSVM('Standardize',1,'KernelFunction','linear');
        svm_model_cross_validation = fitcecoc(trian_input,train_label,...
            'CrossVal','on','Learners',template);
        loss_value = kfoldLoss(svm_model_cross_validation);     
        cross_validation = svm_model_cross_validation;
        fprintf('SVM class Error: %f\n', loss_value);
    end

    if strcmp(classifier, 'NB')
        template = templateNaiveBayes();
        nb_model_cross_validation = fitcecoc(trian_input,train_label,...
            'CrossVal','on','Learners',template);
        loss_value = kfoldLoss(nb_model_cross_validation);     
        cross_validation = nb_model_cross_validation;
        fprintf('NB class Error: %f\n', loss_value);
    end

    if strcmp(classifier, 'KNN')
        template = templateKNN('NumNeighbors', 5,'Standardize',1);
        knn_model_cross_validation = fitcecoc(trian_input,train_label,...
            'CrossVal','on','Learners',template);
        loss_value = kfoldLoss(knn_model_cross_validation);
        cross_validation = knn_model_cross_validation;
        fprintf('Knn class Error: %f\n', loss_value);

    end
    
    if  strcmp(classifier, 'Logistic')
        template =templateLinear();
        log_cross_validation = fitcecoc(trian_input,train_label,...
            'CrossVal','on','Learner', 'logistic', 'Learners',template);
        loss_value = kfoldLoss(log_cross_validation);     
        cross_validation = log_cross_validation;
        fprintf('logistic class Error: %f\n', loss_value);
    end
    
    if  strcmp(classifier, 'dis')
        template = templateDiscriminant();
        dis_cross_validation = fitcecoc(trian_input,train_label,...
            'CrossVal','on','Learners',template);
        loss_value = kfoldLoss(dis_cross_validation);     
        cross_validation = dis_cross_validation;
        fprintf('dis class Error: %f\n', loss_value);
    end
    
    cross_validation = cross_validation.Trained{1};
    label = predict(cross_validation, test_input);
    train_model = cross_validation;
end