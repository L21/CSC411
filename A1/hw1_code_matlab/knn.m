load('mnist_train.mat')
load('mnist_valid.mat')
load('mnist_test.mat')


figure;
hold on;
for k = [1,3,5,7,9]
    output = run_knn(k, train_inputs, train_targets, valid_inputs);
    hit_rate = mean(output==valid_targets);
    plot(k,hit_rate,'*')
    fprintf('k=%d,         hit_rate = %1.5f\n',k, hit_rate)
end
hold off;

figure;
hold on;
for k = [1,3,5,7,9]
    output = run_knn(k, train_inputs, train_targets, test_inputs);
    hit_rate = mean(output==test_targets);
    plot(k,hit_rate,'*')
    fprintf('k=%d,         hit_rate = %1.5f\n',k, hit_rate)
end
hold off;
