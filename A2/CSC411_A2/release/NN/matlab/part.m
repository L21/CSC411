init_nn;
train = run_knn(3, transpose(inputs_train), transpose(target_train), transpose(inputs_train));
valid = run_knn(3, transpose(inputs_train), transpose(target_train), transpose(inputs_valid));
test = run_knn(3, transpose(inputs_train), transpose(target_train), transpose(inputs_test));
train_error = 1 - (sum(xor(target_train,  transpose(train)))/length(inputs_train));
valid_error = 1 - (sum(xor(target_valid,  transpose(valid)))/length(inputs_valid));
test_error = 1 - (sum(xor(target_test,  transpose(test)))/length(inputs_test));
