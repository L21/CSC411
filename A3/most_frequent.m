function value = most_frequent(vector)
svm_value = vector(1);
count_svm = 0;
log_value = vector(2);
count_log = 0;
nn_value = vector(3);
count_nn = 0;
knn_value = vector(4);
count_knn = 0;
% nb_value = vector(5);
% count_nb = 0;
for i = 1:4
    if vector(i) == svm_value
        count_svm = count_svm + 1;
    end
    if vector(i) == log_value
        count_log = count_log + 1;
    end
    if vector(i) == nn_value
        count_nn = count_nn + 1;
    end
    if vector(i) == knn_value
        count_knn = count_knn + 1;
    end
%     if vector(i) == nb_value
%         count_nb = count_nb + 1;
%     end
end
u = unique([count_svm, count_log, count_nn, count_knn]);

if length(u) == 1
    value = svm_value;    
else
    value = mode(vector);
end

end