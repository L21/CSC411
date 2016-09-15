load digits;
min_variance = 0.01;
cluster_number = 2;
ite = 30;
%[p_2,mu_2,vary_2,logProbtr_2]=mogEM(train2,cluster_number,ite,min_variance,1);
[p_3,mu_3,vary_3,logProbtr_3]=mogEM(train3,cluster_number,ite,min_variance,1);

%visualize_digits(mu_2);
%visualize_digits(vary_2);

visualize_digits(mu_3);
visualize_digits(vary_3);

%display(p_2,'mixing proportion');
%logProbtr_2
p_3
logProbtr_3