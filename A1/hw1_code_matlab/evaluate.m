function [ce, frac_correct] = evaluate(targets, y)
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of targets.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
%        frac_correct : (scalar) Fraction of inputs classified correctly.

% TODO: Finish this function
    incorrect = 0;
    ce = 0;
    target_size = size(targets,1);
    for i = 1:size(targets,1)
        if (targets(i) == 1 && y(i) < 0.5)||(targets(i) == 0 && y(i) >= 0.5)
                incorrect = incorrect + 1;
        end
        ce = ce - targets(i) * log(y(i)) - (1-targets(i)) * log(1-y(i));
    end
    frac_incorrect = incorrect/target_size;
    frac_correct = 1 - frac_incorrect;
    
end
