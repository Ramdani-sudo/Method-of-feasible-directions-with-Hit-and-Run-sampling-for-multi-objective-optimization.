function [lambda_max, positive_indices] = Compute_Steep_lambda(x_IFD, d_P, A_nonactiv, b_nonactiv)
    % This function calculates the maximum step size (lambda_max) that maintains feasibility
    % for the current direction in the context of optimization with non-active constraints.

    % Calculate b_chap (b̂) and d_chap (d̂)
    b_chap = b_nonactiv - A_nonactiv * x_IFD;  % Calculate the projected right-hand side
    d_chap = A_nonactiv * d_P;  % Calculate the directional change in the constraints

    % Find indices where d_chap > 0 (positive directions)
    positive_indices = find(d_chap > 0);

    % Check if there are no positive directions
    if isempty(positive_indices)
        lambda_max = inf;  % If no positive direction, set lambda_max to infinity
    else
        % Calculate lambda_max as the minimum of b_chap / d_chap for positive directions
        % Ensure that we avoid division by zero or invalid values in d_chap
        lambda_max = min(b_chap(positive_indices) ./ d_chap(positive_indices));
    end
end
