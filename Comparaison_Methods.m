% Function: Comparaison_Methods_final
% Description: Compares Pareto fronts from three methods (MFD of ramdani et Al., MFD of Morovati and Pourkarimi, NSGA-II) using multiple performance indicators.
% Inputs:
%   - Ynd_MFD: Pareto front  MFD of Ramdani et al. (matrix of objectives)
%   - Ynd_Morovati: Pareto front from MFD of Morovati and Pourkarimi (matrix of objectives)
%   - Y_ND_NSGA: Pareto front from NSGA-II method (matrix of objectives)
%   - Type_obj: Optimization type ('min' or 'max')
% Outputs:
%   - Plot: Visualization of the global Pareto front Y_ND^* and filtered fronts, plus a separate plot of the filtered NSGA-II front (non-normalized)
% Authors: Ramdani Zoubir*, Addoune Smail, Brahmi Boualem
% Affiliation: Faculty of Mathematics and Computer Sciences,
%              University Mohamed El Bachir El Ibrahimi, Bordj Bou Arreridj, Algeria
% Corresponding Author: Ramdani Zoubir (z.ramdani@univ-bba.dz)

function Comparaison_Methods_final(Ynd_MFD, Ynd_Morovati, Y_ND_NSGA, Type_obj)
    % Validate Type_obj
    if ~ismember(lower(Type_obj), {'min', 'max'})
        error('Type_obj must be ''min'' or ''max''.');
    end

    % Check for non-empty and valid fronts
    fronts = {Ynd_MFD, Ynd_Morovati, Y_ND_NSGA};
    names = {'MFD', 'Morovati', 'NSGA-II'};
    valid_fronts = cellfun(@(x) ~isempty(x) && ismatrix(x) && ~any(isnan(x(:))) && ~any(isinf(x(:))), fronts);
    num_valid = sum(valid_fronts);

    if num_valid < 2
        error('At least two non-empty Pareto fronts are required for comparison.');
    end

    % Select valid fronts and their names
    valid_fronts_idx = find(valid_fronts);
    selected_fronts = fronts(valid_fronts_idx);
    selected_names = names(valid_fronts_idx);

    % Validate dimensions
    p = size(selected_fronts{1}, 2); % Number of objectives

    % Warn if fewer than three fronts are available
    if num_valid < 3
        warning('Only %d valid Pareto front(s) available. Comparing %s.', ...
            num_valid, strjoin(selected_names, ' and '));
    end

    % Normalize objectives to ensure uniform scales (for performance indicators and Y_ND^*)
    allPoints = vertcat(selected_fronts{:});
    minVals = min(allPoints, [], 1);
    maxVals = max(allPoints, [], 1);
    range = maxVals - minVals;
    % Check for NaN or Inf in range
    if any(isnan(range)) || any(isinf(range))
        error('Range contains NaN or Inf values after calculation.');
    end
    % Replace values smaller than 1e-10 with 1 to avoid division by zero
    range(range < 0.0000000001) = 1;
    normalized_fronts = cellfun(@(x) (x - minVals) ./ range, selected_fronts, 'UniformOutput', false);

    % Generate global Pareto front Y_ND^* = Y_ND^1 U Y_ND^2 U Y_ND^3 using partial dominance
    ReferenceFront = pareto_front(vertcat(normalized_fronts{:}), Type_obj);
    if isempty(ReferenceFront)
        warning('Global Pareto front Y_ND^* is empty. Returning default values.');
        return;
    end

    % Denormalize ReferenceFront for plotting in the original space
    ReferenceFront_original = ReferenceFront .* range + minVals;

    % Eliminate dominated solutions by keeping points that are not strictly dominated by Y_ND^* (using partial dominance)
    redetermined_fronts_normalized = cell(1, num_valid);
    original_counts = cellfun(@(x) size(x, 1), normalized_fronts); % Number of points before filtering
    for i = 1:num_valid
        redetermined_fronts_normalized{i} = filter_front_with_partial_dominance(normalized_fronts{i}, ReferenceFront, Type_obj);
    end

    % Denormalize the filtered fronts to compute ideal and nadir points in the original space
    redetermined_fronts = cell(1, num_valid);
    for i = 1:num_valid
        redetermined_fronts{i} = redetermined_fronts_normalized{i} .* range + minVals;
    end

    % Compute Ideal and Nadir points for each method (in original space for display)
    ideal_points = zeros(num_valid, p);
    nadir_points = zeros(num_valid, p);
    % Compute Nadir points in normalized space for Hypervolume reference point
    nadir_points_normalized = zeros(num_valid, p);

    for i = 1:num_valid
        % Use the filtered front in original space for Ideal/Nadir display
        front = redetermined_fronts{i};
        % Use the filtered front in normalized space for Hypervolume reference point
        front_normalized = redetermined_fronts_normalized{i};

        % Handle empty fronts
        if isempty(front)
            ideal_points(i, :) = NaN(1, p);
            nadir_points(i, :) = NaN(1, p);
            nadir_points_normalized(i, :) = NaN(1, p);
            continue;
        end

        % Compute Ideal and Nadir Points in original space (for display)
        if strcmpi(Type_obj, 'min')
            ideal_point = min(front, [], 1);
            nadir_point = max(front, [], 1);
        else % max
            ideal_point = max(front, [], 1);
            nadir_point = min(front, [], 1);
        end
        ideal_points(i, :) = ideal_point;
        nadir_points(i, :) = nadir_point;

        % Compute Nadir Point in normalized space (for Hypervolume reference)
        if strcmpi(Type_obj, 'min')
            nadir_point_norm = max(front_normalized, [], 1);
        else % max
            nadir_point_norm = min(front_normalized, [], 1);
        end
        nadir_points_normalized(i, :) = nadir_point_norm;
    end

    % Compute the nadir point of the global Pareto front Y_ND^* (in normalized space, for display)
    if strcmpi(Type_obj, 'min')
        nadir_point_ref = max(ReferenceFront, [], 1);
    else % max
        nadir_point_ref = min(ReferenceFront, [], 1);
    end

    % Compute performance indicators (using normalized fronts)
    delta = 0.1; % Margin for reference point adjustment
    indicators = struct('hv', zeros(1, num_valid), ...
                        'spread', zeros(1, num_valid), ...
                        'purity', zeros(1, num_valid), ...
                        'ipd', zeros(1, num_valid), ...
                        'npd', zeros(1, num_valid), ...
                        'coverage', zeros(1, num_valid), ...
                        'uniformity', zeros(1, num_valid), ...
                        'spacing', zeros(1, num_valid), ...
                        'generalized_spread', zeros(1, num_valid), ...
                        'delta_index', zeros(1, num_valid));

    for i = 1:num_valid
        front = redetermined_fronts_normalized{i};
        % Use the nadir point of the filtered and normalized front as the reference point
        if isempty(front) || any(isnan(nadir_points_normalized(i, :)))
            refPoint_i = ones(1, p); % Fallback: Use [1, 1, ..., 1] as reference point (normalized space maximum)
            fprintf('Warning: Using fallback reference point [1, ..., 1] for %s due to empty front or invalid nadir.\n', selected_names{i});
        else
            % Use the nadir point computed from the filtered and normalized front
            refPoint_i = nadir_points_normalized(i, :);
            % Adjust the reference point to ensure it is strictly dominated
            if strcmpi(Type_obj, 'min')
                refPoint_i = refPoint_i + delta; % Worse than nadir
            else
                refPoint_i = refPoint_i - delta; % Better than nadir
            end
        end

        % Compute indicators
        indicators.hv(i) = computeHypervolume(front, refPoint_i, Type_obj, selected_names{i});
        indicators.ipd(i) = computeIPD(front, ReferenceFront, Type_obj);
        indicators.npd(i) = computeNPD(front, ReferenceFront, Type_obj);
        indicators.spacing(i) = computeStandardSpacing(front);
        indicators.generalized_spread(i) = computeGeneralizedSpread(front, ReferenceFront, Type_obj);
        end

    % Compute Purity Index using original normalized fronts
    indicators.purity = calculate_purity_dynamic(normalized_fronts, ReferenceFront);
    if ismember(3, valid_fronts_idx)
%         fprintf('\n=== Plotting the Filtered NSGA-II Pareto Front (Non-Normalized) ===\n');
        plot_NSGAII_filtered_front(redetermined_fronts{find(valid_fronts_idx == 3)}, Type_obj, p);
else
        fprintf('\n=== Skipping NSGA-II Filtered Front Plot: NSGA-II front is invalid or empty ===\n');
    end

    % Display the number of non-dominated points after filtering
    non_dominated_counts = cellfun(@(x) size(x, 1), redetermined_fronts);
    fprintf('Number of Non-Dominated Points After Filtering Using Partial Dominance ===\n');
    %fprintf('Note: Includes weakly non-dominated solutions with a dominance tolerance of 0.001.\n');
    for i = 1:num_valid
        fprintf('%s: %d points (out of %d original points)\n', ...
            selected_names{i}, non_dominated_counts(i), original_counts(i));
    end
    fprintf('\n');
    for i = 1:num_valid
        if any(isnan(ideal_points(i, :)))
            fprintf('%s: No non-dominated points after filtering.\n', selected_names{i});
            %fprintf('  Ideal Point: [N/A]\n');
            %fprintf('  Nadir Point: [N/A]\n\n');
            continue;
        end
    end

    % Display results
    display_results(indicators, selected_names);
end

% Helper Function: Plot the filtered NSGA-II front (non-normalized)
function plot_NSGAII_filtered_front(NSGAII_front, Type_obj, p)
    figure('Name', 'Filtered NSGA-II Pareto Front (Non-Normalized)', 'NumberTitle', 'off');
    hold on;
    if isempty(NSGAII_front)
        fprintf('Warning: NSGA-II filtered front is empty. No plot generated.\n');
        hold off;
        return;
    end
    if p == 2
        scatter(NSGAII_front(:,1), NSGAII_front(:,2), 40, 'ro', 'filled');
        xlabel('Objective 1');
        ylabel('Objective 2');
        title('Filtered NSGA-II Pareto Front ');
        legend('show');
        grid on;
    elseif p == 3
        scatter3(NSGAII_front(:,1), NSGAII_front(:,2), NSGAII_front(:,3), 40, 'ro', 'filled');
        xlabel('Objective 1'); ylabel('Objective 2'); zlabel('Objective 3');
        view(45, 30);
        grid on;
        view(45, 30);
    else
        fprintf('Number of objectives > 3. Plotting NSGA-II filtered front for Obj1 vs Obj2.\n');
        scatter(NSGAII_front(:,1), NSGAII_front(:,2), 50, 'go', 'DisplayName', 'NSGA-II (Filtered)');
        xlabel('Objective 1');
        ylabel('Objective 2');
        title('Filtered NSGA-II Pareto Front ');
        legend('show');
        grid on;
    end
    hold off;
end

% Helper Function: Filter front using partial dominance
function filtered_front = filter_front_with_partial_dominance(front, ReferenceFront, Type_obj)
    if isempty(front)
        filtered_front = front;
        return;
    end
    N = size(front, 1);
    M = size(ReferenceFront, 1);
    keep = true(N, 1);
    for i = 1:N
        for j = 1:M
            if dominates(ReferenceFront(j, :), front(i, :), Type_obj)
                keep(i) = false;
                break;
            end
        end
    end
    filtered_front = front(keep, :);
end

% Helper Function: Hypervolume (HV)
function hv = computeHypervolume(ParetoFront, refPoint, Type_obj, method_name)
     [N, p] = size(ParetoFront);
    if N == 0
        fprintf('Warning: Filtered front for %s is empty. Hypervolume set to 0.\n', method_name);
        hv = 0;
        return;
    end

%     fprintf('Computing hypervolume for %s: %d points in filtered front.\n', method_name, N);
%     fprintf('Reference point for %s: [', method_name);
%     fprintf(' %.4f', refPoint);
%     fprintf(' ]\n');

    % Adjust for optimization type (convert maximization to minimization)
    ParetoFront = adjust_for_optimization(ParetoFront, Type_obj);
    refPoint = adjust_for_optimization(refPoint, Type_obj);

    % Validate reference point
    if strcmpi(Type_obj, 'min')
        if any(any(ParetoFront > repmat(refPoint, N, 1), 2))
            warning('Reference point for %s is not dominated by all points in the front.', method_name);
        end
    else
        if any(any(ParetoFront < repmat(refPoint, N, 1), 2))
            warning('Reference point for %s is not dominant over all points in the front.', method_name);
        end
    end

    % Sort points lexicographically by all objectives (for p <= 3)
    ParetoFront = sortrows(ParetoFront, 1:p, 'ascend');
    hv = 0;

    if p == 2
        % 2D Hypervolume: Sum of exclusive rectangles
        for i = 1:N
            x1 = refPoint(1) - ParetoFront(i,1);
            if i == 1
                x2 = refPoint(2) - ParetoFront(i,2);
            else
                x2 = ParetoFront(i-1,2) - ParetoFront(i,2);
            end
            % Include all non-negative contributions to capture weakly non-dominated points
            if x1 >= 0 && x2 >= 0
                hv = hv + x1 * x2;
            end
        end
    elseif p == 3
        % 3D Hypervolume: Sum of exclusive cuboids
        for i = 1:N
            x1 = refPoint(1) - ParetoFront(i,1);
            x3 = refPoint(3) - ParetoFront(i,3);
            if i == 1
                x2 = refPoint(2) - ParetoFront(i,2);
            else
                x2 = ParetoFront(i-1,2) - ParetoFront(i,2);
            end
            % Include all non-negative contributions to capture weakly non-dominated points
            if x1 >= 0 && x2 >= 0 && x3 >= 0
                hv = hv + x1 * x2 * x3;
            end
        end
    else
        % For p > 3: Use Monte Carlo approximation, optimized for p > 5
        fprintf('Using Monte Carlo approximation for hypervolume with %d objectives for %s.\n', p, method_name);
        
        % Step 1: Define the bounding box using the ideal point and reference point
%         ideal_point = min(ParetoFront, [], 1); % Best values for minimization (after adjustment)
        lower_bound = ideal_point; % Lower bound of the hyperrectangle
        upper_bound = refPoint; % Upper bound (reference point)

        % Validate bounds
        if any(lower_bound >= upper_bound)
            warning('Invalid bounds for Monte Carlo hypervolume for %s. Returning 0.', method_name);
            hv = 0;
            return;
        end

        % Step 2: Adjust number of samples based on dimensionality
        num_samples = 500000; % Default for p > 5 to improve accuracy in high dimensions
        if p > 5
            num_samples = 1000000; % Increase samples for better accuracy with p > 5
            fprintf('Increased to %d samples for improved accuracy with %d objectives.\n', num_samples, p);
        end

        % Step 3: Generate random points
        random_points = zeros(num_samples, p);
        for m = 1:p
            random_points(:, m) = unifrnd(lower_bound(m), upper_bound(m), num_samples, 1);
        end

        % Step 4: Count points dominated by the Pareto front
        dominated_count = 0;
        for i = 1:num_samples
            point = random_points(i, :);
            % Check if the point is dominated by any point in the Pareto front
            is_dominated = false;
            for j = 1:N
                if all(ParetoFront(j, :) <= point) % For minimization (after adjustment)
                    is_dominated = true;
                    break;
                end
            end
            if is_dominated
                dominated_count = dominated_count + 1;
            end
        end

        % Step 5: Compute the hypervolume
        % Total volume of the bounding box
        box_volume = prod(upper_bound - lower_bound);
        % Proportion of dominated points
        proportion = dominated_count / num_samples;
        % Estimated hypervolume
        hv = proportion * box_volume;

        % Step 6: Validation for high dimensions
        if hv == 0 && N > 0
            warning('Hypervolume estimated as 0 for %s with %d points. Consider increasing num_samples or checking front quality.', method_name, N);
        end
    end

    % Ensure non-negative hypervolume
    hv = max(hv, 0);
end

% Helper Function: Ideal Point Distance (IPD)
function ipd = computeIPD(ParetoFront, ReferenceFront, Type_obj)
    if isempty(ParetoFront) || isempty(ReferenceFront)
        ipd = Inf;
        return;
    end
    ParetoFront = adjust_for_optimization(ParetoFront, Type_obj);
    ReferenceFront = adjust_for_optimization(ReferenceFront, Type_obj);
    ideal_point_front = min(ParetoFront, [], 1);
    ideal_point_ref = min(ReferenceFront, [], 1);
    ipd = norm(ideal_point_ref - ideal_point_front);
end

% Helper Function: Nadir Point Distance (NPD)
function npd = computeNPD(ParetoFront, ReferenceFront, Type_obj)
    if isempty(ParetoFront) || isempty(ReferenceFront)
        npd = Inf;
        return;
    end
    ParetoFront = adjust_for_optimization(ParetoFront, Type_obj);
    ReferenceFront = adjust_for_optimization(ReferenceFront, Type_obj);
    nadir_point_front = max(ParetoFront, [], 1);
    nadir_point_ref = max(ReferenceFront, [], 1);
    npd = norm(nadir_point_ref - nadir_point_front);
end
% Helper Function: Standard Spacing (SP)
function sp = computeStandardSpacing(ParetoFront)
    N = size(ParetoFront, 1);
    if N < 2
        sp = Inf;
        return;
    end
    % Compute distances to nearest neighbor
    min_distances = zeros(N, 1);
    for i = 1:N
        distances = sqrt(sum((ParetoFront - ParetoFront(i, :)).^2, 2));
        distances(i) = Inf; % Exclude distance to itself
        min_distances(i) = min(distances);
    end
    % Compute mean distance
    mean_distance = mean(min_distances);
    % Compute variance of distances
    variance = sum((min_distances - mean_distance).^2) / (N - 1);
    % Standard Spacing is the square root of variance
    sp = sqrt(variance);
    if isnan(sp) || isinf(sp)
        sp = Inf;
    end
end

% Helper Function: Generalized Spread (Updated to follow Deb et al., 2002)
function gs = computeGeneralizedSpread(ParetoFront, ReferenceFront, Type_obj)
    [N_k, p] = size(ParetoFront);
    
    % Case: If N_k < 2, return Inf (no spacing possible)
    if N_k < 2
        gs = Inf;
        return;
    end

    % Adjust for optimization type
    ParetoFront_adj = adjust_for_optimization(ParetoFront, Type_obj);
    ReferenceFront_adj = adjust_for_optimization(ReferenceFront, Type_obj);

    % Sort ParetoFront by first objective
    [~, idx] = sort(ParetoFront_adj(:, 1), 'ascend');
    if strcmpi(Type_obj, 'max')
        idx = flip(idx);
    end
    ParetoFront_adj = ParetoFront_adj(idx, :);

    % Compute consecutive distances d_i
    distances = sqrt(sum(diff(ParetoFront_adj).^2, 2));
    
    % Compute mean distance d_bar
    d_bar = mean(distances);
    
    % Compute sum of absolute deviations
    abs_deviations = sum(abs(distances - d_bar));
    
    % Compute extreme distances d_m^e
    d_e = zeros(p, 1);
    for m = 1:p
        % Find extreme points for objective m in ParetoFront and ReferenceFront
        if strcmpi(Type_obj, 'min')
            min_Pk = min(ParetoFront_adj(:, m));
            max_Pk = max(ParetoFront_adj(:, m));
            min_ref = min(ReferenceFront_adj(:, m));
            max_ref = max(ReferenceFront_adj(:, m));
        else % max
            min_Pk = max(ParetoFront_adj(:, m));
            max_Pk = min(ParetoFront_adj(:, m));
            min_ref = max(ReferenceFront_adj(:, m));
            max_ref = min(ReferenceFront_adj(:, m));
        end
        % Compute distance between extreme points (absolute difference)
        d_e(m) = abs(min_Pk - min_ref) + abs(max_Pk - max_ref);
    end
    sum_d_e = sum(d_e);
    
    % Compute Generalized Spread
    numerator = sum_d_e + abs_deviations;
    denominator = sum_d_e + (N_k - 1) * d_bar;
    if denominator == 0
        gs = Inf;
    else
        gs = numerator / denominator;
    end
end
% Helper Function: Purity Index (Dynamic)
function purity_scores = calculate_purity_dynamic(fronts, ReferenceFront)
    num_valid = length(fronts);
    purity_scores = zeros(1, num_valid);
    for i = 1:num_valid
        purity_scores(i) = purity_metric(fronts{i}, ReferenceFront);
    end
end

% Helper Function: Pareto Front Construction
function F_p = pareto_front(F, Type_obj)
    N = size(F, 1);
    if N <= 1
        F_p = F;
        return;
    end
    is_dominated = false(N, 1);
    for i = 1:N
        for j = 1:N
            if i ~= j
                if dominates(F(j, :), F(i, :), Type_obj)
                    is_dominated(i) = true;
                    break;
                end
            end
        end
    end
    F_p = F(~is_dominated, :);
    F_p = unique(F_p, 'rows', 'stable');
end

% Helper Function: Partial Dominance Check
function is_dom = dominates(p1, p2, Type_obj)
    tolerance = 0.001;
    if strcmpi(Type_obj, 'min')
        is_at_least_as_good = all(p1 <= p2 + tolerance);
        is_strictly_better_all = all(p1 < p2 - tolerance);
        is_dom = is_at_least_as_good && is_strictly_better_all;
    else
        is_at_least_as_good = all(p1 >= p2 - tolerance);
        is_strictly_better_all = all(p1 > p2 + tolerance);
        is_dom = is_at_least_as_good && is_strictly_better_all;
    end
end

% Helper Function: Purity Metric
function purity = purity_metric(F_k, F_p)
    N_k = size(F_k, 1);
    if N_k == 0 || isempty(F_p)
        purity = 0;
        return;
    end
    count = 0;
    epsilon = 0.001;
    for i = 1:N_k
        if any(all(abs(F_p - F_k(i, :)) <= epsilon, 2))
            count = count + 1;
        end
    end
    purity = count / N_k;
end

% Helper Function: Adjust for Optimization Type
function adjusted = adjust_for_optimization(data, Type_obj)
    adjusted = data;
    if strcmpi(Type_obj, 'max')
        adjusted = -data;
    end
end

% Helper Function: Get Extreme Points
function [ref_min, ref_max] = get_extreme_points(ReferenceFront, Type_obj)
    if strcmpi(Type_obj, 'min')
        ref_min = ReferenceFront(find(min(sum((ReferenceFront - min(ReferenceFront,[],1)).^2,2)),1),:);
        ref_max = ReferenceFront(find(min(sum((ReferenceFront - max(ReferenceFront,[],1)).^2,2)),1),:);
    else
        ref_min = ReferenceFront(find(min(sum((ReferenceFront - max(ReferenceFront,[],1)).^2,2)),1),:);
        ref_max = ReferenceFront(find(min(sum((ReferenceFront - min(ReferenceFront,[],1)).^2,2)),1),:);
    end
end

% Helper Function: Display Results
function display_results(indicators, selected_names)
    num_valid = length(selected_names);
    fprintf('----------Performance Indicators Results------------ ');
     fprintf('\n');
    fprintf('Hypervolume (larger is better)\n');
    for i = 1:num_valid
        fprintf('  %s: %.4f\n', selected_names{i}, indicators.hv(i));
    end
    fprintf('\n');
    fprintf('Purity Index (larger is better)\n');
    for i = 1:num_valid
        fprintf('  %s: %.4f\n', selected_names{i}, indicators.purity(i));
    end
    fprintf('\n');
   fprintf('Spacing (smaller is better)\n');
    for i = 1:num_valid
        fprintf('  %s: %.4f\n', selected_names{i}, indicators.spacing(i));
    end
    fprintf('\n');

    fprintf('Generalized Spread (smaller is better)\n');
    for i = 1:num_valid
        fprintf('  %s: %.4f\n', selected_names{i}, indicators.generalized_spread(i));
    end
  end