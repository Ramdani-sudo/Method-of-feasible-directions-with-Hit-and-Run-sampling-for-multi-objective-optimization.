%%MFD_MOROVATI_POURKARIMI - Method of Feasible Directions for solving linearly constrained multi-objective optimization (LC-MOO) problems 
%
% Authors: Ramdani Zoubir*, Addoune Smail, Brahmi Boualem
% Affiliation: Faculty of Mathematics and Computer Sciences,
%              University Mohamed El Bachir El Ibrahimi, Bordj Bou Arreridj, Algeria
% Corresponding Author: Ramdani Zoubir (z.ramdani@univ-bba.dz)
%
% Description:
%   Solves linearly constrained multi-objective optimization (LC-MOO) problems of
%   the form min F(x) subject to Ax <= b. This function generates an approximation
%   of the efficient solution set and the Pareto front using the sampling method
%   from Morovati and Pourkarimi (2019) to create an initial population of feasible
%   points and a multi-objective Armijo line search for optimization.
%   Reference: Morovati, V., Pourkarimi, L.: Extension of Zoutendijk method for
%   solving constrained multiobjective optimization problems. European Journal of
%   Operational Research, 273, 44--57 (2019). https://doi.org/10.1016/j.ejor.2018.08.018
%
% Requirements:
%   - MPT3 Toolbox (https://www.mpt3.org/)
%   - MATLAB 2015 or later
%
% Inputs:
%   funcs_obj - Cell array of objective functions {f1, f2, ..., fm}
%   A         - Constraint matrix (Ax <= b)
%   b         - Right-hand side vector of constraints
%   prec      - Convergence precision for objective improvement
%   prec2     - Tolerance for active constraints
%   max_iter  - Maximum number of iterations
%   sigma     - Armijo line search parameter
%   Type_obj  - Optimization type ('min' or 'max')
%   n_prime   - Number of initial feasible solutions
%   n_d       - Number of random directions for sampling
%
% Outputs:
%   X_eff_IFD    - Matrix of efficient solutions (rows are solutions)
%   Ynd_IFD      - Matrix of non-dominated objective values
%   CPU_Time_IFD - Vector of CPU times per solution
%   NB_iter_IFD  - Vector of iteration counts
%   XS0          - Matrix of initial feasible points from Morovati-Pourkarimi sampling
%
% Example:
% Hanne3 [Collette2013]
%             funcs_obj = {@(x) sqrt(x(1));@(x) sqrt(x(2))};
%             A = [-1 -1; -1 0; 0 -1; 1 0; 0 1]; b = [-5; -0.01; -0.01; 10; 10];
%             LB = [0.1 0.1]; UB = [10 10];
%   [X_eff, Ynd, CPU, Iter, XS0] = MFD_Morovati_Pourkarimi(funcs_obj, A, b, 1e-6, 1e-5, 500, 0.5, 'min', 200, 150);
%
function [X_eff_IFD, Ynd_IFD, CPU_Time_IFD, NB_iter_IFD,XS0] = MFD_Morovati_Pourkarimi(funcs_obj, A, b, prec, prec2,max_iter, sigma, Type_obj,n_prime, n_d)
    % --- Input validation ---
        XS0 = Morovati_Pourkarimi_Sampling(A, b, n_prime, n_d, prec2);
    % Create the polytope using MPT3
    P = Polyhedron('A', A, 'b', b);
    
    % --- Problem dimensions ---
    [nb_const, nb_var] = size(A);
    nb_obj = length(funcs_obj);
    [nXS0, ~] = size(XS0);
    
    % --- Pre-allocate results ---
    X_eff_IFD = [];
    Ynd_IFD = [];
    CPU_Time_IFD = [];
    NB_iter_IFD = [];
    
    % --- Counter for exceeding max_iter ---
    exceed_max_iter_count = 0;
    
    % --- Matrices for quadprog ---
    H1 = [eye(nb_var); zeros(1, nb_var)];
    SQPvect = [zeros(nb_var, 1); 1];
    SQPmat = [H1, zeros(nb_var + 1, 1)];
    
    % --- Options for quadprog ---
    options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
    
    % --- Main loop for each initial solution ---
    for i = 1:nXS0
        tstart = tic;
        x_IFD = XS0(i, :)';
        iter_IFD = 1;
        Funct_IFD = 1;
        
        % Optimization loop
        while abs(Funct_IFD) > prec && iter_IFD <= max_iter
            % Identify active constraints
            active_indices = find(abs(A * x_IFD - b) < prec2);
            A_activ = A(active_indices, :);
            
            % Compute gradients and set up quadratic programming
            gradients_F=compute_gradients(funcs_obj, x_IFD, nb_obj, nb_var);
            gradients_F = [gradients_F; A_activ];
            A_IFD = [gradients_F, -ones(nb_obj + size(A_activ, 1), 1)];
            b_IFD = zeros(nb_obj + size(A_activ, 1), 1);
            
            % Solve quadratic programming problem
            [d_IFD, Funct_IFD, e_IFD] = quadprog(SQPmat, SQPvect, A_IFD, b_IFD, [], [], [], [], [], options);
            
            if e_IFD ~= 1
                warning('Solution %d: quadprog failed with exit flag %d', i, e_IFD);
                break;
            end
            
            if abs(Funct_IFD) <= prec
                % Store results
                X_eff_IFD = [X_eff_IFD; x_IFD'];
                obj_vals = arrayfun(@(k) funcs_obj{k}(x_IFD), 1:nb_obj);
                Ynd_IFD = [Ynd_IFD; obj_vals];
                NB_iter_IFD= [NB_iter_IFD,iter_IFD];
                CPU_Time_IFD=[CPU_Time_IFD, toc(tstart)] ;
                break;
            end
            
            % Update direction and step size
            d_P = d_IFD(1:nb_var);
            non_active_indices = setdiff(1:nb_const, active_indices);
            A_nonactiv = A(non_active_indices, :);
            b_nonactiv = b(non_active_indices);
            
            % Compute step size and update solution
             [lambda_IFD, flag] =Compute_Steep_lambda(x_IFD, d_P, A_nonactiv, b_nonactiv);
           
            if flag == 0
                warning('Solution %d: Invalid lambda at iteration %d', i, iter_IFD);
                break;
            end

            %lambda_IFD=1;
            x1_IFD = x_IFD + lambda_IFD * d_P;
            while ~P.contains(x1_IFD)
                lambda_IFD = lambda_IFD * sigma;
                x1_IFD = x_IFD + lambda_IFD * d_P;
            end
            
            % Armijo condition
            while any(arrayfun(@(k) funcs_obj{k}(x1_IFD) - funcs_obj{k}(x_IFD) - sigma * lambda_IFD * Funct_IFD, 1:nb_obj) > prec2)
                lambda_IFD = lambda_IFD * sigma;
                x1_IFD = x_IFD + lambda_IFD * d_P;
            end
            
            x_IFD = x1_IFD;
            iter_IFD = iter_IFD + 1;
            
            % Check if max_iter is exceeded
            if iter_IFD > max_iter
                exceed_max_iter_count = exceed_max_iter_count + 1;
            end
        end
    end
    
    
    
    % --- Process results ---
    if ~isempty(Ynd_IFD)
        % Remove duplicates
        Ynd_IFD= unique(Ynd_IFD, 'rows', 'stable');
        if strcmp(Type_obj, 'max')
            Ynd_IFD = -Ynd_IFD; % Convert to maximized values for output
        end
        
        % Display results
        fprintf('\n--- Results of the MFD by Morovati and Pourkarimi (%s) ---\n', Type_obj);
        fprintf('CPU Time (s): Min: %.4f, Max: %.4f, Mean: %.4f\n', min(CPU_Time_IFD), max(CPU_Time_IFD), mean(CPU_Time_IFD));
        fprintf('Iterations: Min: %d, Max: %d, Mean: %.4f\n', min(NB_iter_IFD), max(NB_iter_IFD), mean(NB_iter_IFD));
        fprintf('Non-dominated solutions: %d, Efficient solutions: %d (from %d initial)\n', size(Ynd_IFD, 1), size(X_eff_IFD, 1), nXS0);
        % --- Display message about exceeding max_iter ---
    fprintf('Out of %d feasible initial points (XS0), the maximum number of iterations (%d) was exceeded %d times.\n', ...
        nXS0, max_iter, exceed_max_iter_count);
        % --- Visualization ---
        plot_results(A, b, XS0, X_eff_IFD, Ynd_IFD, nb_var, nb_obj, Type_obj);
    else
        warning('No Pareto Front found for Morovati method.');
    end

end

function plot_results(A, b, XS0, X_eff_IFD, Ynd_IFD, nb_var, nb_obj, Type_obj)
        % Plot efficient set and Pareto front
        P = Polyhedron('A', A, 'b', b);
        if nb_var == 2 || nb_var == 3
            try
                %figure;
                plot_polyhedron(A, b);
                title('Efficient Solutions Set by MFD of Morovati and Pourkarimi');
                grid on; hold on;
                if nb_var == 2
                    plot(XS0(1,1), XS0(1,2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
                    scatter(XS0(:,1), XS0(:,2), 20, 'b*', 'MarkerFaceAlpha', 0.3);
                    scatter(X_eff_IFD(:,1), X_eff_IFD(:,2), 40, 'ro', 'filled');
                    xlabel('x_1'); ylabel('x_2');
                    legend('Feasible Set','First Feasible Point x0', 'Initial Feasible Points', 'Efficient Solutions', 'Location', 'best');
                else
                    scatter3(XS0(:,1), XS0(:,2), XS0(:,3), 20, 'b*', 'MarkerFaceAlpha', 0.3);
                    scatter3(X_eff_IFD(:,1), X_eff_IFD(:,2), X_eff_IFD(:,3), 40, 'ro', 'filled');
                    xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
                    view(45, 30);
                    legend('Feasible Set', 'Initial Feasible Points', 'Efficient Solutions', 'Location', 'best');
                end
                hold off;
            catch
            end
        end
        if nb_obj == 2 || nb_obj == 3
            figure;
            hold on;
            if nb_obj == 2
                scatter(Ynd_IFD(:,1), Ynd_IFD(:,2), 40, 'ro', 'filled');
                xlabel('Objective 1'); ylabel('Objective 2');
            else
                scatter3(Ynd_IFD(:,1), Ynd_IFD(:,2), Ynd_IFD(:,3), 40, 'ro', 'filled');
                xlabel('Objective 1'); ylabel('Objective 2'); zlabel('Objective 3');
                view(45, 30);
            end
            title('Pareto Front by MFD of Morovati and Pourkarimi.');
            grid on;
        end
    end
 function gradients = compute_gradients(funcs_obj, x, nb_obj, nb_var)
        h = 1e-6;
        gradients = zeros(nb_obj, nb_var);
        x = x(:)';
        for j = 1:nb_var
            x_plus = x;
            x_minus = x;
            x_plus(j) = x_plus(j) + h;
            x_minus(j) = x_minus(j) - h;
            for k = 1:nb_obj
                gradients(k, j) = (funcs_obj{k}(x_plus') - funcs_obj{k}(x_minus')) / (2 * h);
            end
        end
 end


 %%%%%%%%%%%%%%%%%%%%%%%
 function XS0 = Morovati_Pourkarimi_Sampling(A, b, n_prime, n_d, epsilon,x_init)
    [nb_const, nb_var] = size(A);
    P = Polyhedron('A', A, 'b', b);
    if P.isEmptySet()
        error('Polytope is empty. Check constraints A and b.');
    end
    if ~P.isBounded()
        error('Polytope is unbounded. Solutions may be infinite.');
    end
        % --- Find initial feasible solution ---
    options = optimoptions('linprog', 'Display', 'none');
    [x_init, ~, exitflag] = linprog([], A, b, [], [], [], [], options);
        if exitflag ~= 1
        error('No initial feasible solution found. Check constraints A and b.');
    end
    
    % --- Generate random directions ---
    directions = randn(nb_var, n_d);
    norms = sqrt(sum(directions.^2, 1));
    if any(norms == 0)
        error('Generated zero-norm direction. Try increasing n_d.');
    end
    directions = directions ./ norms; % Normalize directions
    
    % --- Initialize feasible solutions ---
    XS0 = x_init';
    lambda_max = zeros(n_d, 1);
    valid_directions = false(n_d, 1);
    t_u = 3/2; t_l = 2/3;
    
    % --- Generate solutions along random directions ---
    for k = 1:n_d
        d_k = directions(:, k);
        
        % Step 3: Check feasibility of x_init + d_k
        y = x_init + d_k;
        if P.contains(y)
            lambda_k = t_u;
        else
            lambda_k = t_l;
        end
        
        % Step 4: Compute maximum step size
        lambda_max_k = compute_max_step(x_init, d_k, A, b, epsilon);
        if lambda_max_k > 1e-4
            lambda_max(k) = lambda_max_k;
            valid_directions(k) = true;
        else
            continue; % Skip direction if step is too small
        end
        
        % Step 7: Generate feasible solutions along direction
        S_k = generate_solutions_along_direction(x_init, d_k, lambda_max_k, n_prime, A, b, epsilon);
        XS0 = [XS0; S_k];
    end
    
    % --- Ensure n_prime solutions ---
    while size(XS0, 1) < n_prime
        if size(XS0, 1) < 2
            % Trouver un autre point extrême
            [x_new, ~, exitflag] = linprog(-ones(nb_var,1), A, b, [], [], [], [], options);
            if exitflag ~= 1
                error('Cannot generate additional feasible point.');
            end
            XS0 = [XS0; x_new'];
        else
            idx = randperm(size(XS0, 1), 2);
            alpha = rand();
            new_solution = alpha * XS0(idx(1), :) + (1 - alpha) * XS0(idx(2), :);
            if P.contains(new_solution)
                XS0 = [XS0; new_solution];
            end
        end
    end
    
    % --- Select n_prime solutions randomly ---
    num_solutions = size(XS0, 1);
    if num_solutions < n_prime
        warning('Only %d solutions generated, less than n_prime (%d). Returning available solutions.', num_solutions, n_prime);
        XS0 = XS0(1:num_solutions, :);
    else
        XS0 = XS0(randperm(num_solutions, n_prime), :);
    end
XS0=[x_init';XS0];
    % --- Sub-function: Input validation ---
       % --- Sub-function: Compute maximum step size ---
    function lambda_max = compute_max_step(x, d, A, b, epsilon)
        Ad = A * d;
        residuals = b - A * x;
        lambda_candidates = (residuals + epsilon) ./ Ad;
        lambda_candidates = lambda_candidates(Ad > 0);
        if isempty(lambda_candidates)
            lambda_max = 0;
        else
            lambda_max = min(lambda_candidates);
            if lambda_max <= 0
                lambda_max = 0;
            end
        end
    end

    % --- Sub-function: Generate solutions along a direction ---
    function S_k = generate_solutions_along_direction(x_init, d_k, lambda_max_k, n_prime, A, b, epsilon)
        S_k = [];
        M = lambda_max_k;
        t = M / n_prime;
        max_i = floor(lambda_max_k / t);
        for i = 1:max_i
            new_solution = x_init + i * t * d_k;
            if P.contains(new_solution)
                S_k = [S_k; new_solution'];
            end
        end
    end
 end

 function plot_polyhedron(A, b)
[m, n] = size(A);
if length(b) ~= m
    error('Dimensions of A and b do not match.');
end

figure;
if n == 2
    % Compute vertices for 2D polyhedron
    vertices = compute_2d_vertices(A, b);
    if isempty(vertices)
        warning('Unable to compute polyhedron vertices.');
        return;
    end
    
    % Order vertices to form a convex polygon
    k = convhull(vertices(:, 1), vertices(:, 2));
    vertices = vertices(k, :);
    
    % Plot the polyhedron
    fill(vertices(:, 1), vertices(:, 2), 'cyan', 'FaceAlpha', 0.2, 'EdgeColor', 'k', 'LineWidth', 1.5);
    xlabel('x_1'); ylabel('x_2');
    grid on;

elseif n == 3
    % Compute vertices for 3D polyhedron
    vertices = compute_3d_vertices(A, b);
    if isempty(vertices)
        warning('Unable to compute polyhedron vertices.');
        return;
    end
    
    % Compute face triangulation
    K = convhull(vertices(:, 1), vertices(:, 2), vertices(:, 3));
    
    % Plot the polyhedron
    patch('Vertices', vertices, 'Faces', K, 'FaceColor', 'cyan', 'FaceAlpha', 0.2, ...
          'EdgeColor', 'k', 'LineWidth', 1.5);
    xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
    grid on;
    view(3); % 3D view

else
    warning('Visualization not possible in dimension %d. Projecting to 2D...', n);
    % Project to 2D (first two dimensions)
    A_proj = A(:, 1:2);
    vertices = compute_2d_vertices(A_proj, b);
    if isempty(vertices)
        warning('Unable to compute projected vertices.');
        return;
    end
    
    % Order vertices
    k = convhull(vertices(:, 1), vertices(:, 2));
    vertices = vertices(k, :);
    
    % Plot the projection
    fill(vertices(:, 1), vertices(:, 2), 'cyan', 'FaceAlpha', 0.2, 'EdgeColor', 'k', 'LineWidth', 1.5);
    xlabel('x_1'); ylabel('x_2');
    title('2D Projection of Polyhedron');
    grid on;
end

% Internal function to compute 2D vertices
function vertices = compute_2d_vertices(A, b)
    [m, n] = size(A);
    if n ~= 2
        error('compute_2d_vertices only supports 2D polyhedra.');
    end

    vertices = [];
    for i = 1:m
        for j = i+1:m
            % Solve the system A_i x = b_i and A_j x = b_j
            A_eq = [A(i, :); A(j, :)];
            b_eq = [b(i); b(j)];
            if rank(A_eq) == 2 % Check for linearly independent equations
                x = A_eq \ b_eq;
                % Verify if x satisfies all constraints
                if all(A * x <= b + 1e-6)
                    vertices = [vertices; x'];
                end
            end
        end
    end

    % Remove duplicates
    if ~isempty(vertices)
        vertices = unique(vertices, 'rows', 'stable');
    end
end

% Internal function to compute 3D vertices
function vertices = compute_3d_vertices(A, b)
    [m, n] = size(A);
    if n ~= 3
        error('compute_3d_vertices only supports 3D polyhedra.');
    end

    vertices = [];
    for i = 1:m
        for j = i+1:m
            for k = j+1:m
                % Solve the system A_i x = b_i, A_j x = b_j, A_k x = b_k
                A_eq = [A(i, :); A(j, :); A(k, :)];
                b_eq = [b(i); b(j); b(k)];
                if rank(A_eq) == 3 % Check for linearly independent equations
                    x = A_eq \ b_eq;
                    % Verify if x satisfies all constraints
                    if all(A * x <= b + 1e-6)
                        vertices = [vertices; x'];
                    end
                end
            end
        end
    end

    % Remove duplicates
    if ~isempty(vertices)
        vertices = unique(vertices, 'rows', 'stable');
    end
end

end