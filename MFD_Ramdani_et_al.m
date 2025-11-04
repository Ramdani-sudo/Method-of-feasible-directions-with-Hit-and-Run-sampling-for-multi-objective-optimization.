% MFD_RAMDANI_ET_AL - Improved Method of Feasible Directions for solving linearly constrained multi-objective optimization (LC-MOO) problems 
%
% Authors: Ramdani Zoubir*, Addoune Smail, Brahmi Boualem
% Affiliation: Faculty of Mathematics and Computer Sciences,
%              University Mohamed El Bachir El Ibrahimi, Bordj Bou Arreridj, Algeria
% Corresponding Author: Ramdani Zoubir (z.ramdani@univ-bba.dz)
%
% Description:
%   Solves linearly constrained multi-objective optimization (LC-MOO) problems of
%   the form min F(x) subject to Ax <= b. This function generates an approximation
%   of the efficient solution set and the Pareto front using Hit-and-Run sampling
%   to create an initial population of feasible points.
%
% Requirements:
%   - MPT3 Toolbox (https://www.mpt3.org/)
%   - MATLAB 2015 or later
%
% Inputs:
%   funcs_obj - Cell array of objective functions {f1, f2, ..., fm}
%   A         - Constraint matrix (Ax <= b)
%   b         - Right-hand side vector of constraints
%   UB        - Upper bounds for variables
%   LB        - Lower bounds for variables
%   prec      - Convergence precision for objective improvement
%   prec2     - Tolerance for active constraints
%   max_iter  - Maximum number of iterations
%   sigma     - Armijo line search parameter
%   Type_obj  - Optimization type ('min' or 'max')
%   N         - Number of initial feasible solutions
%
% Outputs:
%   X_eff_IFD    - Matrix of efficient solutions (rows are solutions)
%   Ynd_IFD      - Matrix of non-dominated objective values
%   CPU_Time_IFD - Vector of CPU times per solution
%   NB_iter_IFD  - Vector of iteration counts
%   XS0          - Matrix of initial feasible points from Hit-and-Run
%
% Example:

function [X_eff_IFD, Ynd_IFD, CPU_Time_IFD, NB_iter_IFD, XS0] = MFD_Ramdani_et_al(funcs_obj, A, b, UB, LB, prec, prec2, max_iter, sigma, Type_obj, N)
    % --- Input Validation ---
    if ~iscell(funcs_obj) || isempty(funcs_obj)
        error('Invalid funcs_obj: Must be a non-empty cell array of objective functions.');
    end
    if ~ismatrix(A) || ~isvector(b) || size(A, 1) ~= length(b)
        error('Invalid A or b: A must be a matrix, b a vector, with matching dimensions.');
    end
    if ~ismember(lower(Type_obj), {'min', 'max'})
        error('Invalid Type_obj: Must be ''min'' or ''max''.');
    end

    % --- Initialization ---
    [nb_const, nb_var] = size(A);
    nb_obj = length(funcs_obj);
    X_eff_IFD = [];
    Ynd_IFD = [];
    CPU_Time_IFD = [];
    NB_iter_IFD = [];
    exceed_max_iter_count = 0;

    % Setup quadratic programming
    options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
    H1 = [eye(nb_var); zeros(1, nb_var)];
    SQPmat = [H1, zeros(nb_var + 1, 1)];
    SQPvect = [zeros(nb_var, 1); 1];

    % --- Generate Initial Feasible Points ---
    fprintf('[INFO] Generating %d initial feasible points...\n', N);
    XS0 = Hit_and_Run_Sampling(A, b, N, UB, LB);
    [nXS0, ~] = size(XS0);

    % --- Main Loop: Process Each Initial Point ---
    for i = 1:nXS0
        tstart = tic;
        iter_IFD = 1;
        x_IFD = XS0(i, :)';
        Funct_IFD = 1;

        while abs(Funct_IFD) > prec && iter_IFD <= max_iter
            % Compute gradients
            gradients_F = compute_gradients(funcs_obj, x_IFD, nb_obj, nb_var);

            % Setup constraints for quadratic programming
            A_IFD = [gradients_F, -ones(nb_obj, 1); A, zeros(nb_const, 1)];
            b_IFD = [zeros(nb_obj, 1); -(A * x_IFD - b)];

            % Solve quadratic programming subproblem
            [d_IFD, Funct_IFD, e_IFD] = quadprog(SQPmat, SQPvect, A_IFD, b_IFD, [], [], [], [], [], options);

            if e_IFD ~= 1
                break;
            end

            if abs(Funct_IFD) <= prec
                % Store results
                X_eff_IFD = [X_eff_IFD; x_IFD'];
                vv = cellfun(@(f) f(x_IFD), funcs_obj);
                Ynd_IFD = [Ynd_IFD; vv'];
                NB_iter_IFD = [NB_iter_IFD; iter_IFD];
                CPU_Time_IFD = [CPU_Time_IFD; toc(tstart)];
                break;
            end

            % Identify active and non-active constraints
            active_indices = abs(A * x_IFD - b) < prec2;
            A_nonactiv = A(~active_indices, :);
            b_nonactiv = b(~active_indices);
            d_P = d_IFD(1:nb_var);

            % Compute step size (lambda)
            [lambda_IFD, flag] = compute_lambda(x_IFD, d_P, A_nonactiv, b_nonactiv);
            if ~flag
                break;
            end

            % Armijo line search
            x1_IFD = x_IFD + lambda_IFD * d_P;
            obj_vals = cellfun(@(f) f(x_IFD), funcs_obj);
            obj_vals_new = cellfun(@(f) f(x1_IFD), funcs_obj);
            while any(obj_vals_new - obj_vals - sigma * lambda_IFD * Funct_IFD > prec2)
                lambda_IFD = lambda_IFD * sigma;
                x1_IFD = x_IFD + lambda_IFD * d_P;
                obj_vals_new = cellfun(@(f) f(x1_IFD), funcs_obj);
            end

            x_IFD = x1_IFD;
            iter_IFD = iter_IFD + 1;

            if iter_IFD > max_iter
                exceed_max_iter_count = exceed_max_iter_count + 1;
            end
        end
    end

    % --- Process Results ---
    if any(Ynd_IFD)
        % Remove duplicates
        [Ynd_IFD, idx] = unique(Ynd_IFD, 'rows', 'stable');
        X_eff_IFD = X_eff_IFD(idx, :);
        NB_iter_IFD = NB_iter_IFD(idx);
        CPU_Time_IFD = CPU_Time_IFD(idx);

        % Adjust for maximization
        if strcmp(Type_obj, 'max')
            Ynd_IFD = -Ynd_IFD;
        end

        % Display results
        fprintf('\n--- Results of the MFD by Ramdani et al. (%s) ---\n', Type_obj);
        fprintf('CPU Time (s): Min: %.4f, Max: %.4f, Mean: %.4f\n', min(CPU_Time_IFD), max(CPU_Time_IFD), mean(CPU_Time_IFD));
        fprintf('Iterations: Min: %d, Max: %d, Mean: %.4f\n', min(NB_iter_IFD), max(NB_iter_IFD), mean(NB_iter_IFD));
        fprintf('Non-dominated solutions: %d, Efficient solutions: %d (from %d initial)\n', size(Ynd_IFD, 1), size(X_eff_IFD, 1), nXS0);
        fprintf('Out of %d feasible initial points (XS0), the maximum number of iterations (%d) was exceeded %d times.\n', ...
                nXS0, max_iter, exceed_max_iter_count);

        % Plot results
        plot_results(A, b, XS0, X_eff_IFD, Ynd_IFD, nb_var, nb_obj, Type_obj);
    else
        fprintf('No non-dominated solutions found.\n');
    end

    % --- Sub-functions ---

    function gradients = compute_gradients(funcs_obj, x, nb_obj, nb_var)
        % Compute numerical gradients (vectorized)
        h = 1e-6;
        gradients = zeros(nb_obj, nb_var);
        x = x(:)';
        for j = 1:nb_var
            x_plus = x; x_plus(j) = x_plus(j) + h;
            x_minus = x; x_minus(j) = x_minus(j) - h;
            for k = 1:nb_obj
                gradients(k, j) = (funcs_obj{k}(x_plus') - funcs_obj{k}(x_minus')) / (2 * h);
            end
        end
    end

    function [lambda, flag] = compute_lambda(x, d, A, b)
        % Compute step size (lambda)
        lambda = 1;
        flag = 1;
        if ~isempty(A)
            Ad = A * d;
            indices = Ad > 0;
            if any(indices)
                lambda = min((b(indices) - A(indices, :) * x) ./ Ad(indices));
                if lambda <= 0
                    lambda = 0;
                    flag = 0;
                end
            end
        end
    end

   function plot_results(A, b, XS0, X_eff_IFD, Ynd_IFD, nb_var, nb_obj, Type_obj)
        % Plot efficient set and Pareto front
        P = Polyhedron('A', A, 'b', b);
        if nb_var == 2 || nb_var == 3
            try
                %figure;
                plot_polyhedron(A, b);
                title('Efficient Solutions Set by MFD of Ramdani et al.');
                grid on; hold on;
                if nb_var == 2
                    plot(XS0(1,1), XS0(1,2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
                    scatter(XS0(:,1), XS0(:,2), 20, 'b*', 'MarkerFaceAlpha', 0.3);
                    scatter(X_eff_IFD(:,1), X_eff_IFD(:,2), 40, 'ro', 'filled');
                    xlabel('x_1'); ylabel('x_2');
                    legend('Feasible Set','Chebyshev Center', 'Initial Feasible Points', 'Efficient Solutions', 'Location', 'best');
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
            title('Pareto Front by MFD of Ramdani et al.');
            grid on;
        end
    end
    function plot_polyhedron(A, b)
        % Plot the polyhedron defined by Ax <= b
        [m, n] = size(A);
        if length(b) ~= m
            error('Dimensions of A and b do not match.');
        end
        figure;
        if n == 2
            vertices = compute_2d_vertices(A, b);
            if isempty(vertices)
                warning('Unable to compute polyhedron vertices.');
                return;
            end
            k = convhull(vertices(:, 1), vertices(:, 2));
            vertices = vertices(k, :);
            fill(vertices(:, 1), vertices(:, 2), 'cyan', 'FaceAlpha', 0.2, 'EdgeColor', 'k', 'LineWidth', 1.5);
            xlabel('x_1'); ylabel('x_2');
            grid on;
        elseif n == 3
            vertices = compute_3d_vertices(A, b);
            if isempty(vertices)
                warning('Unable to compute polyhedron vertices.');
                return;
            end
            K = convhull(vertices(:, 1), vertices(:, 2), vertices(:, 3));
            patch('Vertices', vertices, 'Faces', K, 'FaceColor', 'cyan', 'FaceAlpha', 0.2, ...
                  'EdgeColor', 'k', 'LineWidth', 1.5);
            xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
            grid on; view(3);
        else
            warning('Visualization not possible in dimension %d. Projecting to 2D...', n);
            A_proj = A(:, 1:2);
            vertices = compute_2d_vertices(A_proj, b);
            if isempty(vertices)
                warning('Unable to compute projected vertices.');
                return;
            end
            k = convhull(vertices(:, 1), vertices(:, 2));
            vertices = vertices(k, :);
            fill(vertices(:, 1), vertices(:, 2), 'cyan', 'FaceAlpha', 0.2, 'EdgeColor', 'k', 'LineWidth', 1.5);
            xlabel('x_1'); ylabel('x_2');
            title('2D Projection of Polyhedron');
            grid on;
        end
    end

    function vertices = compute_2d_vertices(A, b)
        % Compute vertices for 2D polyhedron
        [m, n] = size(A);
        if n ~= 2
            error('compute_2d_vertices only supports 2D polyhedra.');
        end
        vertices = [];
        for i = 1:m
            for j = i+1:m
                A_eq = [A(i, :); A(j, :)];
                b_eq = [b(i); b(j)];
                if rank(A_eq) == 2
                    x = A_eq \ b_eq;
                    if all(A * x <= b + 1e-6)
                        vertices = [vertices; x'];
                    end
                end
            end
        end
        if ~isempty(vertices)
            vertices = unique(vertices, 'rows', 'stable');
        end
    end

    function vertices = compute_3d_vertices(A, b)
        % Compute vertices for 3D polyhedron
        [m, n] = size(A);
        if n ~= 3
            error('compute_3d_vertices only supports 3D polyhedra.');
        end
        vertices = [];
        for i = 1:m
            for j = i+1:m
                for k = j+1:m
                    A_eq = [A(i, :); A(j, :); A(k, :)];
                    b_eq = [b(i); b(j); b(k)];
                    if rank(A_eq) == 3
                        x = A_eq \ b_eq;
                        if all(A * x <= b + 1e-6)
                            vertices = [vertices; x'];
                        end
                    end
                end
            end
        end
        if ~isempty(vertices)
            vertices = unique(vertices, 'rows', 'stable');
        end
    end

    function XS0 = Hit_and_Run_Sampling(A, b, N, UB, LB)
        % Generate N feasible points using Hit-and-Run sampling
        theta = 0.75;
        [nb_const, nb_var] = size(A);
        P = Polyhedron('A', A, 'b', b);
        if P.isEmptySet()
            error('The polytope is empty. Please check your constraints A and b.');
        end
        if ~P.isBounded()
            error('The polytope is unbounded. Solutions may be infinite.');
        end
        data = P.chebyCenter();
        x0 = data.x;
        volume_P = monte_carlo_polyedre_volume(A, b, 100000, LB, UB);
        rho = theta * 2 * ((volume_P / N * gamma(nb_var / 2 + 1)) / (pi^(nb_var / 2)))^(1 / nb_var);
        XS0 = x0';
        i = 2;
        while i <= N
            direction = randn(nb_var, 1);
            direction = direction / norm(direction);
            lambda_min = -10000;
            lambda_max = 10000;
            for j = 1:nb_const
                dot_prod = A(j, :) * direction;
                if dot_prod > 0
                    lambda = (b(j) - A(j, :) * x0) / dot_prod;
                    lambda_max = min(lambda_max, lambda);
                elseif dot_prod < 0
                    lambda = (b(j) - A(j, :) * x0) / dot_prod;
                    lambda_min = max(lambda_min, lambda);
                end
            end
            lambda = lambda_min + (lambda_max - lambda_min) * rand;
            x_new = x0 + lambda * direction;
            if P.contains(x_new)
                if all(vecnorm(XS0(1:i-1, :) - x_new', 2, 2) >= rho)
                    XS0(i, :) = x_new;
                    i = i + 1;
                end
            end
        end
    end

    function volume = monte_carlo_polyedre_volume(A, b, N, LB, UB)
        % Estimate polytope volume using Monte Carlo method
        P = Polyhedron('A', A, 'b', b);
        [~, n] = size(A);
        bbox = P.outerApprox();
        if isempty(LB)
            LB = bbox.Internal.lb;
        end
        if isempty(UB)
            UB = bbox.Internal.ub;
        end
        vol_B = prod(UB - LB);
        if vol_B <= 0 || isinf(vol_B)
            error('Invalid bounding box: Check LB and UB.');
        end
        points_random = rand(N, n) .* (UB - LB) + LB;
        satisfied = all(A * points_random' <= b, 1);
        num_points_inside = sum(satisfied);
        volume = (num_points_inside / N) * vol_B;
    end
end