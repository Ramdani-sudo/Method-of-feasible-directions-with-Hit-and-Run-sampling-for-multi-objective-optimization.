% Description:
%%%This Function solves linearly constrained multi-objective optimization problems of the form 
%%% Min F(x) =[f1(x),f2(x),...fp(x)]
%%%%%%subject to   A x \leq b, and LB≤ x≤  UB
% %%Using Hit-and-Run sampling
%  Outcomes are illustrated with 2D/3D plots of Pareto fronts and efficient sets.
%%% test Problems are numbered consecutively from 1 to 27, collected from the literature.
% References: See individual problem comments for citations
% Authors: Ramdani Zoubir*, Addoune Smail, Brahmi Boualem
% Affiliation: Faculty of Mathematics and Computer Sciences,
%              University Mohamed El Bachir El Ibrahimi, Bordj Bou Arreridj, Algeria
% Corresponding Author: Ramdani Zoubir (z.ramdani@univ-bba.dz)
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
function MFD_For_MOOP_Ramdani_et_al
    clc;
%     mpt_init; % Uncomment if MPT3 toolbox is available
  %%Input configuration parameters    
 N=250;%%% Number of feasible solutions
prec=1e-6;% Convergence precision
sigma=0.5;% Armijo parameter
max_iter=500;% Maximum iterations
prec2=1e-5;% Active constraint tolerance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('\n===============================================================================\n');
    fprintf(' Method of feasible directions Linearly For Solving Constrained Multi-Objective Optimization Solver\n');
    fprintf('===============================================================================\n');
    problem_idx = input('Enter a problem index: ');
    [funcs_obj, A, b, LB, UB, Type_obj, type_problem] = test_problems(problem_idx);
 [nb_const, nb_var] = size(A);
    nb_obj = length(funcs_obj);
    fprintf('\n+------------------- Problem Details -------------------+\n');
    fprintf('| Problem Index        | %d\n', problem_idx);
    fprintf('| Variables           | %d\n', nb_var);
    fprintf('| Objectives          | %d\n', nb_obj);
    fprintf('| Constraints         | %d\n', nb_const);
    fprintf('| Type of objective functions   | %s\n', type_problem);
    fprintf('+------------------------------------------------------+\n');
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
%    fprintf('[INFO] Generating %d initial feasible points...\n', N);
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
        fprintf('\n--- CPU Time and the number of iterations to solve the problem (%s) ---\n', Type_obj);
        fprintf('CPU Time (s): Min: %.4f, Max: %.4f, Mean: %.4f\n', min(CPU_Time_IFD), max(CPU_Time_IFD), mean(CPU_Time_IFD));
        fprintf('Iterations: Min: %d, Max: %d, Mean: %.4f\n', min(NB_iter_IFD), max(NB_iter_IFD), mean(NB_iter_IFD));

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
                title('Decision space and Efficient Solutions Sets ');
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
            title('Objective space and Pareto Front ');
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

function [funcs_obj, A, b, LB, UB, Optimization_Type, Objective_Function_Type] = test_problems(r)
    % Initialize default outputs
    funcs_obj = {}; A = []; b = []; LB = []; UB = []; Optimization_Type = 'min'; Objective_Function_Type = 'Nonlinear';

    switch r
        case 1 % Hanne3 [Collette2013]
            funcs_obj = {@(x) sqrt(x(1));@(x) sqrt(x(2))};
            A = [-1 -1; -1 0; 0 -1; 1 0; 0 1]; b = [-5; -0.01; -0.01; 10; 10];
            LB = [0.1 0.1]; UB = [10 10];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Nonlinear';

        case 2 % CONSTR [Elmouaden2018b]
            funcs_obj = {@(x) x(1);@(x) (1+x(2))/x(1)};
            A = [-9 -1; -9 1; 1 0; -1 0; 0 1; 0 -1]; b = [-6; -1; 1; -0.1; 5; 0];
            LB = [0.1 0]; UB = [1 5];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Nonlinear';

        case 3 % KITA [Elmouaden2018b]
            funcs_obj = {@(x) x(1)^2 - x(2);@(x) -(0.5*x(1) + x(2) + 1)};
            A = [1/6 1; 1/2 1; 5 1; -1 0; 0 -1]; b = [13/2; 15/2; 30; 0; 0];
            LB = [0 0]; UB = [10 10];
            Optimization_Type = 'max';
            Objective_Function_Type = 'Nonlinear';

        case 4 % EX004 [Elmouaden2018b]
            funcs_obj = {@(x) x(1) + 2;@(x) exp(x(2)) + 1};
            A = [-2 -3; 2 3; -1 0; 0 -1; 0 1]; b = [-6; 12; 0; 0; 2];
            LB = [0 0]; UB = [6 2];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Nonlinear';

        case 5 % KM [Elmouaden2018b]
            funcs_obj = {@(x) -(x(1) + x(2) - 5);@(x) -(1/5)*(10*x(1) - x(1)^2 + 4*x(2) - x(2)^2 - 11)};
            A = [3 1; 2 1; 1 2; -1 0; 0 -1]; b = [12; 9; 12; 0; 0];
            LB = [0 0]; UB = [10 10];
            Optimization_Type = 'max';
            Objective_Function_Type = 'Nonlinear';

        case 6 % Example 7.1 [Arora2015]
            funcs_obj = {@(x) x(1)^3 + x(2)^2;@(x) -3*x(1) + 2*x(2)^2};
            A = [1 0; -1 0; 0 1; 0 -1]; b = [1; 0; 2; 2];
            LB = [0 -2]; UB = [1 2];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Nonlinear';

        case 7 % Example 7.0 [Arora2015]
            funcs_obj = {@(x) x(1);@(x) 1 + x(2)^2 - x(1) - 0.1*sin(3*pi*x(1))};
            A = [1 0; -1 0; 0 1; 0 -1]; b = [1; 0; 2; 2];
            LB = [0 -2]; UB = [1 2];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Nonlinear';

        case 8 % ABC-comp [Hwang1979]
            funcs_obj = {@(x) -x(1)*x(2);@(x) (x(1)-4)^2 + x(2)^2};
            A = [1 1; -1 0; 0 -1; 1 0; 0 1]; b = [25; 0; 0; 25; 25];
            LB = [0 0]; UB = [25 25];
            Optimization_Type = 'max';
            Objective_Function_Type = 'Nonlinear';

        case 9 % BeeMe [Hwang1979]
            funcs_obj = {@(x) x(1)^2 - x(2)^2;@(x) x(1)/x(2)};
            A = [1 0; 0 1; -1 0; 0 -1]; b = [2; 2; 1; -1];
            LB = [-1 1]; UB = [2 2];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Nonlinear';

        case 10 % Comet [Eichfelder2008]
            funcs_obj = {@(x) (1+x(3))*(x(1)^3*x(2)^2-10*x(1)-4*x(2));...
                         @(x) (1+x(3))*(x(1)^3*x(2)^2-10*x(1)+4*x(2));...
                         @(x) 3*(1+x(3))*x(1)^2};
            A = [eye(3); -eye(3)]; b = [3.5; 2; 1; -1; 2; 0];
            LB = [1 -2 0]; UB = [3.5 2 1];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Nonlinear';

        case 11 % SW1 [Elmouaden2018b]
            funcs_obj = {@(x) 7*x(1)+20*x(2)+9*x(3);@(x) -4*x(1)-5*x(2)-3*x(3);@(x) -x(3)};
            A = [1.5 1 1.6; 1 2 1; 3 -2 4; -eye(3)]; b = [9; 10; 12; 0; 0; 0];
            LB = [0 0 0]; UB = [10 10 10];
            Optimization_Type = 'max';
            Objective_Function_Type = 'MOLP';

        case 12 % SW2 [Elmouaden2018b]
            funcs_obj = {@(x) -9*x(1)-19.5*x(2)-7.5*x(3);@(x) -7*x(1)-20*x(2)-9*x(3);@(x) 4*x(1)+5*x(2)+3*x(3)};
            A = [1.5 1 1.6; 1 2 1; 3 -2 4; -eye(3)]; b = [9; 10; 12; 0; 0; 0];
            LB = [0 0 0]; UB = [10 10 10];
            Optimization_Type = 'max';
            Objective_Function_Type = 'MOLP';

        case 13 % PEKKA2 [Elmouaden2018b]
            funcs_obj = {@(x) -x(1);@(x) -x(2);@(x) -x(3)};
            A = [1 2 2; 2 2 1; 3 -2 4; -eye(3)]; b = [8; 8; 12; 0; 0; 0];
            LB = [0 0 0]; UB = [8 8 8];
            Optimization_Type = 'max';
            Objective_Function_Type = 'MOLP';

        case 14 % Example 2.1 [Armand1991]
            funcs_obj = {@(x) x(1)+x(2)+0.25*x(3);@(x) -x(1)-x(2)-1.5*x(3)};
            A = [1 1 1; 2 2 1; 1 -1 0; -eye(3)]; b = [3; 4; 0; 0; 0; 0];
            LB = [0 0 0]; UB = [3 3 3];
            Optimization_Type = 'max';
            Objective_Function_Type = 'MOLP';

        case 15 % LISWETM
n = 5;
k = 2;
nk = n + k;

% Ensembles (représentés comme des indices)
Ok = 0:k;           % {0, 1, 2}
In = 1:n;           % {1, 2, 3, 4, 5}
Ink = 1:nk;         % {1, 2, 3, 4, 5, 6, 7}

% Coefficients B
B = zeros(1, k+1);  % B_0, B_1, B_2
B(1) = 1;           % B_0 = 1
for i = 1:k
    B(i+1) = B(i) * i; % B_i = B_{i-1} * i
end

% Coefficients C
C = zeros(1, k+1);  % C_0, C_1, C_2
C(1) = 1;           % C_0 = 1
for i = 1:k
    C(i+1) = (-1)^i * B(k+1) / (B(i+1) * B(k-i+1));
end

% Coefficients T
T = zeros(1, nk);
for i = 1:nk
    T(i) = (i-1) / (n+k-1);
end

% % Fonctions objectifs dans une cellule

funcs_obj = { 
    @(x) -0.0842*x(1) - 0.4991*x(2) - 0.5915*x(3) - 0.6314*x(4) - 0.7206*x(5) - 0.8850*x(6) - 1.0657*x(7) + ...
         0.25*x(1)^2 + 0.25*x(2)^2 + 0.25*x(3)^2 + 0.25*x(4)^2 + 0.25*x(5)^2 + 0.25*x(6)^2 + 0.25*x(7)^2;...
    @(x) -0.0842*x(1) - 0.0956*x(2) - 0.0512*x(3) - 0.0493*x(4) - 0.2004*x(5) - 0.5508*x(6) - 1.0657*x(7) + ...
         0.5*x(1)^2 + 0.5*x(2)^2 + 0.5*x(3)^2 + 0.5*x(4)^2 + 0.5*x(5)^2 + 0.5*x(6)^2 + 0.5*x(7)^2 ...
};
% Matrice des contraintes
A = zeros(n, nk);
for j = 1:n
    for i = 0:k
        A(j, j+k-i) = C(i+1); % C_0 x_{j+2} + C_1 x_{j+1} + C_2 x_j
    end
end

% Convertir en -Ax <= 0
A = [-A;-eye(n+2);eye(n+2)];
b= zeros(n, 1);
    LB = zeros(1,n+2); % x_i >= 0
    UB = 10 * ones(1,n+2); % x_i <= 10
    b=[b;LB';UB'];
                 Optimization_Type = 'min';
            Objective_Function_Type = 'Quadratic';

        case 16 % Nabetani2008 [TAREK20091123]
            funcs_obj = {@(x) -(x(1)*x(2)+x(1)-x(1)^2);@(x) -(0.5*x(1)*x(2)+2*x(2)-x(2)^2)};
            A = [1 1; -1 0; 0 -1]; b = [1; 0; 0];
            LB = [0 0]; UB = [1 1];
            Optimization_Type = 'max';
            Objective_Function_Type = 'Nonlinear';

        case 17 % Kathrin2008 [Klamroth2008]
            funcs_obj = {@(x) -x(1) - x(2) + 5;@(x) -(1/5)*(10*x(1) - x(1)^2 + 4*x(2) - x(2)^2) + 2.2;...
                         @(x) -(x(1) - 5)*(x(2) - 11);@(x) 3*x(1) + x(2) - 12;@(x) -x(2)};
            A = [3 1; 2 1; 1 2; -1 0; 0 -1]; b = [12; 9; 12; 0; 0];
            LB = [0 0]; UB = [ 9 9];
            Optimization_Type = 'max';
            Objective_Function_Type = 'Nonlinear';

        case 18 % Wiecek2001 [TAREK20091123]
            funcs_obj = {@(x) 10*(x(1)-2).^4-10*(2-x(1)).^3+10*(x(2)-2).^4-10*(2-x(2)).^3+10;...
                         @(x) (x(1)-3).^2+(x(2)-3).^2+10};
            A = [-1 -1; 1 0; 0 1; -1 0; 0 -1]; b = [-0.1; 10; 10; 0; 0];
            LB = [0 0]; UB = [10 10];
            Optimization_Type = 'max';
            Objective_Function_Type = 'Nonlinear';

        case 19 % Youness2004 [TAREK20091123]
            funcs_obj = {@(x) x(1)^3;@(x) -(x(1)-x(2))^3};
            A = [1 -1; 2 1; -1 0; 0 -1]; b = [0; 3; 0; 0];
            LB = [0 0]; UB = [3 3];
            Optimization_Type = 'max';
            Objective_Function_Type = 'Nonlinear';

        case 20 % MOLPg-1 [Steuer1986]
            funcs_obj = {@(x) 7*x(1)+5*x(2)+4*x(4)-2*x(5)+7*x(6)+6*x(7)+6*x(8);...
                         @(x) -3*x(1)-7*x(2)+8*x(3)+6*x(5)-7*x(6)-3*x(7);...
                         @(x) x(1)+4*x(2)+3*x(4)-x(5)+7*x(6)-4*x(7)-7*x(8)};
            A = [0 8 0 1 -3 7 -1 0; 7 7 6 4 5 0 0 0; 8 0 0 1 8 0 -3 -3; ...
                 8 3 5 0 1 0 8 0; 1 6 3 3 3 3 4 0; 0 5 -1 0 0 0 3 -3; ...
                 7 -2 0 2 8 8 4 -1; 0 5 5 -2 -1 0 0 6; -eye(8)];
            b = [5; 9; 6; 8; 9; 9; 7; 8; zeros(8,1)];
            LB = zeros(1,8); UB = 20*zeros(1,8);
            Optimization_Type = 'min';
            Objective_Function_Type = 'MOLP';

        case 21 % MOLPg-2 [Steuer1986]
            funcs_obj = {@(x) 3*x(1)+2*x(3)+2*x(4)+3*x(5)+x(6)-x(7)+2*x(8)+3*x(9)-x(10)+x(11)+4*x(12);...
                         @(x) -x(1)+4*x(2)+2*x(5)-x(7)+4*x(8)-x(9)-x(11);...
                         @(x) 3*x(2)+2*x(3)+2*x(4)-x(5)+2*x(6)+4*x(7)+x(8)-x(11)+3*x(12)};
            A = [0 2 -1 0 0 0 0 4 2 0 0 0; 0 0 0 0 0 3 0 0 0 0 0 3; -1 4 4 0 0 3 0 0 0 0 -1 0; ...
                 0 4 0 3 0 0 0 0 2 0 0 4; 0 0 0 0 -1 0 4 2 0 5 0 0; 0 0 2 3 0 0 0 0 0 0 0 5; ...
                 2 0 1 0 0 0 0 2 2 -1 2 4; 0 0 0 0 0 0 0 0 2 0 3 0; 0 0 0 -1 0 0 2 1 4 0 0 0; ...
                 4 -1 0 0 0 0 0 0 0 4 0 0; 0 0 0 -1 0 0 0 0 2 5 0 0; 0 0 0 0 0 0 0 0 0 0 2 0; ...
                 -1 -1 0 0 0 0 0 0 0 0 0 0; 2 0 0 3 0 5 0 0 0 0 0 0; 0 2 1 0 3 5 0 0 0 0 0 0; ...
                 0 2 5 0 4 0 3 0 0 0 0 0; -eye(12)];
            b = [10; 10; 10; 7; 10; 5; 5; 6; 5; 7; 6; 7; 9; 5; 9; 5; zeros(12,1)];
            LB = zeros(1,12); UB = 20*ones(1,12);
            Optimization_Type = 'min';
            Objective_Function_Type = 'MOLP';

        case 22 % MOLPg-3 [Steuer1986]
            funcs_obj = {@(x) x(1)-x(3)+x(4)+2*x(5)+3*x(6)-2*x(7)-x(9)+3*x(10);...
                         @(x) x(1)+x(3)+2*x(4)+3*x(9);...
                         @(x) 2*x(1)+3*x(2)-x(4)-x(6)+3*x(7)-x(8)-2*x(9)+3*x(10)};
            A = [1 0 0 0 0 0 4 0 0 3; 0 -1 0 1 0 0 0 5 0 0; 2 5 4 0 2 0 -1 0 0 0; ...
                 0 -1 2 0 0 4 0 0 5 0; 0 0 1 1 0 0 3 0 0 0; 0 2 0 0 0 0 0 1 -1 3; ...
                 0 0 0 0 0 2 0 0 0 0; 1 0 5 0 1 0 0 0 -1 0; 0 0 0 0 0 0 0 0 5 0; ...
                 3 4 0 0 4 0 0 -1 0 3; 0 3 4 0 0 0 0 3 0 0; 0 0 0 3 0 2 0 0 0 5; ...
                 5 0 1 0 0 0 0 0 0 0; -1 2 0 5 0 0 0 2 1 0; -eye(10)];
            b = [10; 8; 7; 10; 8; 8; 5; 9; 7; 9; 7; 7; 7; 7; zeros(10,1)];
            LB = zeros(1,10); UB = 10*ones(1,10);
            Optimization_Type = 'min';
            Objective_Function_Type = 'MOLP';
case 23 % MOQP-1 [Leyffer2025]
            H1 = [0.023; -0.93; 1.9; 2.1; 5.5; -3.7; -4.7; 0.78; 5.6; -5.7; 4.4; -5.7; 0.23; -3.7; 2.6; -3; 5.2; -4.4; 0.26; 4.7];
            H2 = [5.3; -2; -0.75; -0.35; -4.2; -4.4; 0.39; 2.7; -1.2; -1.7; -2.6; 4.4; 1.5; -3.1; 5.7; 1.7; -3.2; 2.2; 2; -4.4];
            H3 = [-5.7; -2.9; -4.6; -5.2; 4.2; -3.8; -5.6; 2.8; 0.44; -2.7; -1.6; -5.8; 4.7; 4.4; -2.9; 0.83; -4.1; 1.1; -2; 1.9];
            G1 = zeros(20, 20);
            G1(1,1)=2.3; G1(2,2)=2.3; G1(3,3)=2.3; G1(4,4)=2.3; G1(5,5)=2.3; G1(6,6)=2.3; G1(14,6)=-1.1; 
            G1(7,7)=2.3; G1(8,8)=2.3; G1(9,9)=2.3; G1(16,9)=1.4; G1(10,10)=2.3; G1(11,11)=2.3; G1(12,12)=2.3; 
            G1(13,13)=2.3; G1(6,14)=-1.1; G1(14,14)=2.3; G1(16,14)=-1.6; G1(15,15)=2.3; G1(18,15)=0.26; 
            G1(9,16)=1.4; G1(14,16)=-1.6; G1(16,16)=2.3; G1(17,17)=2.3; G1(15,18)=0.26; G1(18,18)=2.3; 
            G1(19,19)=2.3; G1(20,20)=2.3;
            G2 = zeros(20, 20);
            G2(1,1)=1.2; G2(10,1)=0.53; G2(2,2)=1.2; G2(19,2)=0.22; G2(3,3)=1.2; G2(4,4)=1.2; G2(5,5)=1.2; 
            G2(7,5)=-0.81; G2(18,5)=-0.92; G2(6,6)=1.2; G2(5,7)=-0.81; G2(7,7)=1.2; G2(8,8)=1.2; G2(9,9)=1.2; 
            G2(1,10)=0.53; G2(10,10)=1.2; G2(11,11)=1.2; G2(12,12)=1.2; G2(13,13)=1.2; G2(14,14)=1.2; 
            G2(15,15)=1.2; G2(16,16)=1.2; G2(17,17)=1.2; G2(5,18)=-0.92; G2(18,18)=1.2; G2(2,19)=0.22; 
            G2(19,19)=1.2; G2(20,20)=1.2;
            G3 = zeros(20, 20);
            G3(1,1)=2.4; G3(2,2)=2.4; G3(3,3)=2.4; G3(4,4)=2.4; G3(10,4)=0.61; G3(5,5)=2.4; G3(6,6)=2.4; 
            G3(7,7)=2.4; G3(8,8)=2.4; G3(14,8)=-0.059; G3(9,9)=2.4; G3(4,10)=0.61; G3(10,10)=2.4; 
            G3(11,11)=2.4; G3(18,11)=-1; G3(12,12)=2.4; G3(13,13)=2.4; G3(8,14)=-0.059; G3(14,14)=2.4; 
            G3(15,15)=2.4; G3(16,16)=2.4; G3(17,17)=2.4; G3(11,18)=-1; G3(18,18)=2.4; G3(20,18)=-2.2; 
            G3(19,19)=2.4; G3(18,20)=-2.2; G3(20,20)=2.4;
            funcs_obj = {@(x) 0.5*x'*G1*x + H1'*x; @(x) 0.5*x'*G2*x + H2'*x; @(x) 0.5*x'*G3*x + H3'*x};
            A = zeros(10, 20);
            A(7,1)=-0.23; A(8,1)=1.3; A(6,2)=2.1; A(8,2)=1.4; A(4,3)=0.95; A(9,4)=1.9; A(2,6)=2.2; 
            A(2,7)=-3.2; A(5,7)=2.1; A(7,7)=-2.4; A(2,9)=-4; A(4,10)=-3.9; A(7,10)=0.64; A(10,10)=0.031; 
            A(2,11)=0.33; A(5,11)=-0.95; A(2,12)=-3.9; A(4,12)=3.1; A(5,12)=-1.4; A(1,13)=0.49; 
            A(5,13)=0.033; A(6,13)=2.2; A(3,14)=2.3; A(4,14)=2.1; A(6,14)=-0.13; A(8,14)=3.5; A(2,15)=-0.39; 
            A(2,17)=-2.4; A(4,17)=3.3; A(9,17)=2.9; A(10,17)=1; A(5,18)=0.52; A(8,18)=2.2; A(6,19)=2.4; 
            A(9,19)=3.9; A(1,20)=1.2;
            b = [1.7; -11; 2.3; 1.4; 0.31; 1.8; -2; 1.1; 3.7; 1.1];
            LB = zeros(1, 20);
            UB = 10*ones(1, 20);
            A = [A; -eye(20); eye(20)];
            b = [b; LB'; UB'];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Quadratic';

        case 24 % MOQP-2 [Leyffer2025]
            H1 = [3.5; -3.7; 4.9; 5.1; -5.8; 3.2; 5.4; 3.8; 5.1; -3.6; 2.1; 5.1; -1.9; 1.1; 1.4; -6; 5.8; 4.8; 2.3; -0.72];
            H2 = [2.4; 1.3; -2.4; 4.3; -4.7; -2.5; -4.8; -1.2; -2; 5.3; 4.1; -2.9; -5.5; -5.9; 0.89; 2.9; 3.7; 1.7; -3; -4.3];
            H3 = [1.8; 5.4; 3.8; 5.2; -2.3; -2.8; 0.44; -4; -3.5; -3.4; 1.8; -5.4; -3.2; 2; -2.3; -2.3; 2.6; 5.5; -4.4; -5.2];
            G1 = zeros(20, 20);
            G1(1,1)=1.7; G1(7,1)=-0.64; G1(2,2)=1.7; G1(3,3)=1.7; G1(4,4)=1.7; G1(5,5)=1.7; G1(11,5)=1.7; 
            G1(6,6)=1.7; G1(12,6)=0.51; G1(1,7)=-0.64; G1(7,7)=1.7; G1(8,8)=1.7; G1(9,9)=1.7; G1(10,10)=1.7; 
            G1(5,11)=1.7; G1(11,11)=1.7; G1(6,12)=0.51; G1(12,12)=1.7; G1(13,13)=1.7; G1(14,14)=1.7; 
            G1(19,14)=0.59; G1(15,15)=1.7; G1(16,16)=1.7; G1(17,17)=1.7; G1(18,18)=1.7; G1(14,19)=0.59; 
            G1(19,19)=1.7; G1(20,20)=1.7;
            G2 = zeros(20, 20);
            G2(1,1)=1; G2(2,2)=1; G2(3,3)=1; G2(14,3)=-0.048; G2(4,4)=1; G2(5,5)=1; G2(19,5)=0.38; 
            G2(6,6)=1; G2(12,6)=-0.02; G2(7,7)=1; G2(9,7)=-1; G2(8,8)=1; G2(7,9)=-1; G2(9,9)=1; 
            G2(10,10)=1; G2(11,11)=1; G2(6,12)=-0.02; G2(12,12)=1; G2(13,13)=1; G2(14,14)=1; 
            G2(15,15)=1; G2(16,16)=1; G2(17,17)=1; G2(18,18)=1; G2(5,19)=0.38; G2(19,19)=1; 
            G2(20,20)=1;
            G3 = zeros(20, 20);
            G3(1,1)=2.2; G3(2,2)=2.2; G3(3,3)=2.2; G3(4,4)=2.2; G3(20,4)=4.3e-05; G3(5,5)=2.2; 
            G3(6,5)=-0.32; G3(5,6)=-0.32; G3(6,6)=2.2; G3(7,7)=2.2; G3(8,8)=2.2; G3(9,9)=2.2; 
            G3(10,10)=2.2; G3(11,10)=-1.9; G3(12,10)=1.1; G3(10,11)=-1.9; G3(11,11)=2.2; 
            G3(10,12)=1.1; G3(12,12)=2.2; G3(13,13)=2.2; G3(14,14)=2.2; G3(15,15)=2.2; 
            G3(16,16)=2.2; G3(17,17)=2.2; G3(18,18)=2.2; G3(19,19)=2.2; G3(4,20)=4.3e-05; 
            G3(20,20)=2.2;
            funcs_obj = {@(x) 0.5*x'*G1*x + H1'*x; @(x) 0.5*x'*G2*x + H2'*x; @(x) 0.5*x'*G3*x + H3'*x};
            A = zeros(10, 20);
            A(3,1)=-0.82; A(8,1)=2.8; A(10,1)=2.5; A(1,2)=-3; A(4,3)=1.7; A(9,3)=-0.1; A(9,4)=1; 
            A(10,3)=1.1; A(3,4)=1.2; A(7,4)=3.8; A(8,4)=2.4; A(10,4)=-1.5; A(2,5)=3.1; A(8,5)=1.3; 
            A(1,6)=-3.9; A(2,6)=0.75; A(3,7)=-3.3; A(8,7)=1.4; A(2,8)=-2.7; A(6,8)=-2.4; A(7,8)=1.1; 
            A(3,10)=2.2; A(4,10)=2.3; A(2,11)=-1.5; A(6,13)=-1.9; A(1,15)=-1; A(10,15)=1.3; 
            A(4,16)=-2.1; A(10,17)=-1.1; A(2,18)=-2.1; A(3,18)=3.8; A(6,18)=1.7; A(1,19)=1.6; 
            A(8,19)=2.6; A(2,20)=-3.9; A(7,20)=0.37; A(8,20)=3.8;
            b = [-6.4; -6.4; 0.73; 1.9; 0; -2.6; 5.3; 0.55; -0.1; 1.7];
            LB = zeros(1, 20);
            UB = 10*ones(1, 20);
            A = [A; -eye(20); eye(20)];
            b = [b; LB'; UB'];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Quadratic';

        case 25 % MOQP-3 [Leyffer2025]
            H1 = [-1; -4.9; -0.6; 4.4; -1.3; -3; -1.7; 2.9; 1.8; 5.3; 4; -0.36; 1.6; -5.3; 0.51; -0.53; 4.4; 4.3; -0.33; 3.4];
            H2 = [1.9; -6; -4.4; -0.062; -5.5; -3.3; -2.1; 4.8; -2.2; -3; -0.8; 4.1; -3.8; 0.098; -0.57; -2.1; -1.4; 4.6; 3.1; 4.6];
            H3 = [-0.51; 3.6; -4.4; -5.2; -1.5; -1.5; -0.19; 5.6; -1.9; -3; 1; 0.28; -4; -0.16; -0.047; 4.1; 3.7; 4.3; 1.3; 0.79];
            G1 = zeros(20, 20);
            G1(1,1)=1.2; G1(2,1)=0.73; G1(13,1)=0.9; G1(1,2)=0.73; G1(2,2)=1.2; G1(3,3)=1.2; 
            G1(4,4)=1.2; G1(6,4)=0.43; G1(5,5)=1.2; G1(4,6)=0.43; G1(6,6)=1.2; G1(7,7)=1.2; 
            G1(9,7)=0.58; G1(8,8)=1.2; G1(7,9)=0.58; G1(9,9)=1.2; G1(10,10)=1.2; G1(11,11)=1.2; 
            G1(12,12)=1.2; G1(1,13)=0.9; G1(13,13)=1.2; G1(14,14)=1.2; G1(15,15)=1.2; G1(16,16)=1.2; 
            G1(17,17)=1.2; G1(18,18)=1.2; G1(19,19)=1.2; G1(20,20)=1.2;
            G2 = zeros(20, 20);
            G2(1,1)=0.72; G2(20,1)=0.57; G2(2,2)=0.72; G2(3,3)=0.72; G2(4,4)=0.72; G2(5,5)=0.72; 
            G2(6,6)=0.72; G2(7,7)=0.72; G2(8,8)=0.72; G2(9,9)=0.72; G2(10,10)=0.72; G2(11,11)=0.72; 
            G2(12,12)=0.72; G2(13,13)=0.72; G2(14,14)=0.72; G2(15,14)=-0.26; G2(14,15)=-0.26; 
            G2(15,15)=0.72; G2(18,15)=0.68; G2(16,16)=0.72; G2(17,17)=0.76; G2(15,18)=0.68; 
            G2(18,18)=0.72; G2(19,19)=0.72; G2(1,20)=0.57; G2(20,20)=0.72;
            G3 = zeros(20, 20);
            G3(1,1)=1.5; G3(2,2)=1.5; G3(15,2)=-0.38; G3(3,3)=1.5; G3(4,4)=1.5; G3(7,4)=-0.23; 
            G3(5,5)=1.5; G3(6,6)=1.5; G3(4,7)=-0.23; G3(7,7)=1.5; G3(15,7)=-0.3; G3(8,8)=1.5; 
            G3(9,9)=1.5; G3(10,10)=1.5; G3(11,11)=1.5; G3(12,12)=1.5; G3(13,13)=1.5; G3(14,14)=1.5; 
            G3(2,15)=-0.38; G3(7,15)=-0.3; G3(15,15)=1.5; G3(16,16)=1.5; G3(17,17)=1.5; 
            G3(18,18)=1.5; G3(20,18)=-1.5; G3(19,19)=1.5; G3(18,20)=-1.5; G3(20,20)=1.5;
            funcs_obj = {@(x) 0.5*x'*G1*x + H1'*x; @(x) 0.5*x'*G2*x + H2'*x; @(x) 0.5*x'*G3*x + H3'*x};
            A = zeros(10, 20);
            A(1,1)=-3.3; A(6,1)=-3; A(2,2)=1.9; A(1,3)=1.9; A(6,3)=-3.7; A(7,3)=-3; A(2,4)=1.5; 
            A(4,4)=-1.9; A(6,4)=-1; A(7,6)=-0.13; A(8,7)=2.2; A(1,8)=-4; A(3,8)=3.8; A(6,8)=1.5; 
            A(7,8)=3.6; A(9,8)=-2.6; A(1,9)=0.82; A(3,9)=-2.4; A(10,9)=-1.9; A(5,10)=-0.79; 
            A(10,10)=-0.34; A(8,11)=-1.6; A(9,11)=1.5; A(4,12)=0.12; A(1,14)=3.7; A(1,15)=-0.82; 
            A(6,16)=3.5; A(7,16)=-1.1; A(10,16)=2.7; A(5,17)=-0.11; A(3,18)=0.75; A(3,19)=3.6; 
            A(6,19)=-0.18; A(4,20)=1.1; A(5,20)=2; A(7,20)=-1.4;
            b = [-1.7; 3.3; 5.8; -0.71; 1.1; -2.8; -2; 0.56; -1.1; 0.34];
            LB = zeros(1, 20);
            UB = 10*ones(1, 20);
            A = [A; -eye(20); eye(20)];
            b = [b; LB'; UB'];
            Optimization_Type = 'min';
            Objective_Function_Type = 'Quadratic';
        case 26 
            funcs_obj={@(x) x(1)^2+x(2)^2;@(x) (x(1)-5)^2+(x(2)-5)^2};
    A = [1 0; 0 1;-1 0; 0 -1];
    b = [ 5;5;5;5];
    LB=[-5 -5];
    UB=[5 5];
    Optimization_Type = 'min';
            Objective_Function_Type = 'Quadratic';
        case 27 
    funcs_obj={@(x) (x(1)-5)^2+(x(2)-3)^2;@(x) (x(1)+4)^2+(x(2)-3)^2};
    A = [ -2 -1; 1 -1; 1 0; 0 1; -1 0; 0 -1; -5 4;1 1];
    b = [ -4; 5; 8; 8; 0; 0; 30;12];
A=[2 3;-1 3;2 -1;-eye(2)];
b = [ 18; 9;10; 0; 0];
 LB=[0 0];
      UB=[8 8];
Optimization_Type = 'min';
            Objective_Function_Type = 'Quadratic';
        otherwise
            error('Problem %d is not defined. Please choose r between 1-25.', r);
    end
end
%%%%References
% [1] M. El Moudden, A. El Ghali, A new reduced gradient method for solving linearly constrained
% multiobjective optimization problems, Comput. Optim. Appl. 71 (2018) 719–741. https://doi.org/10.1007/s10589-018-0023-1.

% [2] Y. Collette, P. Siarry, Multiobjective optimization: Principles and case studies, Springer Science &
% Business Media (2013).

% [3] C.-L. Hwang, A. S. Md. Masud, Multiple Objective Decision Making — Methods and Applications
% (A State-of-the-Art Survey), Springer Berlin, Heidelberg (1979). https://doi.org/10.1007/978-3-642-45511-7. 

% [4] R. K. Arora, OPTIMIZATION: Algorithms and Applications, CRC Press, Taylor & Francis Group(2015). 

%  [5] R. E. Steuer, MULTIPLE CRITERIA OPTIMIZATION: Theory, Computation and Applications,John Wiley & Sons (1986).

%  [6] P. Armand, C. Malivert, Determination of the Efficient Set in Multiobjective Linear Programming,J. Optim. Theory Appl. 70, no. 3 (1991) 35–47.

% [7] G. Eichfelder, Adaptive Scalarization Methods in Multiobjective Optimization, Springer Berlin, Heidelberg (2008). https://doi.org/10.1007/978-3-540-79159-1.

% [8]  E. Tarek, Method of centers algorithm for multi-objective programming problems, Acta Mathematica Scientia 29, no. 5 (2009) 1128–1142. https://doi.org/10.1016/S0252-9602(09)545 60091-6.

% [9]   K. Klamroth, K. Miettinen, Integrating approximation and interactive decision making in multicriteria optimization, Oper. Res. 56 (2008) 222–234. http://dx.doi.org/10.1287/opre.1070. 0425.

% [10] S. Leyffer, Multiobjective Test problems, 2025. https://wiki.mcs.anl.gov/leyffer/index. php/MacMOOP.


end
