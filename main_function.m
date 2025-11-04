% MAIN_FUNCTION - 
%
% Authors: Ramdani Zoubir*, Addoune Smail and Brahmi Boualem
% Affiliation: Faculty of Mathematics and Computer Sciences,
%              University Mohamed El Bachir El Ibrahimi of Bordj Bou Arreridj, Algeria
% Corresponding Author: Ramdani Zoubir (z.ramdani@univ-bba.dz)
% Description:
%   This function serves as the main script for solving linearly constrained
%   multi-objective optimization (LC-MOO) problems using three methods:
%   1. Improved Method of Feasible Directions (MFD_Ramdani_et_al)
%   2. MFD of Morovati and Pourkarimi 
%   3. NSGA-II (Genetic Algorithm)
%   It loads problem data, generates feasible solutions, applies the methods,
%   and compares their Pareto fronts using performance metrics (Hypervolume,
%   IGD, Spread, Epsilon, Purity, IPD). The results are displayed in the console
%   and visualized via 2D/3D Pareto front and efficient set plots.
%
% Inputs:
%   None (prompts user for problem index)
%
% Outputs:
%   Console output with problem details and performance metrics
%   Plots of Pareto fronts and efficient sets
%
% Requirements:
%   - MPT3 Toolbox (https://www.mpt3.org/)
%   - MATLAB 2015 or later
%
% Example:
%   main_function()
%
function main_function()
    clc;
%     mpt_init; % Uncomment if MPT3 toolbox is available

    % Configuration parameters
    config = struct(...
        'nb_feas_sol', 200, ... % Number of feasible solutions
        'n_d', 150, ...         % Number of random directions
        'epsilon', 1e-5, ...    % Active constraint tolerance
        'prec', 1e-6, ...       % Convergence precision
        'sigma', 0.5, ...       % Armijo parameter
        'max_iter', 500 ...     % Maximum iterations
    );

    % Prompt user for problem selection
    fprintf('\n============================================================\n');
    fprintf(' Linearly Constrained Multi-Objective Optimization Solver\n');
    fprintf('============================================================\n');
    fprintf('Refer to test_problems.m for available problems.\n');
    problem_idx = input('Enter a problem index: ');

    % Load and validate problem data
    try
        [funcs_obj, A, b, LB, UB, Type_obj, type_problem] = load_problem(problem_idx);
    catch ME
        error('Failed to load problem %d: %s', problem_idx, ME.message);
    end

    % Estimate bounds using MPT3 if not provided
    [LB, UB] = estimate_bounds(A, b, LB, UB);

    % Validate problem data
    validate_problem(funcs_obj, A, b, Type_obj);

    % Display problem details
    display_problem_info(problem_idx, funcs_obj, A, type_problem, Type_obj);

    % Run optimization methods
    results = run_methods(funcs_obj, A, b, LB, UB, Type_obj, config);

    % Compare performance metrics
    compare_results(results, Type_obj);
end

% Helper function: Load problem data
function [funcs_obj, A, b, LB, UB, Type_obj, type_problem] = load_problem(idx)
    try
        [funcs_obj, A, b, LB, UB, Type_obj, type_problem] = test_problems(idx);
    catch
        rethrow(MException('Main:LoadProblem', 'Invalid problem index %d', idx));
    end
end

% Helper function: Estimate bounds using MPT3
function [LB, UB] = estimate_bounds(A, b, LB, UB)
    P = Polyhedron('A', A, 'b', b);
    if isempty(LB)
        LB = min(P.V, [], 1);
    end
    if isempty(UB)
        UB = max(P.V, [], 1);
    end
end

% Helper function: Validate problem data
function validate_problem(funcs_obj, A, b, Type_obj)
    if ~iscell(funcs_obj) || isempty(funcs_obj)
        error('Invalid funcs_obj: Must be a non-empty cell array of objective functions.');
    end
    if ~ismatrix(A) || ~isvector(b) || size(A, 1) ~= length(b)
        error('Invalid A or b: A must be a matrix, b a vector, with matching dimensions.');
    end
    if ~ismember(lower(Type_obj), {'min', 'max'})
        error('Invalid Type_obj: Must be ''min'' or ''max''.');
    end
end

% Helper function: Display problem information
function display_problem_info(idx, funcs_obj, A, type_problem, Type_obj)
    [nb_const, nb_var] = size(A);
    nb_obj = length(funcs_obj);
    fprintf('\n+------------------- Problem Details -------------------+\n');
    fprintf('| Problem Index        | %d\n', idx);
    fprintf('| Variables           | %d\n', nb_var);
    fprintf('| Objectives          | %d\n', nb_obj);
    fprintf('| Constraints         | %d\n', nb_const);
    fprintf('| Problem Type        | %s\n', type_problem);
    fprintf('| Type of objective functions   | %s\n', Type_obj);
    fprintf('+------------------------------------------------------+\n');
end

% Helper function: Run optimization methods
function results = run_methods(funcs_obj, A, b, LB, UB, Type_obj, config)
    results = struct('MFD_Ramdani', [], 'Vahiv', [], 'NSGAII', []);

    % Method 1: MFD_Ramdani_et_al
    fprintf('\n[INFO] Running MFD of Ramdani et al...\n');
    try
        [X_eff, Ynd] = MFD_Ramdani_et_al(funcs_obj, A, b, UB, LB, ...
            config.prec, config.epsilon, config.max_iter, config.sigma, ...
            Type_obj, config.nb_feas_sol);
        results.MFD_Ramdani = struct('X_eff', X_eff, 'Ynd', Ynd);
        fprintf('[SUCCESS] MFD_Ramdani completed.\n');
    catch ME
        warning('MFD_Ramdani failed: %s', ME.message);
        results.MFD_Ramdani = struct('X_eff', [], 'Ynd', []);
    end

    % Method 2: Vahiv (Morovati and Pourkarimi)
    fprintf('\n[INFO] Running MFD of Morovati and Pourkarimi...\n');
    try
        [X_eff, Ynd] = MFD_Morovati_Pourkarimi(funcs_obj, A, b, ...
            config.prec, config.epsilon, config.max_iter, config.sigma, ...
            Type_obj, config.nb_feas_sol, config.n_d);
        results.Vahiv = struct('X_eff', X_eff, 'Ynd', Ynd);
        fprintf('[SUCCESS] MFD_MOROVATI_POURKARIMI completed.\n');
    catch ME
        warning('MFD_MOROVATI_POURKARIMI failed: %s', ME.message);
        results.Vahiv = struct('X_eff', [], 'Ynd', []);
    end

    % Method 3: NSGA-II
    fprintf('\n[INFO] Running NSGA-II...\n');
    try
        [X_eff, Ynd] = nsgaII_linear_contraintes(funcs_obj, LB, UB, A, b, ...
            config.nb_feas_sol, Type_obj);
        results.NSGAII = struct('X_eff', X_eff, 'Ynd', Ynd);
        fprintf('[SUCCESS] NSGA-II completed.\n');
    catch ME
        warning('NSGA-II failed: %s', ME.message);
        results.NSGAII = struct('X_eff', [], 'Ynd', []);
    end
end

% Helper function: Compare performance metrics
function compare_results(results, Type_obj)
    fprintf('\n[INFO] Comparing performance metrics...\n');
    Ynd_MFD_Ramdani = results.MFD_Ramdani.Ynd;
    Ynd_Vahiv = results.Vahiv.Ynd;
    Ynd_NSGAII = results.NSGAII.Ynd;

    if isempty(Ynd_MFD_Ramdani) && isempty(Ynd_Vahiv) && isempty(Ynd_NSGAII)
        warning('All methods returned empty fronts. Skipping comparison.');
        return;
    end

    try
        Comparaison_Methods(Ynd_MFD_Ramdani, Ynd_Vahiv, Ynd_NSGAII, Type_obj);
        fprintf('[SUCCESS] Comparison completed.\n');
    catch ME
        warning('Comparison failed: %s', ME.message);
    end
end