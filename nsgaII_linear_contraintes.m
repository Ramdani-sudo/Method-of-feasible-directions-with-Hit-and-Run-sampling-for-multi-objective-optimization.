% NSGAII_LINEAR_CONTRAINTES - NSGA-II for Linearly Constrained Multi-Objective Optimization
%
% Authors: Ramdani Zoubir*, Addoune Smail, Brahmi Boualem
% Affiliation: Faculty of Mathematics and Computer Sciences,
%              University Mohamed El Bachir El Ibrahimi, Bordj Bou Arreridj, Algeria
% Corresponding Author: Ramdani Zoubir (z.ramdani@univ-bba.dz)
%
% Description:
%   Solves linearly constrained multi-objective optimization (LC-MOO) problems of
%   the form min F(x) subject to Ax <= b and LB <= x <= UB. This function is an
%   adaptation of the Constrained NSGA-II algorithm by Persson, J. (MATLAB Central
%   File Exchange, 2025, https://www.mathworks.com/matlabcentral/fileexchange/75338-constrained-nsga-ii-with-seeding-enabled),
%   modified to handle linear constraints and include weakly non-dominated solutions
%   using a dominance tolerance of 0.001. It generates an approximation of the efficient
%   solution set and the Pareto front using a genetic algorithm approach with tournament
%   selection, recombination, and mutation, ensuring feasibility within the constrained
%   polytope.
%   Reference: Ramdani, Z., Addoune, S., Brahmi, B.: Method of feasible directions with
%   Hit and Run sampling for solving linearly constrained multi-objective optimization
%   problems. Mathematical Sciences and Applications E-Notes, 13, 1-21 (2025).
%
% Requirements:
%   - MATLAB 2015 or later
%   - No external toolboxes required
%
% Inputs:
%   objFun   - Cell array of objective functions {f1, f2, ..., fm}
%   LB       - Lower bounds for variables (vector of length n)
%   UB       - Upper bounds for variables (vector of length n)
%   A        - Constraint matrix (Ax <= b)
%   b        - Right-hand side vector of constraints
%   popSize  - Population size (positive integer)
%   Type_obj - Optimization type ('min' or 'max')
%
% Outputs:
%   X_eff_NSGA - Matrix of efficient solutions (rows are solutions)
%   Ynd_NSGA   - Matrix of non-dominated objective values (including weakly non-dominated;
%                Ynd_NSGA for minimization, -Ynd_NSGA for maximization)
%
% Example:
%   % Hanne3 [Collette2013]
%   objFun = {@(x) sqrt(x(1)); @(x) sqrt(x(2))};
%   A = [-1 -1; -1 0; 0 -1; 1 0; 0 1]; b = [-5; -0.01; -0.01; 10; 10];
%   LB = [0.1 0.1]; UB = [10 10];
%   popSize = 200;
%   [X_eff, Ynd] = nsgaII_linear_contraintes(objFun, LB, UB, A, b, popSize, 'min');
%

function [X_eff_NSGA, Ynd_NSGA] = nsgaII_linear_contraintes(objFun, LB, UB, A, b, popSize, Type_obj)
    % Paramètres d'optimisation
    genMax = 1000; % Nombre maximum de générations
    kTour = 2;     % Taille du tournoi
    mP = 0.2;      % Probabilité de mutation initiale
    
    % Dimensions
    [nbconst, nVar] = size(A);
    nObj = length(objFun);
    nb_obj = nObj;
    nb_var = nVar;
    
    % Initialisation de la population
    lhs = lhsdesign(popSize, nVar);
    pop = LB + lhs .* (UB - LB);
    
    % Évaluation initiale
    popF = zeros(popSize, nObj);
    popG = zeros(popSize, nbconst);
    for p = 1:popSize
        x = pop(p, :)'; % Transposer en vecteur colonne
        popF(p, :) = arrayfun(@(k) objFun{k}(x), 1:nObj);
        popG(p, :) = (A * pop(p, :)' - b)';
    end
    
    % Classement initial
    [pop, popF, popG] = rankPopConst(pop, popF, popG);
    
    % Boucle principale
    gen = 1;
    while gen < genMax
        gen = gen + 1;
        mP = mP * (1 - gen / genMax); % Réduction progressive de mP
        
        % Sélection des parents
        popPar = tournSel(pop, kTour);
        
        % Recombinaison et mutation
        Q = recombG(popPar);
        Q = mutate(Q, mP, LB, UB);
        Q = insideLimits(Q, LB, UB);
        
        % Évaluation des enfants
        Qf = zeros(popSize, nObj);
        Qg = zeros(popSize, nbconst);
        for p = 1:popSize
            x = Q(p, :)'; % Transposer en vecteur colonne
            Qf(p, :) = arrayfun(@(k) objFun{k}(x), 1:nObj);
            Qg(p, :) = (A * Q(p, :)' - b)';
        end
        
        % Combinaison des populations
        R = [pop; Q];
        Rf = [popF; Qf];
        Rg = [popG; Qg];
        
        % Classement et sélection des meilleurs
        [R, Rf, Rg] = rankPopConst(R, Rf, Rg);
        pop = R(1:popSize, :);
        popF = Rf(1:popSize, :);
        popG = Rg(1:popSize, :);
    end
    
    % Résultats
    X_eff_NSGA = pop;
    Ynd_NSGA = popF;
    Ynd_NSGA = unique(Ynd_NSGA, 'rows');
    % Adjust Ynd_NSGA for max case
    if strcmp(Type_obj, 'max')
        Ynd_NSGA = -Ynd_NSGA;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plot NSGA-II Pareto front (2D or 3D)
    if ~isempty(Ynd_NSGA)
        if nb_obj == 2
            figure;
            scatter(Ynd_NSGA(:,1), Ynd_NSGA(:,2), 'ro', 'filled', 'DisplayName', 'NSGA-II (including weakly non-dominated)');
            xlabel('Objective 1');
            ylabel('Objective 2');
            title('NSGA-II Pareto Front (Weakly Non-Dominated Included)');
            legend('Location', 'best');
            grid on;
        elseif nb_obj == 3
            figure;
            scatter3(Ynd_NSGA(:,1), Ynd_NSGA(:,2), Ynd_NSGA(:,3), 'ro', 'filled', 'DisplayName', 'NSGA-II (including weakly non-dominated)');
            xlabel('Objective 1');
            ylabel('Objective 2');
            zlabel('Objective 3');
            title('NSGA-II Pareto Front (Weakly Non-Dominated Included)');
            legend('Location', 'best');
            grid on;
        end
    else
        disp('No Pareto front found for NSGA-II.');
    end

    % Plot NSGA-II decision space (2D or 3D)
    if (nb_var == 2 || nb_var == 3) && ~isempty(X_eff_NSGA)
%         figure;
%         hold on;
        try
%             P = Polyhedron('A', A, 'b', b);
%             P.plot('color', 'cyan', 'alpha', 0.2, 'linewidth', 1.5, 'DisplayName', 'Feasible Region');
            plot_polyhedron(A, b);
           %title('Efficient Set by NSGA-II Algorithm');
           grid on;
           hold on;
        catch ME
            fprintf('Warning: Failed to plot polyhedron: %s\n', ME.message);
        end
        if nb_var == 2
            scatter(X_eff_NSGA(:,1), X_eff_NSGA(:,2), 'ro', 'filled', 'DisplayName', 'NSGA-II');
            xlabel('x_1');
            ylabel('x_2');
        else
            scatter3(X_eff_NSGA(:,1), X_eff_NSGA(:,2), X_eff_NSGA(:,3), 'ro', 'filled', 'DisplayName', 'NSGA-II');
            xlabel('x_1');
            ylabel('x_2');
            zlabel('x_3');
            view(45, 30);
        end
        title('Efficient Set by NSGA-II Algorithm');
        %legend('show');
        grid on;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function [rankedPop, rankedF, rankedG] = rankPopConst(pop, popF, popG)
    [m, ~] = size(popG);
    normG = normConst(popG);
    [feasIdx, infeasIdx] = discRankConst(normG);
    
    if isempty(feasIdx)
        rankedF = popF(infeasIdx, :);
    else
        popFfeas = popF(feasIdx, :);
        Prank = fastNomDomSort(popFfeas);
        rankedF = crowDistSort(popFfeas, Prank);
        if ~isempty(infeasIdx)
            rankedF = [rankedF; popF(infeasIdx, :)];
        end
    end
    
    idx = findIndices(popF, rankedF);
    rankedPop = pop(idx, :);
    rankedG = popG(idx, :);
end

function normG = normConst(popG)
    [m, n] = size(popG);
    resError = max(popG, zeros(m, n));
    maxError = max(resError);
    maxError(maxError == 0) = 1; % Éviter division par zéro
    normG = resError ./ maxError;
end

function [feasIdx, infeasIdx] = discRankConst(normG)
    [m, ~] = size(normG);
    totG = sum(normG, 2);
    [sortG, idx] = sort(totG);
    countFeas = sum(sortG == 0);
    feasIdx = idx(1:countFeas);
    infeasIdx = idx(countFeas + 1:m);
    if countFeas == 0, feasIdx = []; end
    if countFeas == m, infeasIdx = []; end
end

function myRank = fastNomDomSort(pop)
    [m, ~] = size(pop);
    myRank = zeros(m, 1);
    [Idx, domMat] = findPar(pop);
    myRank(Idx) = 1;
    k = length(Idx);
    rank = 1;
    while k < m
        rank = rank + 1;
        ranked = myRank > 0;
        domMat(ranked, :) = 0;
        dominated = find(sum(domMat, 1) == 1);
        myRank(dominated) = rank;
        k = k + length(dominated);
    end
end

function [Idx, domMat] = findPar(pop)
    [m, n] = size(pop);
    domMat = zeros(m, m);
    Idx = zeros(m, 1);
    i = 0;
    tolerance = 0.001; % Tolerance for weak non-domination
    for p = 1:m
        % Check dominance with tolerance
        is_at_least_as_good = sum(repmat(pop(p, :), m, 1) <= pop + tolerance, 2);
        is_strictly_better = sum(repmat(pop(p, :), m, 1) < pop - tolerance, 2);
        notSameDes = sum(repmat(pop(p, :), m, 1) == pop, 2) ~= n;
        dominates = (is_at_least_as_good == n) & (is_strictly_better == n) & notSameDes;
        domMat(p, :) = dominates';
        
        % Check if p is non-dominated (not strictly dominated by any other solution)
        is_dominated = false;
        for q = 1:m
            if q ~= p
                q_at_least_as_good = all(pop(q, :) <= pop(p, :) + tolerance);
                q_strictly_better = all(pop(q, :) < pop(p, :) - tolerance);
                if q_at_least_as_good && q_strictly_better
                    is_dominated = true;
                    break;
                end
            end
        end
        if ~is_dominated
            i = i + 1;
            Idx(i) = p;
        end
        domMat(p, p) = 1;
    end
    Idx = Idx(1:i);
end

function rankedF = crowDistSort(pop, pRank)
    [m, n] = size(pop);
    [sortRank, idx] = sort(pRank);
    nPf = sortRank(end);
    sortPop = pop(idx, :);
    crowSort = zeros(m, 1);
    counter = 0;
    for i = 1:nPf
        noI = sum(pRank == i);
        tempPop = sortPop(counter + 1:counter + noI, :);
        crowDist = crowDistOneFront(tempPop);
        [crowSort(counter + 1:counter + noI), cidx] = sort(crowDist, 'descend');
        sortPop(counter + 1:counter + noI, :) = tempPop(cidx, :);
        counter = counter + noI;
    end
    rankedF = sortPop;
end

function dist = crowDistOneFront(pop)
    [m, n] = size(pop);
    maxdist = max(max(pop)) - min(min(pop));
    dist = zeros(m, 1);
    for i = 1:n
        [tempV, idx] = sort(pop(:, i));
        nMin = tempV(1);
        nMax = tempV(end);
        dist(idx(1)) = dist(idx(1)) + n * 10 * maxdist;
        dist(idx(end)) = dist(idx(end)) + n * 10 * maxdist;
        for j = 2:m-1
            dist(idx(j)) = dist(idx(j)) + (tempV(j + 1) - tempV(j - 1)) / (nMax - nMin);
        end
    end
end

function idx = findIndices(origPop, sortedPop)
    [m, ~] = size(origPop);
    idx = zeros(m, 1);
    for i = 1:m
        for j = 1:m
            if all(sortedPop(i, :) == origPop(j, :)) && ~any(idx == j)
                idx(i) = j;
                break;
            end
        end
    end
end

function popPar = tournSel(pop, k)
    [m, n] = size(pop);
    popPar = zeros(m, 2 * n);
    for i = 1:m
        candM = min(ceil(rand(k, 1) * m));
        popPar(i, 1:n) = pop(candM, :);
        candF = min(ceil(rand(k, 1) * m));
        if candM == candF
            candF = min(candF + 1, m);
        end
        popPar(i, n + 1:2 * n) = pop(candF, :);
    end
end

function Q = recombG(popPar)
    [m, n] = size(popPar);
    Q = zeros(m, n / 2);
    dist = abs(popPar(:, 1:n/2) - popPar(:, n/2 + 1:n));
    temp = rand(m, n/2) < 0.5;
    norm1 = popPar(:, 1:n/2) + 1./(2 * dist) .* randn(m, n/2);
    norm2 = popPar(:, n/2 + 1:n) + 1./(2 * dist) .* randn(m, n/2);
    Q = temp .* norm1 + (1 - temp) .* norm2;
end

function popMut = mutate(pop, mP, LB, UB)
    [m, n] = size(pop);
    popMut = rand(m, n) .* (UB - LB) + LB;
    mutIdx = rand(m, n) < mP / n;
    popMut = pop .* (1 - mutIdx) + popMut .* mutIdx;
end

function okPop = insideLimits(pop, LB, UB)
    [m, ~] = size(pop);
    okPop = max(pop, LB);
    okPop = min(okPop, UB);
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