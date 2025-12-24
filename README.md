# Multi-Objective Optimization Solver using Method of Feasible Directions and Hit-and-Run Sampling

This repository contains a MATLAB implementation of a solver for linearly constrained multi-objective optimization problems (MOOPs). The solver minimizes (or maximizes) multiple objective functions subject to linear constraints \( Ax \leq b \) and bounds \( LB \leq x \leq UB \). It uses the Method of Feasible Directions (MFD) combined with Hit-and-Run sampling for generating initial feasible points. The outcomes include efficient solutions, non-dominated points (Pareto front), CPU times, and iteration counts, with visualizations in 2D/3D.

The method is designed for problems of the form:
\[
\min F(x) = [f_1(x), f_2(x), \dots, f_p(x)]
\]
subject to \( Ax \leq b \), \( LB \leq x \leq UB \).

## Authors
- Ramdani Zoubir* (Corresponding Author: z.ramdani@univ-bba.dz)
- Addoune Smail
- Brahmi Boualem

**Affiliation:** Faculty of Mathematics and Computer Sciences, University Mohamed El Bachir El Ibrahimi, Bordj Bou Arreridj, Algeria.

## Requirements
- MATLAB 2015 or later
- MPT3 Toolbox (Multi-Parametric Toolbox 3) for polyhedron operations and Chebyshev center computation. Download from [https://www.mpt3.org/](https://www.mpt3.org/). Uncomment `mpt_init;` in the code if installed.
- No additional toolboxes required beyond standard MATLAB Optimization Toolbox for `quadprog`.

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Add the repository folder to your MATLAB path:
   ```matlab
   addpath('/path/to/your-repo-name');
   ```
3. Install MPT3 Toolbox if not already done:
   - Follow instructions on the MPT3 website.
   - Ensure it's on your MATLAB path.

## Usage
Run the main function `MFD_For_MOOP_Ramdani_et_al` in MATLAB. It prompts for a problem index (1-27) and solves the selected test problem.

### Example
```matlab
MFD_For_MOOP_Ramdani_et_al;
```
- Enter a problem index when prompted (e.g., 1 for Hanne3).
- The script will:
  - Generate initial feasible points using Hit-and-Run sampling.
  - Solve using MFD.
  - Display problem details, CPU times, iterations.
  - Plot the feasible set, efficient solutions, and Pareto front (in 2D/3D if applicable).

### Custom Problems
To solve a custom MOOP:
1. Define your objective functions as a cell array (e.g., `funcs_obj = {@(x) x(1)^2, @(x) (x(1)-5)^2};`).
2. Set constraints `A`, `b`, bounds `LB`, `UB`, and type `'min'` or `'max'`.
3. Modify the script to use your inputs instead of `test_problems`.
4. Adjust parameters like `N=250` (initial points), `prec=1e-6` (convergence), `max_iter=500`, etc.

### Inputs (Configurable Parameters)
- `N`: Number of initial feasible solutions (default: 250).
- `prec`: Convergence precision for objective improvement (default: 1e-6).
- `prec2`: Tolerance for active constraints (default: 1e-5).
- `max_iter`: Maximum iterations per solution (default: 500).
- `sigma`: Armijo line search parameter (default: 0.5).
- `Type_obj`: Optimization type (`'min'` or `'max'`).
- For custom runs: `funcs_obj` (cell array of function handles), `A`, `b`, `LB`, `UB`.

### Outputs
- `X_eff_IFD`: Matrix of efficient solutions (rows are solutions).
- `Ynd_IFD`: Matrix of non-dominated objective values (Pareto front).
- `CPU_Time_IFD`: Vector of CPU times per solution.
- `NB_iter_IFD`: Vector of iteration counts per solution.
- `XS0`: Matrix of initial feasible points from Hit-and-Run.

## Test Problems
The function `test_problems(problem_idx)` provides 27 benchmark problems from the literature:
- 1-10: Nonlinear problems (e.g., Hanne3, CONSTR, KITA).
- 11-14: Multi-Objective Linear Programming (MOLP) (e.g., SW1, PEKKA2).
- 15-19: Quadratic and nonlinear (e.g., LISWETM, Comet).
- 20-22: Larger MOLP instances (e.g., MOLPg-1 with 8 variables).
- 23-25: Multi-Objective Quadratic Programming (MOQP) with 20 variables.
- 26-27: Simple quadratic examples.

Each problem includes specific objectives, constraints, bounds, and type. References are in the code comments.

## Plots and Visualization
- For 2D/3D variable spaces: Plots the feasible polyhedron, initial points, and efficient set.
- For 2D/3D objective spaces: Plots the Pareto front.
- Uses `Polyhedron` from MPT3 for plotting.

## Limitations
- Assumes convex feasible sets (polytope must be bounded and non-empty).
- Numerical gradients are used (finite differences).
- For maximization, objectives are negated internally.
- If MPT3 is unavailable, plotting may fail (comment out `plot_results` if needed).

## References
See the code comments for full citations. Key ones:
1. M. El Moudden, A. El Ghali, A new reduced gradient method for solving linearly constrained multiobjective optimization problems, Comput. Optim. Appl. 71 (2018) 719–741.
2. Y. Collette, P. Siarry, Multiobjective optimization: Principles and case studies, Springer (2013).
3. C.-L. Hwang, A. S. Md. Masud, Multiple Objective Decision Making — Methods and Applications, Springer (1979).
4. R. K. Arora, OPTIMIZATION: Algorithms and Applications, CRC Press (2015).
5. R. E. Steuer, MULTIPLE CRITERIA OPTIMIZATION: Theory, Computation and Applications, Wiley (1986).
6. Additional references for specific problems (e.g., Eichfelder2008, Klamroth2008, Leyffer2025).

## License
This code is released under the MIT License. See [LICENSE](LICENSE) file for details.

If you use this code in your research, please cite the authors and relevant references. For questions, contact the corresponding author.
