%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Nonlinear GMM Estimation                                        %%%%%
%%%%% Joonhwi Joo                                                     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Note. This code requires Matlab R2015b or higher, 
%%% with both ipopt and ADigator v1.3 installed
clear;
clearvars -global;
clc;

%parpool; % A parallel pool should be running
startupadigator;

% id to be fixed during the estimation
fix_id = 7;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 1. Preliminary                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Load and clean the data
estimation_data = readtable('estimation_data_matched_directional.csv','TreatAsEmpty',{'.','NA'});

coordinates = readtable('coordinates_matched.csv','TreatAsEmpty',{'.','NA'});

%%% Extract data
s_ij_id_comb = [estimation_data.i_id estimation_data.j_id];
s_ij = estimation_data.s_ij;
dist = estimation_data.dist;

coord = coordinates(:,{'id'});
coord = table2array(coord);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 3. Initialization                                               %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Find K (all the city locations are known)
K = size(coord, 1);
L = 0;

% N is the current sample size, as well as the possible i,j combination
% size
N = size(s_ij, 1); 

%%% Initialize the values
sigma0 = 10; % (sigma == 1/2 zeta). Note the sign!!
alpha0 = ones((K+L),1);
alpha0 = alpha0 * 100;

%%% Initialize theta0
theta0 = [sigma0;alpha0];
dim_theta = size(theta0,1);

%%% Indices
index.sigma_start = 1;
index.alpha_start = 1 + 1;
index.alpha_end = 1 + K + L';

%%% Save auxdatas
%auxdata.K = K;
%auxdata.L = L;
auxdata.I = (K+L);
auxdata.N = N;
auxdata.id_comb = s_ij_id_comb;
auxdata.s_ij = s_ij;
auxdata.index = index;
auxdata.dim_theta = dim_theta;
auxdata.dist = dist;
auxdata.dist_sq = dist.^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 4. Constraint Formulation                                       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% No constraints to be formulated as all the locations are known

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 5. Optimization - First Stage                                   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sigma should be larger than zero
options.lb(1,1) = 0;
options.ub(1,1) = Inf;

% alpha's should be larger than zero
options.lb(index.alpha_start:index.alpha_end,1) = 0;
options.ub(index.alpha_start:index.alpha_end,1) = Inf;

options.lb((index.alpha_start+fix_id-1),1) = 100;
options.ub((index.alpha_start+fix_id-1),1) = 100;

%%% Setup structure for adigatorGenFiles4Ipopt
setup.numvar = dim_theta;
setup.objective = 'sqerr_sum';
%setup.constraint = 'gmm_constraint';
setup.auxdata = auxdata;
setup.order = 1;

%% adigatorGenFiles4Ipopt generates everything required by ipopt
%tic
%funcs = adigatorGenFiles4Ipopt(setup);
%gentime = toc;

% Functions
funcs.objective = @(theta)sqerr_sum(theta,auxdata);
funcs.gradient = @(theta)sqerr_sum_Grd(theta,auxdata);
%funcs.constraints = @(theta)gmm_constraint(theta,auxdata);
%funcs.jacobian = @(theta)gmm_constraint_jacobian(theta,auxdata);
%funcs.jacobianstructure = @()gmm_constraint_jacobian_structure(auxdata);
%funcs.hessian = @(theta,l_1,l_2)hess_finite_difference(theta,l_1,l_2,auxdata);
%funcs.hessianstructure = @()hess_sparsity(auxdata);

%% ipopt options
%options.ipopt.jac_c_constant = 'no';
%options.ipopt.hessian_constant = 'no';
%options.ipopt.hessian_approximation = 'limited-memory';
%%options.ipopt.hessian_approximation = 'exact';
%options.ipopt.limited_memory_update_type = 'bfgs';
%options.ipopt.limited_memory_max_history = 100;
%options.ipopt.limited_memory_max_skipping = 1;
%options.ipopt.mu_strategy = 'adaptive';
%%options.ipopt.derivative_test = 'first-order';
%options.ipopt.tol = 1e-8;
%options.ipopt.acceptable_tol = 1e-7;
%options.ipopt.acceptable_iter = 100;
%options.ipopt.max_iter = 40000;
%options.ipopt.linear_solver = 'ma57';
%%options.ipopt.fixed_variable_treatment = 'make_parameter';
%
%
%%%%%% Run the optimization
%diary output_ipopt.out
%[result info] = ipopt(theta0, funcs, options);
%diary off
%
%save('workspace_d_step5.mat');
%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% 6. Different Initial Values for the first-stage estimation      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Randomly give around 99% perturbation to each element of theta0
%%%% Set new initial values as the estimates
%%theta0 = result;
%current_max_iter = 10000;
%options.ipopt.max_iter = current_max_iter;
%options.ipopt.hessian_approximation = 'limited-memory';
%
%delete(gcp('nocreate'));
%%p = parpool('torque', 80);
%p = parpool('local',4);
%p.IdleTimeout = Inf;
%
%% Random trial number to find the minimum
%B = 100;
%
%objfunc = zeros(B,1);
%status = zeros(B,1);
%param_matrix = zeros(dim_theta, B);
%
%options.ipopt.print_level = 3; % Message display level (suppress)
%parfor i = 1:B
%    ok_flag_init = 0;
%    
%    %while ok_flag_init < 1
%    perturb = -1 + 2 * rand(dim_theta,1);
%    % 20% of initial value perturbation
%    theta_perturb = theta0 + 0.99 * theta0 .* perturb;    
%           
%        
%    [result info] = ipopt(theta_perturb, funcs, options);
%    objfunc(i,1) = sqerr_sum(result, auxdata);
%    status(i,1) = info.status;
%    
%    param_matrix(:,i) = result;    
%    
%end
%options.ipopt.print_level = 5; % Message display level (to the default)
%save('workspace_d_step6.mat');
%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% 7. First Stage Postestimation                                   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Select out the minimum-objective function estimates
%objfunc_min = [(1:B)' status objfunc];
%
%ok_sign = 0;
%
%%%% If the index is achieved after the max.iter, keep running it to find
%%%% the minimum
%while ok_sign < 1
%    [value, ind] = min(objfunc_min(:,3));
%    if objfunc_min(ind, 2) ~= 0 % 0 is the converged sign
%        [result info] = ipopt(param_matrix(:,ind), funcs, options);
%        objfunc_min(i,3) = sqerr_sum(result, auxdata);
%        objfunc_min(i,2) = info.iter;
%        param_matrix(:,ind) = result; 
%        
%    elseif objfunc_min(ind, 2) == 0
%        theta_firststage = param_matrix(:,ind);
%        ok_sign = 1;
%    end
%end
%ok_sign = 0; % Turn the ok_sign back
%
%%temp = param_matrix(:,3494);
%%csvwrite('theta_firststage_plot_secondmin.csv', temp);
%
%save('workspace_d_step7.mat');
%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% 8. Closed-form Covariance - Derivatives                         %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%load('workspace_step7.mat');
%
%%% Closed-form covariance matrix

theta_firststage = csvread('theta_firststage.csv');
theta_est = theta_firststage;
theta_est(1) = theta_est(1)/2.0;
% Residual vector
e_hat = error_residual(theta_est, auxdata);

%%%% Finite differences jacobian for verification
%current_jacobian_finitediff = zeros(N, dim_theta);
%
%for k = 1:dim_theta
%theta_est_epsilon = theta_est;
%theta_est_epsilon(k) = theta_est_epsilon(k) + 1e-8;
%e_hat_epsilon = error_residual(theta_est_epsilon, auxdata);
%finite_diff = (e_hat_epsilon - e_hat) / 1e-8;
%
%current_jacobian_finitediff(:,k) = finite_diff;
%end


%%% Autodiff jacobian
auxdata.result_selector = 1;

%%% Setup structure for adigatorGenFiles4Ipopt
setup_err.numvar = dim_theta;
setup_err.objective = 'error_residual_k';
setup_err.constraint = 'error_residual';
setup_err.auxdata = auxdata;
setup_err.order = 1;

%funcs_error = adigatorGenFiles4Ipopt(setup_err);

current_jacobian = full(error_residual_Jac(theta_est, auxdata));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 9. Closed-form Covariance - calculation and simulation          %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Find the indices that are fixed during the estimation and not.
%fix_id = 1;

fix_id_var = fix_id + index.alpha_start - 1;

indices_variable = 1:dim_theta;
indices_variable = indices_variable(indices_variable ~= fix_id_var);

% Calculate the covariance matrix
e_hat_sq = e_hat.^2;
nabla_f = current_jacobian(:,indices_variable);

variable_size = size(indices_variable, 2);

D_mat = zeros(variable_size, variable_size);
V_mat = zeros(variable_size, variable_size);

for i = 1:N
    V_mat = V_mat + (nabla_f(i,:)' * nabla_f(i,:)) * e_hat_sq(i);
    D_mat = D_mat + nabla_f(i,:)' * nabla_f(i,:);
    
end

V_mat = V_mat;
D_mat = D_mat;

Sigma_mat_homo = inv(D_mat) * (sum(e_hat_sq) / N);
Sigma_mat_white = inv(D_mat) * V_mat * inv(D_mat);

theta_se_homo = sqrt(diag(Sigma_mat_homo));
theta_se_white = sqrt(diag(Sigma_mat_white));

theta_est_variable = theta_est(indices_variable,:);


dlmwrite('variance_white.csv', Sigma_mat_white, 'precision', 16);
dlmwrite('variance_homo.csv', Sigma_mat_homo, 'precision', 16);
