%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Nonlinear GMM Estimation                                        %%%%%
%%%%% Joonhwi Joo                                                     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Note. This code requires Matlab R2015b or higher, 
%%% with both ipopt and ADigator v1.3 installed

%%% Note. Make sure to adjust the parallel pool setups in section 6

%%% Note. This code does not impose the dynamic constraints by default.
%%% They may be imposed by adjusting the relevant parts of the code.

clear;
clearvars -global;
clc;

startupadigator; % Load ADiGator module

static_constraint_impose = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 1. Preliminary                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Load and clean the data
estimation_data = readtable('ppml_estimation_nondirectional_data.csv','TreatAsEmpty',{'.','NA'});

coordinates = readtable('coordinates.csv','TreatAsEmpty',{'.','NA'});

constraints_static = readtable('constraints_static.csv', ...
    'TreatAsEmpty',{'.','NA'});
%constraints_dynamic = readtable('constraints_dynamic.csv', ...
%    'TreatAsEmpty',{'.','NA'});

%%% Coordinates lower and upper bound in degrees
varphi_lowerbound = 27;
varphi_upperbound = 45;
lambda_lowerbound = 36;
lambda_upperbound = 42;

%%% Bounds in radians, default is to comment out.
%varphi_lowerbound = varphi_lowerbound * pi / 180;
%varphi_upperbound = varphi_upperbound * pi / 180;
%lambda_lowerbound = lambda_lowerbound * pi / 180;
%lambda_upperbound = lambda_upperbound * pi / 180;


%%% Convert radian to degree
% Use only if using Euclidean distance
coordinates.long_x = coordinates.long_x * (180 / pi);
coordinates.lat_y = coordinates.lat_y * (180 / pi);


%%% Extract data
% Shares
s_ij_id_comb = [estimation_data.i_id estimation_data.j_id];
s_ij = estimation_data.s_ij;
s_ji = estimation_data.s_ji;

% Coordinates
coord = coordinates(:,{'id', 'cert', 'long_x', 'lat_y', 'validity'});
coord = table2array(coord);
validity = coord(:,5);

coord_unknown = coord((coord(:,2) > 2),:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 3. Initialization                                               %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Find K and L
% K is the number of known cities. L is unknown cities.
K = sum(validity);
K = K(1,1);
L = size(coord, 1) - K;
% N is the current sample size, as well as the possible i,j combination
% size
N = size(s_ij, 1); 

%%% Initialize the values
sigma0 = 1; % (sigma == 1/2 zeta). Note the sign!!
tilde_delta0 = 2; % tilde_delta0 is obsolete in this code. Just set it > 0

%%% Initialize lambda and varphi at the original locations
varphi0 = coord(:,3); % varphi is the longitude_x
lambda0 = coord(:,4); % lambda is the latitude_y

varphi_known = varphi0(1:K);
lambda_known = lambda0(1:K);

alpha0 = ones((K+L),1);

%%% Initialize theta0
theta0 = [sigma0;tilde_delta0;varphi0;lambda0;alpha0];
dim_theta = size(theta0,1);

%%% Indices
index.sigma_start = 1;
index.tilde_delta_start = 1 + 1;
index.varphi_known_start = 1 + 1 + 1;
index.varphi_known_end = 1 + 1 + K;
index.varphi_unknown_start = 1 + 1 + K + 1;
index.varphi_unknown_end = 1 + 1 + K + L;
index.lambda_known_start = 1 + 1 + K + L + 1;
index.lambda_known_end = 1 + 1 + K + L + K;
index.lambda_unknown_start = 1 + 1 + K + L + K + 1;
index.lambda_unknown_end = 1 + 1 + K + L + K + L;
index.alpha_start = 1 + 1 + K + L + K + L + 1;
index.alpha_end = 1 + 1 + K + L + K + L + K + L;


%%% Data selector vector
data_selector = [1:size(s_ij_id_comb,1)]';

%%% Save auxdatas to be passed on to the optimizer
%auxdata.K = K;
%auxdata.L = L;
auxdata.I = (K+L);
auxdata.N = N;
auxdata.id_comb = s_ij_id_comb;
auxdata.s_ij = s_ij;
auxdata.index = index;
auxdata.dim_theta = dim_theta;
auxdata.data_selector = data_selector;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 4. Constraint Formulation                                       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Static constraints
%%% This part of the code sets the bounds for the varphi's and lambda's of
%%% each city to be estimated. 

%%% leave_ind is the variable whether it is to be estimated
constraints_static.leave_ind = ismember(constraints_static.id, coord_unknown(:,1));
constraints_static = constraints_static(constraints_static.leave_ind == 1,:);

%%% Default bound is what is set above
ub_varphi_id = constraints_static.ub_varphi;
lb_varphi_id = constraints_static.lb_varphi;
ub_lambda_id = constraints_static.ub_lambda;
lb_lambda_id = constraints_static.lb_lambda;

ub_varphi = zeros(L,1);
lb_varphi = zeros(L,1);
ub_lambda = zeros(L,1);
lb_lambda = zeros(L,1);

%%% Overwrite the default bounds if they are specified by the constraints
%%% data
if static_constraint_impose == 1
for i = 1:L
    if isfinite(ub_varphi_id(i)) == 0
        ub_varphi(i) = varphi_upperbound;
    else
        ub_varphi(i) = coordinates.long_x(ub_varphi_id(i));
    end
    
    if isfinite(lb_varphi_id(i)) == 0
        lb_varphi(i) = varphi_lowerbound;
    else
        lb_varphi(i) = coordinates.long_x(lb_varphi_id(i));
    end
    
    if isfinite(ub_lambda_id(i)) == 0
        ub_lambda(i) = lambda_upperbound;
    else
        ub_lambda(i) = coordinates.lat_y(ub_lambda_id(i));
    end
    
    if isfinite(lb_lambda_id(i)) == 0
        lb_lambda(i) = lambda_lowerbound;
    else
        lb_lambda(i) = coordinates.lat_y(lb_lambda_id(i));
    end
end

else

ub_varphi = varphi_upperbound;
lb_varphi = varphi_lowerbound;
ub_lambda = lambda_upperbound;
lb_lambda = lambda_lowerbound;

end

%%%%% Dynamic constraints are not imposed in this version of estimation
%%%%% code

%%%%% Dynamic constraints
%const_dim = size(constraints_dynamic,1);

%jacobian_sparsity = sparse(zeros(const_dim, dim_theta));
%jacobian_sparsity(1,(index.varphi_known_start+18-1)) = 1;
%jacobian_sparsity(1,(index.varphi_known_start+23-1)) = 1;
%jacobian_sparsity(2,(index.lambda_known_start+19-1)) = 1;
%jacobian_sparsity(2,(index.lambda_known_start+27-1)) = 1;

%jacobian = jacobian_sparsity;
%jacobian(1,(index.varphi_known_start+23-1)) = -1;
%jacobian(2,(index.lambda_known_start+27-1)) = -1;

%auxdata.jacobian = jacobian;
%auxdata.jacobian_sparsity = jacobian_sparsity;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 5. Optimization - First Stage                                   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Dynamic constraint lower bound and upper bound
%options.cl = zeros(const_dim,1);
%options.cu = Inf(const_dim,1);

% Theta lower bound and upper bound
options.lb = zeros(dim_theta, 1);
options.ub = zeros(dim_theta, 1);

options.lb(1:dim_theta, 1) = -Inf;
options.ub(1:dim_theta, 1) = Inf;

% sigma should be larger than zero
options.lb(1, 1) = 0;

% tilde_delta should be larger than zero
% Note. This will not affect the objective function nor the estimates anyway
options.lb(2, 1) = 0;

%%% Fix delta
options.lb(2, 1) = 0;
options.ub(2, 1) = 0;

% alpha's should be larger than zero
options.lb(index.alpha_start:index.alpha_end,1) = 0;

options.lb((index.alpha_start+2),1) = 100;
options.ub((index.alpha_start+2),1) = 100;

% varphi and lambda lower and upper bounds for unknown
options.lb(index.varphi_unknown_start:index.varphi_unknown_end,1) = lb_varphi;
options.ub(index.varphi_unknown_start:index.varphi_unknown_end,1) = ub_varphi;
options.lb(index.lambda_unknown_start:index.lambda_unknown_end,1) = lb_lambda;
options.ub(index.lambda_unknown_start:index.lambda_unknown_end,1) = ub_lambda;

% Fix the known varphi's and lambda's; they are data
options.lb(index.varphi_known_start:index.varphi_known_end,1) = varphi_known;
options.ub(index.varphi_known_start:index.varphi_known_end,1) = varphi_known;
options.lb(index.lambda_known_start:index.lambda_known_end,1) = lambda_known;
options.ub(index.lambda_known_start:index.lambda_known_end,1) = lambda_known;


%%% Setup structure for adigatorGenFiles4Ipopt
setup.numvar = dim_theta;
setup.objective = 'sqerr_sum';
setup.auxdata = auxdata;
setup.order = 1;

% adigatorGenFiles4Ipopt generates everything required by ipopt
% This part usually takes about a min
tic
funcs = adigatorGenFiles4Ipopt(setup);
gentime = toc;

% Functions
funcs.objective = @(theta)sqerr_sum(theta,auxdata);
funcs.gradient = @(theta)sqerr_sum_Grd(theta,auxdata);

% ipopt options
options.ipopt.jac_c_constant = 'no';
options.ipopt.hessian_constant = 'no';
options.ipopt.hessian_approximation = 'limited-memory';
%options.ipopt.hessian_approximation = 'exact';
options.ipopt.limited_memory_update_type = 'bfgs';
options.ipopt.limited_memory_max_history = 100;
options.ipopt.limited_memory_max_skipping = 1;
options.ipopt.mu_strategy = 'adaptive';
%options.ipopt.derivative_test = 'first-order';
options.ipopt.tol = 1e-8;
options.ipopt.acceptable_tol = 1e-7;
options.ipopt.acceptable_iter = 100;
options.ipopt.max_iter = 40000;
options.ipopt.linear_solver = 'ma57';
%options.ipopt.fixed_variable_treatment = 'make_parameter';


%%%%% Run the initial optimization
diary output_ipopt.out
[result info] = ipopt(theta0, funcs, options);
diary off

save('workspace_step5.mat');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 6. Different Initial Values for the first-stage estimation      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Randomly give around 20% perturbation to each element of theta0
%%% Set new initial values as the estimates
current_max_iter = 10000;
options.ipopt.max_iter = current_max_iter;
options.ipopt.hessian_approximation = 'limited-memory';

%%%%% Parallel pool setting. 
%%% Make sure it has a proper parallel pool setup to your environment
delete(gcp('nocreate'));
%p = parpool('torque', 80);
p = parpool('local',56);
p.IdleTimeout = Inf;

% Random trial number to find the minimum. Make sure you try at least 20000
B = 20000;

objfunc = zeros(B,1);
status = zeros(B,1);
param_matrix = zeros(dim_theta, B);

options.ipopt.print_level = 3; % Message display level (suppress)
parfor i = 1:B
    ok_flag_init = 0;
    
    
    perturb = -1 + 2 * rand(dim_theta,1);
    % 20% of initial value perturbation
    theta_perturb = theta0 + 0.2 * theta0 .* perturb;    
           
        
    [result info] = ipopt(theta_perturb, funcs, options);
    objfunc(i,1) = sqerr_sum(result, auxdata);
    status(i,1) = info.status;
    
    param_matrix(:,i) = result;    
    
end
options.ipopt.print_level = 5; % Message display level (to the default)
save('workspace_step6.mat');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 7. First Stage Postestimation                                   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Select out the minimum-objective function estimates
objfunc_min = [(1:B)' status objfunc];

ok_sign = 0;

%%% If the index is achieved after the max.iter, keep running it to find
%%% the minimum
while ok_sign < 1
    [value, ind] = min(objfunc_min(:,3));
    if objfunc_min(ind, 2) ~= 0 % 0 is the converged sign
        [result info] = ipopt(param_matrix(:,ind), funcs, options);
        objfunc_min(i,3) = sqerr_sum(result, auxdata);
        objfunc_min(i,2) = info.iter;
        param_matrix(:,ind) = result; 
        
    elseif objfunc_min(ind, 2) == 0
        theta_firststage = param_matrix(:,ind);
        ok_sign = 1;
    end
end
ok_sign = 0; % Turn the ok_sign back

%%% Formulate the table of estimated coordinates
coordinates_estimated = array2table(coord, 'VariableNames', ...
    {'id', 'cert', 'long_x', 'lat_y', 'validity'});

coordinates_estimated.long_x_est = ...
    theta_firststage(index.varphi_known_start:index.varphi_unknown_end);
coordinates_estimated.lat_y_est = ...
    theta_firststage(index.lambda_known_start:index.lambda_unknown_end);

%%% Save as the first-stage estimates (w/o SE)
writetable(struct2table(index), 'plot_data_index.csv');
writetable(coordinates_estimated, 'first_stage_result.csv');
csvwrite('theta_firststage_plot.csv', theta_firststage);

save('workspace_step7.mat');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 8. Closed-form Covariance - Derivatives                         %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Closed-form covariance matrix for the estimates
theta_est = theta_firststage;
% Residual vector
e_hat = error_residual(theta_est, auxdata);

%%% Finite differences jacobian for verification (obsolete)
%current_jacobian_finitediff = zeros(N, dim_theta);

%for k = 1:dim_theta
%theta_est_epsilon = theta_est;
%theta_est_epsilon(k) = theta_est_epsilon(k) + 1e-8;
%e_hat_epsilon = error_residual(theta_est_epsilon, auxdata);
%finite_diff = (e_hat_epsilon - e_hat) / 1e-8;

%current_jacobian_finitediff(:,k) = finite_diff;
%end



%%% Autodiff jacobian, use ADiGator again
auxdata.result_selector = 1;

%%% Setup structure for adigatorGenFiles4Ipopt
setup_err.numvar = dim_theta;
setup_err.objective = 'error_residual_k';
setup_err.constraint = 'error_residual';
setup_err.auxdata = auxdata;
setup_err.order = 1;

funcs_error = adigatorGenFiles4Ipopt(setup_err);

current_jacobian = full(error_residual_Jac(theta_est, auxdata));

save('workspace_closedform_step8.mat');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 9. Closed-form Covariance - calculation and simulation          %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Find the indices that are fixed during the estimation and not.

% Get the id for Kanes, alpha of which is fixed during the estimation
Kanes_id = find(strcmp(coordinates.name, 'Kanes'));

% Fixed and variable indices; covariance matrix is valid only for variable
% indices
indices_fixed = [index.tilde_delta_start ...
    index.varphi_known_start:index.varphi_known_end ...
    index.lambda_known_start:index.lambda_known_end ...
    (index.alpha_start + Kanes_id - 1)];

indices_variable = 1:dim_theta;
indices_variable = indices_variable(~ismember(indices_variable, indices_fixed));

% Calculate the covariance matrix following the usual sandwitch formula,
% for homoskedastic and White, respectively
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



%%%%% Simulate the contour data with the calculated homo and white SE
rng(19423);

plot_data_size = 20000;

plot_data_homo = repmat(theta_est, 1, plot_data_size);
plot_data_white = repmat(theta_est, 1, plot_data_size);

for i = 1:plot_data_size
    current_homo = mvnrnd(zeros(variable_size,1), Sigma_mat_homo)';
    current_white = mvnrnd(zeros(variable_size,1), Sigma_mat_white)';
    
    current_homo = current_homo + theta_est_variable;
    current_white = current_white + theta_est_variable;
    
    plot_data_homo(indices_variable,i) = current_homo;
    plot_data_white(indices_variable,i) = current_white;
end

plot_data_homo = [theta_est plot_data_homo];
plot_data_white = [theta_est plot_data_white];

csvwrite('plot_data_homoskedastic.csv', plot_data_homo);
csvwrite('plot_data_white.csv', plot_data_white);



%%%%% Calculate T^(1/vartheta)'s and its SE using Delta method
vartheta = 4;

auxdata.vartheta = vartheta;

T_one_over_vartheta = T_one_over_vartheta_vec(theta_est, auxdata);

%%% Finite difference for verification (obsolete)
%T_finite_diff_grad = zeros((K+L), dim_theta);
%for k = 1:dim_theta
%theta_est_epsilon = theta_est;
%theta_est_epsilon(k) = theta_est_epsilon(k) + 1e-8;
%T_epsilon = T_one_over_vartheta_vec(theta_est_epsilon, auxdata);
%T_finite_diff = (T_epsilon - T_one_over_vartheta) / 1e-8;

%T_finite_diff_grad(:,k) = T_finite_diff;
%end


%%% AutoDiff the T function
auxdata.result_selector = 1;
     
setup_T.numvar = dim_theta;
setup_T.objective = 'T_one_over_vartheta_k';
setup_T.constraint = 'T_one_over_vartheta_vec';
setup_T.auxdata = auxdata;
setup_T.order = 1;

funcs_error = adigatorGenFiles4Ipopt(setup_T);

T_grad_matrix = full(T_one_over_vartheta_vec_Jac(theta_est, auxdata));

nabla_T = T_grad_matrix(:,indices_variable);

% Delta method kicks in
T_covar_matrix_white = nabla_T * Sigma_mat_white * nabla_T';
T_covar_matrix_homo = nabla_T * Sigma_mat_homo * nabla_T';

T_one_over_vartheta_se_white = sqrt(diag(T_covar_matrix_white));
T_one_over_vartheta_se_homo = sqrt(diag(T_covar_matrix_homo));


%%% Simulation verification for T_se
T_data_white = zeros((K+L), plot_data_size);

for bb = 1:plot_data_size
    theta_bb = plot_data_white(:,bb);
   T_data_white(:,bb) = T_one_over_vartheta_vec(theta_bb, auxdata);
   
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 10. Report Table                                                %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% White SE
theta_se = zeros(dim_theta,1); % standard error matrix
theta_se(indices_variable,1) = theta_se_white;


%%% Remove roundoff errors in the std error estimates
for i = 1:dim_theta
   if theta_se(i) < 1e-10
      theta_se(i) = 0;
   end
end


%%% Formulate the tables
report_table = array2table(coord(:,1:5), 'VariableNames', ...
    {'id', 'cert', 'long_x', 'lat_y', 'validity'});

name_table = coordinates(:,{'id', 'name'});
report_table = join(report_table, name_table, 'Keys', 'id');

report_table.alpha = theta_firststage(index.alpha_start:index.alpha_end);
report_table.alpha_se = theta_se(index.alpha_start:index.alpha_end);
report_table.varphi_est = theta_firststage(index.varphi_known_start:...
    index.varphi_unknown_end);
report_table.varphi_se = theta_se(index.varphi_known_start:...
    index.varphi_unknown_end);
report_table.lambda_est = theta_firststage(index.lambda_known_start:...
    index.lambda_unknown_end);
report_table.lambda_se = theta_se(index.lambda_known_start:...
    index.lambda_unknown_end);

sigma_est_se = zeros(K+L,1);
sigma_est_se(1) = theta_firststage(1);
sigma_est_se(2) = theta_se(1);


report_table.sigma_est_se = sigma_est_se;
report_table.T_one_over_vartheta = T_one_over_vartheta;
report_table.T_one_over_vartheta_se = T_one_over_vartheta_se_white;


writetable(report_table, 'report_table_whitese.csv');


%%%%% Homoskedastic SE
theta_se = zeros(dim_theta,1); % standard error matrix
theta_se(indices_variable,1) = theta_se_homo;


%%% Remove roundoff errors in the std error estimates
for i = 1:dim_theta
   if theta_se(i) < 1e-10
      theta_se(i) = 0;
   end
end

%%% Formulate the tables
report_table = array2table(coord(:,1:5), 'VariableNames', ...
    {'id', 'cert', 'long_x', 'lat_y', 'validity'});

name_table = coordinates(:,{'id', 'name'});
report_table = join(report_table, name_table, 'Keys', 'id');

report_table.alpha = theta_firststage(index.alpha_start:index.alpha_end);
report_table.alpha_se = theta_se(index.alpha_start:index.alpha_end);
report_table.varphi_est = theta_firststage(index.varphi_known_start:...
    index.varphi_unknown_end);
report_table.varphi_se = theta_se(index.varphi_known_start:...
    index.varphi_unknown_end);
report_table.lambda_est = theta_firststage(index.lambda_known_start:...
    index.lambda_unknown_end);
report_table.lambda_se = theta_se(index.lambda_known_start:...
    index.lambda_unknown_end);

sigma_est_se = zeros(K+L,1);
sigma_est_se(1) = theta_firststage(1);
sigma_est_se(2) = theta_se(1);


report_table.sigma_est_se = sigma_est_se;
report_table.T_one_over_vartheta = T_one_over_vartheta;
report_table.T_one_over_vartheta_se = T_one_over_vartheta_se_homo;

writetable(report_table, 'report_table_homose.csv');


save('workspace_closedform_step10.mat');

