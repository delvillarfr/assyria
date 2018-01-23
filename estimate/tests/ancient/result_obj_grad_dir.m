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

startupadigator;  % Load ADiGator module

static_constraint_impose = 1;

wahsusana_drop = 0; % Whether to drop Wasusana. 1 to drop. 0 is default.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 1. Preliminary                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Load and clean the data
estimation_data = readtable('ppml_estimation_directional_data.csv','TreatAsEmpty',{'.','NA'});
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


if wahsusana_drop == 1
   lb_varphi_id(lb_varphi_id == 14) = NaN;
   ub_varphi_id(ub_varphi_id == 14) = NaN;
   lb_lambda_id(lb_lambda_id == 14) = NaN;
   ub_lambda_id(ub_lambda_id == 14) = NaN;
end



ub_varphi = zeros(L,1);
lb_varphi = zeros(L,1);
ub_lambda = zeros(L,1);
lb_lambda = zeros(L,1);

%%% Overwrite the default bounds if they are specified by the constraints
%%% data
if static_constraint_impose ==1
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

options.lb((index.alpha_start+1),1) = 100;
options.ub((index.alpha_start+1),1) = 100;

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

%% adigatorGenFiles4Ipopt generates everything required by ipopt
%% This part usually takes about a min
%tic
%funcs = adigatorGenFiles4Ipopt(setup);
%gentime = toc;

% Functions
funcs.objective = @(theta)sqerr_sum(theta,auxdata);
funcs.gradient = @(theta)sqerr_sum_Grd(theta,auxdata);

theta_opt = csvread('theta_firststage_plot.csv');

disp(theta_opt)
disp(size(theta_opt))
% Check nothing should be done to it.
f = funcs.objective(theta_opt)
g = funcs.gradient(theta_opt)

t.obj = f
t.grad = g

writetable(struct2table(t), 'result_obj_grad_dir.csv')
