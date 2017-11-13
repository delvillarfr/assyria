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
writetable(array2table(theta0), 'theta0_nondir.csv')
