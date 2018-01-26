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
fix_id = 12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 1. Preliminary                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Load and clean the data
estimation_data = readtable('estimation_data_ancientsystem_directional.csv','TreatAsEmpty',{'.','NA'});

coordinates = readtable('coordinates_ancientsystem.csv','TreatAsEmpty',{'.','NA'});

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

alphas = table2array(readtable('a_rand.csv'));
sigmas = table2array(readtable('sigma_rand.csv'));

results = ones(K*(K-1), 100)

for i = 1: 100
	%%% Initialize theta0
	alpha = alphas(i, :)';
	sigma = sigmas(i, :)';
	theta0 = [sigma;alpha];

	results(:, i) = sij_getter(theta0, auxdata)
end

writetable(array2table(results), 'sij_rand_dir.csv')
