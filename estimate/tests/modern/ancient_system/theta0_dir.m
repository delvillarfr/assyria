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

writetable(array2table(theta0), 'theta0_dir.csv')
