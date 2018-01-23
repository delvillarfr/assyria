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

%%% Initialize lambda and varphi at the original locations
varphi0 = coord(:,3); % varphi is the longitude_x
lambda0 = coord(:,4); % lambda is the latitude_y

varphi_known = varphi0(1:K);
lambda_known = lambda0(1:K);

alpha0 = ones((K+L),1);

lats = table2array(readtable('lats_rand_nondir.csv', 'HeaderLines', 1));
longs = table2array(readtable('longs_rand_nondir.csv', 'HeaderLines', 1));
disp(lats);

results = ones(100,1);

for i = 1: 100
	%%% Initialize theta0
	la = lats(i, :)';
	lo = longs(i, :)';
	disp(la);
	theta0 = [sigma0;tilde_delta0;varphi_known;lo;lambda_known;la;alpha0];
	dim_theta = size(theta0,1);
	disp(theta0);

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

	results(i) = sqerr_sum(theta0, auxdata)
end

writetable(array2table(results), 'sqerr_rand_nondir.csv')
