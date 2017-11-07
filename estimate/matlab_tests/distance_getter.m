function dist_sq = distance_getter(theta, auxdata)
%%% This function is the objective function
% Extract necessary variables from auxdata
N = auxdata.N;
id_comb = auxdata.id_comb;
s_ij = auxdata.s_ij;
index = auxdata.index;
data_selector = auxdata.data_selector;

% Extract current sigma from theta
sigma = theta(1);
alpha_i = theta(index.alpha_start:index.alpha_end);

% Extract current varphi and lambda from theta
varphi = theta((index.varphi_known_start):(index.varphi_unknown_end),1);
lambda = theta((index.lambda_known_start):(index.lambda_unknown_end),1);

% Calculate the distances for all the combinations
dist_sq = zeros(N,1);

for k = 1:N
    i = id_comb(k,1);
    j = id_comb(k,2);
    dist_sq(k,1) = euclidean_dist_sq(varphi(i),lambda(i),varphi(j),lambda(j));
end

end
