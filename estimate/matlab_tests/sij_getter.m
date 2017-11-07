function alpha_d_sq_sigma = sij_getter(theta, auxdata)
%%% This function is the objective function
% Extract necessary variables from auxdata
I = auxdata.I;
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

alpha_d_sq_sigma = zeros(N,1);

for k = 1:N
    i = id_comb(k,1);
    j = id_comb(k,2);
    dist_sq(k,1) = euclidean_dist_sq(varphi(i),lambda(i),varphi(j),lambda(j));
  
    alpha_d_sq_sigma(k,1) = alpha_i(i,1) * ((dist_sq(k,1))^(-sigma));
end

for k = 1:I
   alpha_d_sq_sigma(((k-1)*(I-1)+1):(k*(I-1)),1) = ...
       alpha_d_sq_sigma(((k-1)*(I-1)+1):(k*(I-1)),1) ./ ...
       sum(alpha_d_sq_sigma(((k-1)*(I-1)+1):(k*(I-1)),1));
    
    
end


end
