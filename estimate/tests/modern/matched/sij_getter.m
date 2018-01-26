function alpha_d_sq_sigma = sqerr_sum(theta, auxdata)
%%% This function is the GMM objective function
% Extract necessary variables from auxdata
%K = auxdata.K;
%L = auxdata.L;
I = auxdata.I;
N = auxdata.N;
id_comb = auxdata.id_comb;
s_ij = auxdata.s_ij;
%dist = auxdata.dist;
index = auxdata.index;
dist_sq = auxdata.dist_sq;

% Extract current sigma and xi from theta
sigma = theta(1);
alpha_i = theta(index.alpha_start:index.alpha_end);

% Calculate the distances for all the combinations


alpha_d_sq_sigma = zeros(N,1);

for k = 1:N
    i = id_comb(k,1);
    %j = id_comb(k,2);
  
    alpha_d_sq_sigma(k,1) = alpha_i(i,1) * ((dist_sq(k,1))^(-sigma));
    %alpha_d_sq_sigma(k,1) = alpha_i(j,1) * ((dist_sq(k,1))^(-sigma));   
end

for k = 1:I
   alpha_d_sq_sigma(((k-1)*(I-1)+1):(k*(I-1)),1) = ...
       alpha_d_sq_sigma(((k-1)*(I-1)+1):(k*(I-1)),1) ./ ...
       sum(alpha_d_sq_sigma(((k-1)*(I-1)+1):(k*(I-1)),1));
    
    
end

end



