data = readtable('test_distance_fn.csv');


dist = zeros(500, 1)
dist_sq = zeros(500, 1)

for i = 1:500
	dist(i) = euclidean_dist( data.longi(i), data.lati(i), data.longj(i), data.latj(i))
	dist_sq(i) = euclidean_dist_sq( data.longi(i), data.lati(i), data.longj(i), data.latj(i))
end

writetable(array2table(dist), 'distances_jhwi.csv')
writetable(array2table(dist_sq), 'distances_sq_jhwi.csv')
