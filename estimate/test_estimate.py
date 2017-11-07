'''
Tests of Model Parameter Estimation
'''
import estimate
import pandas as pd
import numpy as np


def test_iticount_imports_is_composed_solely_of_cities_in_coordinates():
    e = estimate.Estimate('directional')
    cities_i = np.sort( e.df_iticount['id_i'].unique() )
    np.testing.assert_array_equal( cities_i,
                                   np.sort(e.df_coordinates['id'].values) )


def test_iticount_exports_is_composed_solely_of_cities_in_coordinates():
    e = estimate.Estimate('directional')
    cities_j = np.sort( e.df_iticount['id_j'].unique() )
    np.testing.assert_array_equal( cities_j,
                                   np.sort(e.df_coordinates['id'].values) )


def test_iticount_has_all_possible_city_combinations():
    ''' Meaningful only if previous two tests pass '''
    e = estimate.Estimate('directional')
    assert len(e.df_iticount) == (len(e.df_coordinates)
                                  * (len(e.df_coordinates)-1))


def test_same_known_and_unknown_cities():
    ''' 15 (11) (un)known cities, acc to line 73 of main_script_directional.m '''
    e = estimate.Estimate('directional')
    assert len(e.df_unknown) == 11
    assert len(e.df_known) == 15


def test_same_constraints_after_coord_replace_directional():
    e = estimate.Estimate('directional')
    replaced_dyn = e.replace_id_coord(e.df_constr_dyn)
    replaced_stat = e.replace_id_coord(e.df_constr_stat)


def test_same_distances_with_haversine_approx_and_euclidean_dist():
    e = estimate.Estimate('directional')
    coords = pd.read_csv('test_distance_fn.csv')
    dist_mine = e.haversine_approx(coords[['lati', 'longi']].values,
                                   coords[['latj', 'longj']].values)

    dist_jhwi = pd.read_csv('distances_jhwi.csv')

    np.testing.assert_array_equal(np.round(dist_jhwi.values, 6),
                                  np.round(dist_mine.reshape((500,1)), 6))


def test_same_distances_with_haversine_approx_and_euclidean_dist_sq():
    e = estimate.Estimate('directional')
    coords = pd.read_csv('test_distance_fn.csv')
    dist_mine = (e.haversine_approx(coords[['lati', 'longi']].values,
                                    coords[['latj', 'longj']].values))**2

    dist_jhwi = pd.read_csv('distances_sq_jhwi.csv')

    np.testing.assert_array_equal(np.round(dist_jhwi.values, 5),
                                  np.round(dist_mine.reshape((500,1)), 5))


def test_coordinate_pairs_match_those_of_their_corresponding_iticount_entry():
    e = estimate.Estimate('directional')
    lats = pd.read_csv('lats_test.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('longs_test.csv')
    longs = longs.iloc[0, :].values

    e.df_unknown['lat_y'] = lats
    e.df_unknown['long_x'] = longs

    coords = e.df_known.append(e.df_unknown)
    print(coords)
    
    merged = e.df_iticount.copy()
    merged = merged.astype(np.float64, errors='ignore')
    coords_merged = []
    for s in ['_i', '_j']:
        merged = merged.merge(coords,
                              how='left',
                              left_on='id'+s,
                              right_on='id')
        merged = merged.rename(columns = {'long_x': 'long_x'+s,
                                          'lat_y': 'lat_y'+s})
        coords_merged.append(merged[['lat_y'+s, 'long_x'+s]].values)
    
    coords_estimate = e.get_coordinate_pairs(lats, longs)
    
    for i in range(2):
        #np.testing.assert_array_equal(coords_estimate[i], coords_merged[i])    
        np.testing.assert_array_almost_equal(coords_estimate[i],
                                             coords_merged[i])    



def test_same_fetched_distances_directional():
    e = estimate.Estimate('directional')
    zeta = 2
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(26)
    lats = pd.read_csv('lats_test.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('longs_test.csv')
    longs = longs.iloc[0, :].values
    
    distances_mine = e.fetch_dist(lats, longs).reshape((650,1))
    #i = e.df_iticount['id_i'].values
    #j = e.df_iticount['id_j'].values
    #d = np.column_stack((i, j, distances_mine**2))
    #my_dat = pd.DataFrame(d, columns=['id_i', 'id_j', 'dist'])
    #my_dat.to_csv('dist_mine.csv', index=False)
    distances_jhwi = pd.read_csv('distances_fetched_jhwi.csv')
    np.testing.assert_array_almost_equal(distances_mine, 
                                         np.sqrt(distances_jhwi))


def test_same_s_ij_model_directional():
    e = estimate.Estimate('directional')
    zeta = 2
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(26)
    lats = pd.read_csv('lats_test.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('longs_test.csv')
    longs = longs.iloc[0, :].values

    dists = e.fetch_dist(lats, longs)
    #df = pd.DataFrame(dists)
    #df.to_csv('ll.csv', index=False)

    sij_mine = e.s_ij_model(zeta, alpha, dists).reshape((650, 1))
    sij_jhwi = pd.read_csv('sij_jhwi.csv')
    np.testing.assert_array_almost_equal(sij_mine, sij_jhwi)


def test_same_sqerr_sum_directional():
    e = estimate.Estimate('directional')
    zeta = [2]
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(26).tolist()
    lats = pd.read_csv('lats_test.csv')
    longs = pd.read_csv('longs_test.csv')
    
    sqerr_mine = np.ones((100, 1))
    for i in range(100):
        lat_guess = lats.iloc[i, :].tolist()
        lng_guess = longs.iloc[i, :].tolist()
        print(lat_guess)
        sqerr_mine[i] = e.sqerr_sum(zeta + alpha + lat_guess + lng_guess)

    sqerr_jhwi = pd.read_csv('sqerr_jhwi.csv').values
    np.testing.assert_array_almost_equal(sqerr_mine, sqerr_jhwi)
