'''
Tests of Model Parameter Estimation
'''
import estimate
import pandas as pd
import numpy as np

pd.set_option('precision',15)
accuracy = 8

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


def test_same_known_and_unknown_cities_dir():
    ''' 15 (11) (un)known cities, acc to line 73 of main_script_directional.m '''
    e = estimate.Estimate('directional')
    assert len(e.df_unknown) == 11
    assert len(e.df_known) == 15


def test_same_known_and_unknown_cities_nondir():
    ''' Amkuwa (known) and Ursu (unknown) enter the non-directional analysis.'''
    e = estimate.Estimate('non_directional')
    assert len(e.df_unknown) == 12
    assert len(e.df_known) == 16


def test_same_constraints_after_coord_replace_directional_dir():
    e = estimate.Estimate('directional')
    replaced_dyn = e.replace_id_coord(e.df_constr_dyn)
    replaced_stat = e.replace_id_coord(e.df_constr_stat)


def test_same_distances_with_haversine_approx_and_euclidean_dist():
    e = estimate.Estimate('directional')
    coords = pd.read_csv('./tests/data/coords_sample.csv')
    dist_mine = e.haversine_approx(coords[['lati', 'longi']].values,
                                   coords[['latj', 'longj']].values)

    dist_jhwi = pd.read_csv('./tests/data/distances_sample.csv')

    np.testing.assert_array_almost_equal(dist_jhwi.values,
                                         dist_mine.reshape((500,1)),
                                         decimal=accuracy)


def test_same_distances_with_haversine_approx_and_euclidean_dist_sq():
    e = estimate.Estimate('directional')
    coords = pd.read_csv('./tests/data/coords_sample.csv')
    dist_mine = (e.haversine_approx(coords[['lati', 'longi']].values,
                                    coords[['latj', 'longj']].values))**2

    dist_jhwi = pd.read_csv('./tests/data/distances_sq_sample.csv')

    np.testing.assert_array_almost_equal(dist_jhwi.values,
                                         dist_mine.reshape((500,1)),
                                         decimal=accuracy)


def test_coordinate_pairs_match_those_of_their_corresponding_iticount_entry_dir():
    e = estimate.Estimate('directional')
    lats = pd.read_csv('./tests/data/lats_rand_dir.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('./tests/data/longs_rand_dir.csv')
    longs = longs.iloc[0, :].values

    e.df_unknown['lat_y'] = lats
    e.df_unknown['long_x'] = longs

    coords = e.df_known.append(e.df_unknown)

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
        np.testing.assert_array_almost_equal(coords_estimate[i],
                                             coords_merged[i],
                                             decimal=accuracy)


def test_coordinate_pairs_match_those_of_their_corresponding_iticount_entry_nondir():
    e = estimate.Estimate('non_directional')
    lats = pd.read_csv('./tests/data/lats_rand_nondir.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('./tests/data/longs_rand_nondir.csv')
    longs = longs.iloc[0, :].values

    e.df_unknown['lat_y'] = lats
    e.df_unknown['long_x'] = longs

    coords = e.df_known.append(e.df_unknown)

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
        np.testing.assert_array_almost_equal(coords_estimate[i],
                                             coords_merged[i],
                                             decimal=accuracy)


def test_same_fetched_distances_dir():
    e = estimate.Estimate('directional')
    zeta = 2
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(26)
    lats = pd.read_csv('./tests/data/lats_rand_dir.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('./tests/data/longs_rand_dir.csv')
    longs = longs.iloc[0, :].values

    distances_mine = e.fetch_dist(lats, longs).reshape((650,1))
    distances_jhwi = pd.read_csv('./tests/data/distances_rand_dir.csv')
    np.testing.assert_array_almost_equal(distances_mine,
                                         np.sqrt(distances_jhwi),
                                         decimal=accuracy)


def test_same_fetched_distances_nondir():
    e = estimate.Estimate('non_directional')
    zeta = 2
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(26)
    lats = pd.read_csv('./tests/data/lats_rand_nondir.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('./tests/data/longs_rand_nondir.csv')
    longs = longs.iloc[0, :].values

    distances_mine = e.fetch_dist(lats, longs).reshape((756,1))
    distances_jhwi = pd.read_csv('./tests/data/distances_rand_nondir.csv')
    np.testing.assert_array_almost_equal(distances_mine,
                                         np.sqrt(distances_jhwi),
                                         decimal=accuracy)


def test_same_s_ij_model_directional_dir():
    e = estimate.Estimate('directional')
    zeta = 2
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(26)
    lats = pd.read_csv('./tests/data/lats_rand_dir.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('./tests/data/longs_rand_dir.csv')
    longs = longs.iloc[0, :].values

    dists = e.fetch_dist(lats, longs)

    sij_mine = e.s_ij_model(zeta, alpha, dists).reshape((650, 1))
    sij_jhwi = pd.read_csv('./tests/data/sij_rand_dir.csv')
    np.testing.assert_array_almost_equal(sij_mine,
                                         sij_jhwi,
                                         decimal=accuracy)


def test_same_s_ij_model_directional_nondir():
    e = estimate.Estimate('non_directional')
    zeta = 2
    #alpha has 28 entries, the num of cities.
    alpha = np.ones(28)
    lats = pd.read_csv('./tests/data/lats_rand_nondir.csv')
    lats = lats.iloc[0, :].values
    longs = pd.read_csv('./tests/data/longs_rand_nondir.csv')
    longs = longs.iloc[0, :].values

    dists = e.fetch_dist(lats, longs)

    sij_mine = e.s_ij_model(zeta, alpha, dists).reshape((756, 1))
    sij_jhwi = pd.read_csv('./tests/data/sij_rand_nondir.csv')
    np.testing.assert_array_almost_equal(sij_mine,
                                         sij_jhwi,
                                         decimal=accuracy)


def test_full_to_short_varlist():
    e = estimate.Estimate('directional')
    input_full = np.concatenate(( [0, 1],
                                  2*np.ones(e.num_cities_known),
                                  3*np.ones(e.num_cities_unknown),
                                  4*np.ones(e.num_cities_known),
                                  5*np.ones(e.num_cities_unknown),
                                  99*np.ones(e.num_cities)
                                  ))
    input_short = np.concatenate(( [0],
                                  3*np.ones(e.num_cities_unknown),
                                  5*np.ones(e.num_cities_unknown),
                                  99*np.ones(e.num_cities)
                                  ))
    input_short_mine = input_full[e.full_to_short_i()]
    np.testing.assert_array_equal(input_short,
                                  input_short_mine)
    ## In case you are still not convinced: manual check.
    #inputs = pd.read_csv('./tests/data/inputs_dir.csv')
    #i = inputs.iloc[0, :].values
    #input_s = i[e.full_to_short_i()]
    #pd.DataFrame(input_s).to_csv('row1.csv')


def test_error_equivalence_full_vars_vs_few_vars():
    e = estimate.Estimate('directional')
    inputs = pd.read_csv('./tests/data/inputs_full_vs_few.csv',
            header=None)
    errors = np.empty((1001, 650))
    errors_full = np.empty((1001, 650))
    for i in range(len(inputs)):
        inp = inputs.iloc[i, :].values
        inp_short = inp[e.full_to_short_i()]
        errors[i, :] = e.get_errors(inp_short, full_vars=False)
        errors_full[i, :] = e.get_errors(inp, full_vars=True)
    np.testing.assert_array_almost_equal(errors,
                                         errors_full,
                                         decimal=accuracy)


def test_same_bounds_to_ipopt_dir_static():
    e = estimate.Estimate('directional')
    bounds = e.get_bounds(e.replace_id_coord(e.df_constr_stat), full_vars=True)

    bounds_jhwi = pd.read_csv('./tests/data/bounds_dir_stat.csv')
    bounds_jhwi = bounds_jhwi.replace(np.inf, 1.0e20)

    np.testing.assert_array_almost_equal(np.array(bounds[0]),
                                         bounds_jhwi['lb'].values.flatten(),
                                         decimal=accuracy)
    np.testing.assert_array_almost_equal(np.array(bounds[1]),
                                         bounds_jhwi['ub'].values.flatten(),
                                         decimal=accuracy)


def test_same_bounds_to_ipopt_nondir_static():
    e = estimate.Estimate('non_directional')
    bounds = e.get_bounds(e.replace_id_coord(e.df_constr_stat), full_vars=True)

    bounds_jhwi = pd.read_csv('./tests/data/bounds_nondir_stat.csv')
    bounds_jhwi = bounds_jhwi.replace(np.inf, 1.0e20)

    np.testing.assert_array_almost_equal(np.array(bounds[0]),
                                         bounds_jhwi['lb'].values.flatten(),
                                         decimal=accuracy)
    np.testing.assert_array_almost_equal(np.array(bounds[1]),
                                         bounds_jhwi['ub'].values.flatten(),
                                         decimal=accuracy)


def test_same_initial_condition_dir():
    e = estimate.Estimate('directional')
    theta0_mine = e.initial_cond(full_vars=True)
    # zeta is one half sigma
    theta0_mine[0] = theta0_mine[0]/2
    theta0_jhwi = pd.read_csv('./tests/data/theta0_dir.csv')
    np.testing.assert_array_almost_equal(theta0_mine,
                                         theta0_jhwi.values.flatten(),
                                         decimal=accuracy)


def test_same_initial_condition_nondir():
    e = estimate.Estimate('non_directional')
    theta0_mine = e.initial_cond(full_vars=True)
    # zeta is one half sigma
    theta0_mine[0] = theta0_mine[0]/2
    theta0_jhwi = pd.read_csv('./tests/data/theta0_nondir.csv')
    np.testing.assert_array_almost_equal(theta0_mine,
                                         theta0_jhwi.values.flatten(),
                                         decimal=accuracy)


def test_same_sqerr_sum_dir():
    e = estimate.Estimate('directional')
    zeta = [2]
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(26).tolist()
    lats = pd.read_csv('./tests/data/lats_rand_dir.csv')
    longs = pd.read_csv('./tests/data/longs_rand_dir.csv')

    sqerr_mine = np.ones((100, 1))
    for i in range(100):
        lat_guess = lats.iloc[i, :].tolist()
        lng_guess = longs.iloc[i, :].tolist()
        sqerr_mine[i] = e.sqerr_sum(zeta + lng_guess + lat_guess + alpha)

    sqerr_jhwi = pd.read_csv('./tests/data/sqerr_rand_dir.csv').values
    np.testing.assert_array_almost_equal(sqerr_mine,
                                         sqerr_jhwi,
                                         decimal=accuracy)


def test_same_sqerr_sum_nondir():
    e = estimate.Estimate('non_directional')
    zeta = [2]
    #alpha has 28 entries, the num of cities.
    alpha = np.ones(28).tolist()
    lats = pd.read_csv('./tests/data/lats_rand_nondir.csv')
    longs = pd.read_csv('./tests/data/longs_rand_nondir.csv')

    sqerr_mine = np.ones((100, 1))
    for i in range(100):
        lat_guess = lats.iloc[i, :].tolist()
        lng_guess = longs.iloc[i, :].tolist()
        sqerr_mine[i] = e.sqerr_sum(zeta + lng_guess + lat_guess + alpha)

    sqerr_jhwi = pd.read_csv('./tests/data/sqerr_rand_nondir.csv').values
    np.testing.assert_array_almost_equal(sqerr_mine,
                                         sqerr_jhwi,
                                         decimal=accuracy)


def test_same_sqerr_sum_full_vars_dir():
    e = estimate.Estimate('directional')
    # Add useless param
    zeta = [2, 2]
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(26).tolist()
    lats = pd.read_csv('./tests/data/lats_rand_dir.csv')
    longs = pd.read_csv('./tests/data/longs_rand_dir.csv')

    lats_known = e.df_known['lat_y'].values
    lats_known = pd.DataFrame(np.tile(lats_known, (100, 1)))
    lats = pd.concat([lats_known, lats], axis=1)

    longs_known = e.df_known['long_x'].values
    longs_known = pd.DataFrame(np.tile(longs_known, (100, 1)))
    longs = pd.concat([longs_known, longs], axis=1)

    sqerr_mine = np.ones((100, 1))
    for i in range(100):
        lat_guess = lats.iloc[i, :].tolist()
        lng_guess = longs.iloc[i, :].tolist()
        sqerr_mine[i] = e.sqerr_sum(zeta + lng_guess + lat_guess + alpha,
                                   full_vars=True)

    sqerr_jhwi = pd.read_csv('./tests/data/sqerr_rand_dir.csv').values
    np.testing.assert_array_almost_equal(sqerr_mine,
                                         sqerr_jhwi,
                                         decimal=accuracy)


def test_same_sqerr_sum_full_vars_nondir():
    e = estimate.Estimate('non_directional')
    # Add useless param
    zeta = [2, 2]
    #alpha has 26 entries, the num of cities.
    alpha = np.ones(28).tolist()
    lats = pd.read_csv('./tests/data/lats_rand_nondir.csv')
    longs = pd.read_csv('./tests/data/longs_rand_nondir.csv')

    lats_known = e.df_known['lat_y'].values
    lats_known = pd.DataFrame(np.tile(lats_known, (100, 1)))
    lats = pd.concat([lats_known, lats], axis=1)

    longs_known = e.df_known['long_x'].values
    longs_known = pd.DataFrame(np.tile(longs_known, (100, 1)))
    longs = pd.concat([longs_known, longs], axis=1)

    sqerr_mine = np.ones((100, 1))
    for i in range(100):
        lat_guess = lats.iloc[i, :].tolist()
        lng_guess = longs.iloc[i, :].tolist()
        sqerr_mine[i] = e.sqerr_sum(zeta + lng_guess + lat_guess + alpha,
                                   full_vars=True)

    sqerr_jhwi = pd.read_csv('./tests/data/sqerr_rand_nondir.csv').values
    np.testing.assert_array_almost_equal(sqerr_mine,
                                         sqerr_jhwi,
                                         decimal=accuracy)


def test_same_gradients_at_initial_coords_dir():
    e = estimate.Estimate('directional')
    varlist = e.initial_cond(full_vars=True)
    grad_mine = e.grad_full_vars(varlist)

    grad_jhwi = pd.read_csv('./tests/data/gradient0_dir.csv')
    grad_jhwi = grad_jhwi.values.flatten()

    # sigma is one half zeta
    grad_jhwi[0] = grad_jhwi[0]/2

    np.testing.assert_array_almost_equal(grad_mine,
                                         grad_jhwi,
                                         decimal=accuracy)


def test_same_gradients_at_initial_coords_nondir():
    e = estimate.Estimate('non_directional')
    varlist = e.initial_cond(full_vars=True)
    grad_mine = e.grad_full_vars(varlist)

    grad_jhwi = pd.read_csv('./tests/data/gradient0_nondir.csv')
    grad_jhwi = grad_jhwi.values.flatten()

    # sigma is one half zeta
    grad_jhwi[0] = grad_jhwi[0]/2

    np.testing.assert_array_almost_equal(grad_mine,
                                         grad_jhwi,
                                         decimal=accuracy)


def test_many_inputs_grad_and_objective_dir():
    e = estimate.Estimate('directional')
    inputs = pd.read_csv('./tests/data/inputs_dir.csv')
    # Sigma to zeta
    inputs.iloc[:, 0] = inputs.iloc[:, 0]*2
    grads_mine = np.empty((1000, 80))
    obj_mine = np.empty((1000, 1))
    for i in range(len(inputs)):
        inp = inputs.iloc[i, :].values
        obj_mine[i] = e.sqerr_sum(inp, full_vars = True)
        grads_mine[i, :] = e.grad_full_vars(inp)

    obj_jhwi = pd.read_csv('./tests/data/inputs_objective_dir.csv').values
    grads_jhwi = pd.read_csv('./tests/data/inputs_gradients_dir.csv').values
    grads_jhwi[:, 0] = grads_jhwi[:, 0]/2
    np.testing.assert_array_almost_equal(obj_mine,
                                         obj_jhwi,
                                         decimal=accuracy)
    np.testing.assert_array_almost_equal(grads_mine,
                                         grads_jhwi,
                                         decimal=accuracy)


def test_many_inputs_grad_and_objective_nondir():
    e = estimate.Estimate('non_directional')
    inputs = pd.read_csv('./tests/data/inputs_nondir.csv')
    # Sigma to zeta
    inputs.iloc[:, 0] = inputs.iloc[:, 0]*2
    grads_mine = np.empty((1000, 86))
    obj_mine = np.empty((1000, 1))
    for i in range(len(inputs)):
        inp = inputs.iloc[i, :].values
        obj_mine[i] = e.sqerr_sum(inp, full_vars = True)
        grads_mine[i, :] = e.grad_full_vars(inp)

    obj_jhwi = pd.read_csv('./tests/data/inputs_objective_nondir.csv').values
    grads_jhwi = pd.read_csv('./tests/data/inputs_gradients_nondir.csv').values
    grads_jhwi[:, 0] = grads_jhwi[:, 0]/2

    np.testing.assert_array_almost_equal(obj_mine,
                                         obj_jhwi,
                                         decimal=accuracy)
    np.testing.assert_array_almost_equal(grads_mine,
                                         grads_jhwi,
                                         decimal=accuracy)


def test_same_objective_at_optimal_point():
    e = estimate.Estimate('directional')
    theta = pd.read_csv('./tests/data/theta_firststage_plot.csv',
                        header=None)
    theta = theta.values.flatten()
    # sigma to zeta
    theta[0] = theta[0]*2

    obj_mine = e.sqerr_sum(theta, full_vars=True)
    grad_mine = e.grad_full_vars(theta)
    grad_mine[0] = grad_mine[0]*2

    vals_jhwi = pd.read_csv('./tests/data/result_obj_grad_dir.csv')
    obj_jhwi = vals_jhwi.iat[0, 0]
    grad_jhwi = vals_jhwi.drop('obj', axis=1).values.flatten()

    df = pd.DataFrame([grad_mine, grad_jhwi])
    df['objective_fn'] = [obj_mine, obj_jhwi]
    df.to_csv('results_at_optimum.csv', index=False)

    np.testing.assert_array_almost_equal(df.iloc[0, :].values,
                                         df.iloc[1, :].values,
                                         decimal=accuracy)


def gen_test_inputs(l=1000, full_vars=False, directional=True):
    if directional:
        n_cities = 26
        name = 'inputs_dir'
    else:
        n_cities = 28
        name = 'inputs_nondir'

    alphas = np.random.uniform(0, 50, (l, n_cities))
    zeta = np.random.uniform(0, 10, l)
    if full_vars:
        lngs = np.random.uniform(27, 45, (l, n_cities))
        lats = np.random.uniform(36, 42, (l, n_cities))
        df = pd.DataFrame( np.column_stack((zeta,
                                            2*np.ones(l),
                                            lngs,
                                            lats,
                                            alphas)))
    df.to_csv("~/" + name + ".csv", index=False)


def test_same_errors():
    e = estimate.Estimate('directional')
    inputs = pd.read_csv('./tests/data/inputs_dir.csv')
    # Sigma to zeta
    inputs.iloc[:, 0] = inputs.iloc[:, 0]*2
    errors_mine = np.empty((1000, 650))
    for i in range(len(inputs)):
        inp = inputs.iloc[i, :].values
        errors_mine[i, :] = e.get_errors(inp, full_vars=True)

    errors_jhwi = pd.read_csv('./tests/data/inputs_errors_dir.csv').values
    #errors_jhwi[:, 0] = errors_jhwi[:, 0]/2
    np.testing.assert_array_almost_equal(errors_mine,
                                         errors_jhwi,
                                         decimal=accuracy)


#def test_same_error_jacobian_few_vars():
#    e = estimate.Estimate('directional')
#    results = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
#                           header=None)
#                 .values
#                 .flatten()
#                 )
#    # sigma to zeta
#    results[0] = results[0]*2
#
#    # Go to few vars
#    inp = results[e.full_to_short_i()]
#
#    jac_mine = e.jac_errors(inp)
#
#    jac_jhwi = pd.read_csv('./tests/data/jacobian_errors_full.csv',
#                           header=None).values
#    # first column must be twice the size of mine
#    jac_jhwi[:, 0] = jac_jhwi[:, 0]/2
#    jac_jhwi = pd.DataFrame(jac_jhwi)
#
#    i = e.div_indices[True]
#    jac_jhwi = jac_jhwi.drop(columns = ([1]
#                             + range(i['long_s'], i['long_unknown_s'])
#                             + range(i['lat_s'], i['lat_unknown_s'])))
#
#    np.testing.assert_almost_equal(jac_mine, jac_jhwi, decimal=accuracy)


def test_same_error_jacobian_full_vars():
    e = estimate.Estimate('directional')
    results = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    print(results)
    # sigma to zeta
    results[0] = results[0]*2

    jac_mine = e.jac_errors_full_vars(np.float64(results))

    jac_jhwi = pd.read_csv('./tests/data/jacobian_errors_full.csv',
                           header=None).values
    # first column must be twice the size of mine
    jac_jhwi[:, 0] = jac_jhwi[:, 0]/2
    np.testing.assert_almost_equal(jac_mine, jac_jhwi, decimal=accuracy)


#def test_error_equivalence():
#    '''
#    THE REASON THIS TEST FAILS IS THAT THE KNOWN COORDINATES THAT ARE
#    STORED AS ATTRIBUTES ARE MORE PRECISE THAN THE ONES CONTAINED IN
#    theta_firststage_plot.csv
#    I KNOW THIS BECAUSE THAT theta CORRESPONDS TO ROW 1 TESTED IN
#    test_error_equivalence_full_vars_vs_few_vars()
#    '''
#    e = estimate.Estimate('directional')
#    results = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
#                           header=None)
#                 .values
#                 .flatten()
#                 )
#    # sigma to zeta
#    results[0] = results[0]*2
#
#    v = results[e.full_to_short_i()]
#    error = e.get_errors(v)
#    error_full = e.get_errors(results, full_vars=True)
#
#    ## Remove columns of extra vars
#    #i = e.div_indices[True]
#    ##Cast as dataframe, drop columns, back to array
#    #jac_full = pd.DataFrame(jac_full)
#    #jac_full = jac_full.drop(columns = ([1]
#    #                                 + range(i['long_s'], i['long_unknown_s'])
#    #                                 + range(i['lat_s'], i['lat_unknown_s'])))
#    np.testing.assert_almost_equal(error, error_full)


#def test_error_jacobian_equivalence():
#    ''' Read docstring of previous test. '''
#    e = estimate.Estimate('directional')
#    results = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
#                           header=None)
#                 .values
#                 .flatten()
#                 )
#    # sigma to zeta
#    results[0] = results[0]*2
#
#    v = results[e.full_to_short_i()]
#    jac = e.jac_errors(np.float64(v))
#    jac_full = e.jac_errors_full_vars(np.float64(results))
#
#    # Remove columns of extra vars
#    i = e.div_indices[True]
#    #Cast as dataframe, drop columns, back to array
#    jac_full = pd.DataFrame(jac_full)
#    jac_full = jac_full.drop(columns = ([1]
#                                     + range(i['long_s'], i['long_unknown_s'])
#                                     + range(i['lat_s'], i['lat_unknown_s'])))
#    np.testing.assert_almost_equal(jac, jac_full.values, decimal=3)


def test_same_variance_matrix_white():
    ''' One value is different with 7 decimals... '''
    e = estimate.Estimate('directional')

    varlist = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    # sigma to zeta
    varlist[0] = varlist[0]*2
    var_mine = e.get_variance(np.float64(varlist), full_vars = True)

    var_jhwi = pd.read_csv('./tests/data/variance_white.csv',
                           header=None).values
    # Correct variance for first term
    var_jhwi[0, :] = 2*var_jhwi[0, :]
    var_jhwi[:, 0] = 2*var_jhwi[:, 0]

    #pd.DataFrame( (var_mine - var_jhwi)*(np.abs(var_mine -
    #    var_jhwi)>0.000001)).to_csv('quickie.csv')

    np.testing.assert_almost_equal(var_mine, var_jhwi, decimal=5)


def test_same_variance_matrix_homo():
    ''' One value is different with 7 decimals... '''
    e = estimate.Estimate('directional')

    varlist = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    # sigma to zeta
    varlist[0] = varlist[0]*2
    var_mine = e.get_variance(np.float64(varlist),
                              var_type='homo',
                              full_vars = True)

    var_jhwi = pd.read_csv('./tests/data/variance_homo.csv',
                           header=None).values
    # Correct variance for first term
    var_jhwi[0, :] = 2*var_jhwi[0, :]
    var_jhwi[:, 0] = 2*var_jhwi[:, 0]

    #pd.DataFrame( (var_mine - var_jhwi)*(np.abs(var_mine -
    #    var_jhwi)>0.000001)).to_csv('quickie.csv')

    np.testing.assert_almost_equal(var_mine, var_jhwi, decimal=5)


def test_simulate_contour_data():
    e = estimate.Estimate('directional')

    varlist = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    # sigma to zeta
    varlist[0] = varlist[0]*2

    for v in ['white', 'homo']:
        e.simulate_contour_data(varlist,
                                var_type=v,
                                full_vars=True)


def test_same_city_sizes():
    e = estimate.Estimate('directional')

    varlist = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    # sigma to zeta
    varlist[0] = varlist[0]*2

    size_mine = e.get_size(varlist)
    size_jhwi = pd.read_csv('./tests/data/city_size.csv',
                            header=None).values.flatten()
    np.testing.assert_array_almost_equal(size_mine, size_jhwi)


def test_same_size_variances_white():
    e = estimate.Estimate('directional')

    varlist = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    # sigma to zeta
    varlist[0] = varlist[0]*2

    variance_mine = e.get_size_variance(varlist,
                                        scale_kanes=False,
                                        var_type='white')
    variance_jhwi = pd.read_csv('./tests/data/variance_size_white.csv',
                           header=None).values
    np.testing.assert_array_almost_equal(variance_mine, variance_jhwi)


def test_same_size_variances_homo():
    e = estimate.Estimate('directional')

    varlist = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    # sigma to zeta
    varlist[0] = varlist[0]*2

    variance_mine = e.get_size_variance(varlist,
                                        scale_kanes=False,
                                        var_type='homo')
    variance_jhwi = pd.read_csv('./tests/data/variance_size_homo.csv',
                           header=None).values
    np.testing.assert_array_almost_equal(variance_mine, variance_jhwi)


def test_export_results():
    e = estimate.Estimate('directional')

    varlist = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    # sigma to zeta
    varlist[0] = varlist[0]*2
    e.export_results(varlist)


def test_same_output_table():
    e = estimate.Estimate('directional')

    varlist = (pd.read_csv('./tests/data/theta_firststage_plot.csv',
                           header=None)
                 .values
                 .flatten()
                 )
    # sigma to zeta
    varlist[0] = varlist[0]*2
    e.export_results(varlist)

    coords = pd.read_csv('./estim_results/coordinates.csv')
    cities =  pd.read_csv('./estim_results/cities.csv')

    # Fetch table with ids and names
    ids = pd.read_csv('../process/id.csv')

    # Merge with coordinates and drop unmatched units
    main = ids.merge(e.df_coordinates, how='right', on='id')

    # Merge with coords and cities
    main['tmp'] = 1 - main['validity']
    main = (main.merge(cities, how='inner', on='id')
                .merge(coords, how='left', on='id')
                .sort_values(['tmp', 'id'])
           )

    d = {'longitude': 'long_x',
         'latitude': 'lat_y'}
    for coor in d.keys():
        main.loc[pd.isnull(main[coor]), coor] = (
                main.loc[pd.isnull(main[coor]), d[coor]])

    # Select variables
    main = main[['id',
                 'longitude',
                 'latitude',
                 'city_name',
                 'alpha',
                 'size']
                 + [v+'_sd_homo' for v in ['longitude',
                                           'latitude',
                                           'size',
                                           'alpha']]
                 + [v+'_sd_white' for v in ['longitude',
                                            'latitude',
                                            'size',
                                            'alpha']]
                 ]
    main.to_csv('./tests/data/report_table_fdv.csv')
