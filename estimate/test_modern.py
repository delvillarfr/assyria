'''
Tests of Model Parameter Estimation
'''
import estimate
import pandas as pd
import numpy as np

pd.set_option('precision',15)
accuracy = 8

def test_iticount_imports_is_composed_solely_of_cities_in_coordinates():
    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)
        cities_i = np.sort( e.df_iticount['i_id'].unique() )
        np.testing.assert_array_equal( cities_i,
                                       np.sort(e.df_coordinates['id'].values) )


def test_iticount_exports_is_composed_solely_of_cities_in_coordinates():
    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)
        cities_j = np.sort( e.df_iticount['j_id'].unique() )
        np.testing.assert_array_equal( cities_j,
                                       np.sort(e.df_coordinates['id'].values) )


def test_iticount_has_all_possible_city_combinations():
    ''' Meaningful only if previous two tests pass '''
    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)
        assert len(e.df_iticount) == (len(e.df_coordinates)
                                      * (len(e.df_coordinates)-1))


def test_same_bounds_to_ipopt_dir():
    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)
        bounds = e.get_bounds()

        bounds_jhwi = pd.read_csv('./tests/modern/'+s+'/data/bounds_dir.csv')
        bounds_jhwi = bounds_jhwi.replace(np.inf, 1.0e20)

        np.testing.assert_array_almost_equal(np.array(bounds[0]),
                                             bounds_jhwi['lb'].values.flatten(),
                                             decimal=accuracy)
        np.testing.assert_array_almost_equal(np.array(bounds[1]),
                                             bounds_jhwi['ub'].values.flatten(),
                                             decimal=accuracy)


def test_same_initial_condition_dir():
    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)
        theta0_mine = e.initial_cond()
        # zeta is one half sigma
        theta0_mine[0] = theta0_mine[0]/2
        theta0_jhwi = pd.read_csv('./tests/modern/'+s+'/data/theta0_dir.csv')
        np.testing.assert_array_almost_equal(theta0_mine,
                                             theta0_jhwi.values.flatten(),
                                             decimal=accuracy)


def test_same_s_ij_model_directional_dir():
    a_data = pd.read_csv('./tests/modern/a_rand.csv', header=None)
    sigma_data = pd.read_csv('./tests/modern/sigma_rand.csv', header=None)

    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)
        distances = e.df_iticount['dist'].values

        sij_mine = np.empty((e.num_cities*(e.num_cities-1), 100))
        for i in range(100):
            sigma_guess = sigma_data.iloc[i, :].values
            zeta_guess = sigma_guess*2.0
            a_guess = a_data.iloc[i, : e.num_cities].values
            sij_mine[:, i] = e.s_ij_model(zeta_guess, a_guess, distances)
        sij_jhwi = pd.read_csv('./tests/modern/'+s+'/data/sij_rand_dir.csv')
        np.testing.assert_array_almost_equal(sij_mine,
                                             sij_jhwi,
                                             decimal=accuracy)


def test_same_sqerr_sum_dir():
    a_data = pd.read_csv('./tests/modern/a_rand.csv', header=None)
    sigma_data = pd.read_csv('./tests/modern/sigma_rand.csv', header=None)

    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)

        sqerr_mine = np.ones(1000)
        for i in range(1000):
            sigma_guess = sigma_data.iloc[i, :].values
            zeta_guess = [sigma_guess*2.0]
            a_guess = a_data.iloc[i, : e.num_cities].tolist()
            sqerr_mine[i] = e.sqerr_sum(zeta_guess + a_guess)

        sqerr_jhwi = pd.read_csv('./tests/modern/'+s+'/data/sqerr_rand_dir.csv').values
        np.testing.assert_array_almost_equal(sqerr_mine,
                                             sqerr_jhwi.flatten(),
                                             decimal=accuracy)


#def test_same_gradients_at_initial_coords_dir():
#    e = estimate.EstimateModern('directional')
#    varlist = e.initial_cond(full_vars=True)
#    grad_mine = e.grad_full_vars(varlist)
#
#    grad_jhwi = pd.read_csv('./tests/modern/'+s+'/data/gradient0_dir.csv')
#    grad_jhwi = grad_jhwi.values.flatten()
#
#    # sigma is one half zeta
#    grad_jhwi[0] = grad_jhwi[0]/2
#
#    np.testing.assert_array_almost_equal(grad_mine,
#                                         grad_jhwi,
#                                         decimal=accuracy)


#def test_same_error_jacobian_full_vars():
#    e = estimate.EstimateModern('directional')
#    results = (pd.read_csv('./tests/modern/'+s+'/data/theta_firststage_plot.csv',
#                           header=None)
#                 .values
#                 .flatten()
#                 )
#    print(results)
#    # sigma to zeta
#    results[0] = results[0]*2
#
#    jac_mine = e.jac_errors_full_vars(np.float64(results))
#
#    jac_jhwi = pd.read_csv('./tests/modern/'+s+'/data/jacobian_errors_full.csv',
#                           header=None).values
#    # first column must be twice the size of mine
#    jac_jhwi[:, 0] = jac_jhwi[:, 0]/2
#    np.testing.assert_almost_equal(jac_mine, jac_jhwi, decimal=accuracy)
#
#
#
#def test_same_variance_matrix():
#    ''' One value is different with 7 decimals... '''
#    e = estimate.EstimateModern('directional')
#
#    varlist = (pd.read_csv('./tests/modern/'+s+'/data/theta_firststage_plot.csv',
#                           header=None)
#                 .values
#                 .flatten()
#                 )
#    # sigma to zeta
#    varlist[0] = varlist[0]*2
#    var_mine = e.get_variance(np.float64(varlist), full_vars = True)
#
#    var_jhwi = pd.read_csv('./tests/modern/'+s+'/data/variance_white.csv',
#                           header=None).values
#    # Correct variance for first term
#    var_jhwi[0, :] = 2*var_jhwi[0, :]
#    var_jhwi[:, 0] = 2*var_jhwi[:, 0]
#
#    #pd.DataFrame( (var_mine - var_jhwi)*(np.abs(var_mine -
#    #    var_jhwi)>0.000001)).to_csv('quickie.csv')
#
#    np.testing.assert_almost_equal(var_mine, var_jhwi, decimal=5)
#
#
#def test_simulate_contour_data():
#    e = estimate.EstimateModern('directional')
#
#    varlist = (pd.read_csv('./tests/modern/'+s+'/data/theta_firststage_plot.csv',
#                           header=None)
#                 .values
#                 .flatten()
#                 )
#    # sigma to zeta
#    varlist[0] = varlist[0]*2
#
#    e.simulate_contour_data(varlist, full_vars=True)
#
#
#def test_same_city_sizes():
#    e = estimate.EstimateModern('directional')
#
#    varlist = (pd.read_csv('./tests/modern/'+s+'/data/theta_firststage_plot.csv',
#                           header=None)
#                 .values
#                 .flatten()
#                 )
#    # sigma to zeta
#    varlist[0] = varlist[0]*2
#
#    size_mine = e.get_size(varlist)
#    ## Unpack arguments
#    #zeta = varlist[0]
#
#    #i = e.div_indices[True]
#    #lng_guess = varlist[i['long_s']: i['long_e']]
#    #lat_guess = varlist[i['lat_s']: i['lat_e']]
#    #alpha = varlist[i['a_s']:]
#
#    #size_mine = e.get_size(zeta,
#    #                       alpha,
#    #                       e.fetch_dist(lat_guess,
#    #                                    lng_guess,
#    #                                    True)
#    #                      )
#    size_jhwi = pd.read_csv('./tests/modern/'+s+'/data/city_size.csv',
#                            header=None).values.flatten()
#    np.testing.assert_almost_equal(size_mine, size_jhwi)
#
#
#def test_reason_for_nan_in_IPOPT():
#    e = estimate.EstimateModern('directional')
#    input_short = pd.read_csv('./tests/modern/'+s+'/data/nan_input2.csv',
#                              header=None).values.flatten()
#
#    i = e.div_indices[False]
#    input_long = np.concatenate(([input_short[0], 4],
#                                 e.df_known['long_x'].values.flatten(),
#                                 input_short[i['long_s']: i['long_e']],
#                                 e.df_known['lat_y'].values.flatten(),
#                                 input_short[i['lat_s']: i['lat_e']],
#                                 input_short[i['a_s']:]
#                               ))
#
#    # Unpack arguments
#    zeta = input_long[0]
#
#    i = e.div_indices[True]
#    lng_guess = input_long[i['long_s']: i['long_e']]
#    lat_guess = input_long[i['lat_s']: i['lat_e']]
#    alpha = input_long[i['a_s']:]
#
#    pd.DataFrame(np.column_stack((lng_guess, lat_guess)),
#             columns=['lng', 'lat']).to_csv('./tests/modern/'+s+'/data/coordinates_nan.csv')
#
#    #assert len(lat_guess) == len(lng_guess)
#
#    s_ij_model = e.s_ij_model(zeta,
#                                 alpha,
#                                 e.fetch_dist(lat_guess,
#                                                 lng_guess,
#                                                 True)
#                                )
#
#    print('----------')
#    print(e.fetch_dist(lat_guess, lng_guess, True))
#    print('----------')
#    print(e.get_errors(input_short))
#    print('----------')
#    print(e.get_errors(input_long, full_vars=True))
#    print('----------')
#    print(e.sqerr_sum(input_short))
#    print('----------')
#    print(e.sqerr_sum(input_long, full_vars=True))
#    print('----------')
#    print(e.grad(input_short))
#    print('----------')
#    print(e.grad_full_vars(input_long))
#    print('----------')
#    e.solve(input_short, solver='mumps')
#    #e.solve(input_long, full_vars=True, solver='mumps')
#    #k
#
#def test_bounds_to_ipopt_dir():
#    e = estimate.EstimateModern('directional')
#    replaced_no = e.replace_id_coord(e.df_constr_stat, no_constr=True)
#    bounds = e.get_bounds(replaced_no,
#                          full_vars=True,
#                          set_elasticity=5.0)
#    print(bounds[0])
#    print(bounds[1])
