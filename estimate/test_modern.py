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


def test_same_variance_matrix():
    ''' One value is different with 7 decimals... '''
    for s in ['all', 'ancient_system', 'matched']:
        for v in ['white', 'homo']:
            e = estimate.EstimateModern('directional', s)

            varlist = (pd.read_csv('./tests/modern/'+s+'/data/theta_firststage.csv',
                                   header=None)
                         .values
                         .flatten()
                         )
            #var_mine = e.get_variance(np.float64(varlist), full_vars = True)
            var_mine = e.get_variance(varlist, var_type=v)

            var_jhwi = pd.read_csv('./tests/modern/'+s+'/data/variance_'+v+'.csv',
                                   header=None).values
            # Correct variance for first term
            var_jhwi[0, :] = 2.0*var_jhwi[0, :]
            var_jhwi[:, 0] = 2.0*var_jhwi[:, 0]

            #pd.DataFrame( (var_mine - var_jhwi)*(np.abs(var_mine -
            #    var_jhwi)>0.000001)).to_csv('quickie.csv')

            np.testing.assert_almost_equal(var_mine, var_jhwi, decimal=5)


def test_same_city_sizes():
    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)

        varlist = (pd.read_csv('./tests/modern/'+s+'/data/theta_firststage.csv',
                               header=None)
                     .values
                     .flatten()
                     )

        size_mine = e.get_size(varlist)
        size_jhwi = pd.read_csv('./tests/modern/'+s+'/data/city_size_dir.csv',
                                header=None).values.flatten()
        np.testing.assert_almost_equal(size_mine, size_jhwi)


def test_refresh():
    for s in ['all', 'ancient_system', 'matched']:
        e = estimate.EstimateModern('directional', s)

        varlist = (pd.read_csv('./tests/modern/'+s+'/data/theta_firststage.csv',
                               header=None)
                     .values
                     .flatten()
                     )
        e.export_results(varlist)
