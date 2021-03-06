# -*- coding: utf-8 -*-

""" Parameter Estimation

This module provides all functions to estimate the model parameters.
"""

import ConfigParser


import pandas as pd
import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd import hessian
import ipopt

import sys



# Configuration
#if __name__ == "__main__":
## execute only if run as a script
config = ConfigParser.ConfigParser()

## file keys.ini should be in process.py parent directory.
config.read('../keys.ini')



class Loader(object):
    """ Class to load datasets for lost cities exercise.

    It provides methods to get the iticount, coordinates and constraints
    datasets with a chosen set of known and lost cities.

    Args:
        build_type (str): One of "directional" or "non-directional".
    """

    def __init__(self):
        pass


    def random_certs(self, num_cities, lost_cities):
        """ Generate random cert with fixed number of lost cities.

        In practice, what is required is an array with `lost_cities` 3s
        randomly chosen.

        Args:
            lost_cities (int): The number of lost cities.

        Returns:
            (np.ndarray): The array of certs.
        """
        certs = np.zeros(num_cities)
        lost_i = np.random.choice(num_cities,
                                  size = lost_cities,
                                  replace = False)
        certs[lost_i] = 3

        return certs


    def reindex_coordinates(self, certs, coords):
        """ Reindex coordinates

        Args:
            certs (np.ndarray): The array of location confidence. If
                equal to 3, the corresponding is considered lost.
            coords (pandas.DataFrame): The dataframe of coordinates.

        Returns:
            pandas.DataFrame: The resorted dataframe of coordinates with
            new cert.
        """
        coords['cert'] = certs
        coords = coords.sort_values('cert')
        coords['id_shuff'] = range(len(coords))

        return coords


    def reindex_iticount(self,
                         iticount,
                         coords,
                         merge_key='id',
                         notation='fdv'):
        """ Reindex and re-sort iticount data.

        The sorting is based on the new id_shuff, first on j, then on i.

        Args:
            iticount (pandas.DataFrame): The iticount data.
                notation (str): The convention for column names in iticount.
                It can be either 'fdv' or 'jhwi'.

        Returns:
            pandas.DataFrame: The reindexed, resorted iticount data.
        """
        # Keep only relevant vars
        c = coords[[merge_key, 'id_shuff']]
        itikeys = {'fdv': [merge_key+'_i', merge_key+'_j'],
                   'jhwi': ['i_'+merge_key, 'j_'+merge_key]}
        for i in itikeys[notation]:
            iticount = pd.merge(iticount,
                                c,
                                how = 'left',
                                left_on = i,
                                right_on = merge_key)
        iticount = iticount.rename({'id_shuff_x': 'id_shuff_i',
                                    'id_shuff_y': 'id_shuff_j'}, axis=1)

        return iticount.sort_values(['id_shuff_j', 'id_shuff_i'])


    def reindex_constr(self, constraint, coords, merge_key='id'):
        """ Reindex and re-sort constraints data.

        Args:
            constraint (pandas.DataFrame): The constraints data.

        Returns:
            pandas.DataFrame: The reindexed, resorted constraints data.
        """
        # Keep only relevant vars
        c = coords[[merge_key, 'id_shuff']]
        constraint = pd.merge(constraint,
                              c,
                              how = 'left',
                              on = merge_key)
        return constraint.sort_values('id_shuff')


class LoaderNew(object):
    """ Class to load datasets for given lost and known cities.

    It provides methods to get the iticount, coordinates and constraints
    datasets with a chosen set of known and lost cities.
    """

    def __init__(self):
        pass


    def random_certs(self, n_cities, lost_cities):
        """ Generate random cert with fixed number of lost cities.

        Args:
            n_cities (int): The total number of cities.
            lost_cities (int): The number of lost cities.

        Returns:
            (np.ndarray): The array of certs. If an entry is equal to 3, the
            corresponding city is lost. Otherwise, it is known.
        """
        certs = np.zeros(n_cities)
        lost_i = np.random.choice(n_cities,
                                  size = lost_cities,
                                  replace = False)
        certs[lost_i] = 3

        return certs


    def reindex_coordinates(self, certs, coords):
        """ Reindex coordinates

        Args:
            certs (np.ndarray): The one-dimensional array of location
                confidence. If equal to 3, the corresponding city is
                considered lost.
            coords (pandas.DataFrame): The dataframe of coordinates.

        Returns:
            pandas.DataFrame: The sorted dataframe of coordinates with
                new cert. It is sorted by cert and then by id.
        """
        coords['cert'] = certs
        coords = coords.sort_values(['cert', 'id'])
        coords['id_shuff'] = range(len(coords))

        return coords


    def reindex_iticount(self, iticount, coords):
        """ Reindex and re-sort iticount data.

        The sorting is based on the new id_shuff, first on j, then on i.

        Args:
            iticount (pandas.DataFrame): The iticount data.
                notation (str): The convention for column names in iticount.
                It can be either 'fdv' or 'jhwi'.

        Returns:
            pandas.DataFrame: The reindexed, resorted iticount data.
        """
        # Keep only relevant vars
        c = coords[['id', 'id_shuff']]

        for i in ['id_i', 'id_j']:
            iticount = pd.merge(iticount,
                                c,
                                how = 'left',
                                left_on = i,
                                right_on = 'id')
        iticount = iticount.rename(index = str,
                                   columns = {'id_shuff_x': 'id_shuff_i',
                                              'id_shuff_y': 'id_shuff_j'})

        return iticount.sort_values(['id_shuff_j', 'id_shuff_i'])


    def reindex_constr(self, constraint, coords):
        """ Reindex and re-sort constraints data.

        Args:
            constraint (pandas.DataFrame): The constraints data.

        Returns:
            pandas.DataFrame: The reindexed constraints data without known
                cities.
        """
        # Keep only relevant vars
        c = coords[['id', 'id_shuff', 'cert']]
        constraint = c.merge(constraint,
                             how = 'left',
                             on = 'id')

        #constraint = pd.merge(constraint,
        #                      c,
        #                      how = 'left',
        #                      on = 'id')
        constraint = (constraint.loc[constraint['cert'] == 3]
                                .sort_values('id_shuff'))
        return constraint




class EstimateBase(object):
    """ Base Class for main estimation procedures.

    Defines attributes and methods used later for
    * Estimation with directional and non-directional ancient data
    * Estimation with directional and non-directional modern (all,
        ancient_system and ancient_system_matched) data.

    ... and more

    * saves the gradient of the objective function as attribute, to avoid
        calling autograd multiple times.
    * saves the jacobian of errors
    Args:
        build_type (str): One of "directional" or "non-directional".
    """

    def __init__(self, build_type, omega = None):
        self.build_type = build_type
        self.omega = omega

        # Automatic differentiation of objective and errors

        ## Errors
        def error_autograd(v):
            return self.get_errors(v)
        self.jac_errors = jacobian(error_autograd)

        ## Objective (assume full_vars is False)
        def objective(varlist):
            """ This is the formulation for autograd. """
            return self.sqerr_sum(varlist)
        self.grad = grad(objective)

        ## With Feasible Generalized Nonlinear Least Squares
        if omega is not None:
            self.omega_inv = np.linalg.inv(omega)
            def objective_gen(varlist):
                """ This is the formulation for autograd. """
                return self.sqerr_sum_gen(varlist, self.omega_inv)
            self.grad_gen = grad(objective_gen)


    def haversine_approx(self, coord_i, coord_j):
        """ Compute distances from 2 coordinates arrays.

        The distances are computed using the approximation to the Haversine
        formula discussed in the paper.

        Args:
            coord_i (np.ndarray): The first set of coordinates. It must have
                the latitude in column 0 and the longitude in column 1.
            coord_j (np.ndarray): The second set of coordinates.

        Returns:
            np.ndarray: The one-dimensional array of distances.
        """
        factor_out = 10000.0/90
        factor_in = np.cos(37.9 * np.pi / 180)

        lat_diff = coord_j[:, 0] - coord_i[:, 0]
        lng_diff = coord_j[:, 1] - coord_i[:, 1]

        diff = np.column_stack((lat_diff, factor_in * lng_diff))

        # Re-declare object dtype: numpy bug
        # https://stackoverflow.com/questions/18833639/attributeerror-in-python-numpy-when-constructing-function-for-certain-values
        #return factor_out * np.sqrt( np.float64(np.sum(diff**2, axis=1)) )
        return factor_out * np.sqrt( np.sum(diff**2, axis=1) )


    def tile_nodiag(self, arr, default):
        """ Tile a 1-dimensional array avoiding entry i in repetition i.

        The array is tiled `self.num_cities` times.

        To increase execution speed, the indices to extract from tiled array
        have been pre-specified in `__init__`.

        Args:
            arr (np.ndarray): A 1-dim array that should be of length
                `self.num_cities` if default == True.
            default (bool): It is either True of False. If True,
                ``self.index_nodiag`` is used for speed. If false, this
                fancy index is computed on the spot.

        Returns:
            np.ndarray: an array repeating `arr` the number of times given
            by `self.num_cities`, but extracting value in index j on
            repetition j.

        Examples:
            >>> self.tile_nodiag(arr)
            array([2, 3, 1, 3, 1, 2])
        """
        if default:
            n_cities = self.num_cities
            fancy_i = self.index_nodiag
        else:
            n_cities = len(arr)
            # The fancy index selects off-diagonal elements.
            i = np.repeat(np.arange(1, n_cities), n_cities)
            fancy_i = i + np.arange(n_cities*(n_cities - 1))

        arr_tiled = np.tile(arr, n_cities)
        return arr_tiled[fancy_i]


    def coord_combinations(self, lat, lng, default):
        """ Form all different coordinate combinations.

        Args:
            lat (np.ndarray or list): A 1-dimensional array of longitudes.
            lng (np.ndarray or list): A 1-dimensional array of latitudes.

        Returns:
            tuple: Two arrays with latitudes and longitudes. Row `i` of each
            array is a pair of different coordinates. If there are `n`
            different coordinates, each array has `n(n-1)` rows.
        """
        n_coords = len(lat)
        assert n_coords == len(lng)

        coord_j = np.column_stack((
            np.repeat(lat, n_coords - 1),
            np.repeat(lng, n_coords - 1)
        ))
        coord_i = np.column_stack((
            self.tile_nodiag(lat, default),
            self.tile_nodiag(lng, default)
        ))

        return (coord_i, coord_j)


    def get_coordinate_pairs(self, lat_guess, lng_guess, full_vars=False):
        """ Forms coordinates of all pairs of different locations.

        This function leverages that

        * `self.df_iticount` is sorted according to `id_jhwi_j` first and then
            by `id_jhwi_i`.
        * `self.df_coordinates` is sorted according to `id_jhwi`.

        Args:
            lat_guess (np.ndarray): The 1-dimensional array of latitudes.
            lng_guess (np.ndarray): The 1-dimensional array of longitudes.
            full_vars (bool): If True, the known city coordinates are assumed
                to be included.
        """
        if full_vars:
            lats = lat_guess
            longs = lng_guess
        else:
            lats = np.concatenate((self.df_known['lat_y'].values, lat_guess))
            longs = np.concatenate((self.df_known['long_x'].values, lng_guess))

        return self.coord_combinations(lats, longs, default = True)


    def fetch_dist(self, lat_guess, lng_guess, full_vars=False):
        """ Compute the distances of all pairs of different locations.

        Calls `self.get_coordinate_pairs` uses its output to call
        `self.haversine_approx`.
        """
        coords = self.get_coordinate_pairs(lat_guess, lng_guess, full_vars)
        return self.haversine_approx( coords[0], coords[1] )


    def s_ij_model(self, zeta, alpha, distances):
        """ Compute the model-predicted trade shares.

        The idea is to cast elements as matrix, add over `axis=1`,
        repeat the result by the number of cities less one, and divide
        elements by this new 1-dim array.

        Args:
            zeta (float): The distance elasticity of trade.
            alpha (np.ndarray): City-specific alphas.
            distances (np.ndarray): Contains distances between all j, i
                pairs of cities, excluding j, j pairs.

        Returns:
            np.ndarray: The model-predicted trade shares.
        """
        a = self.tile_nodiag(alpha, default = True)
        elems = a * (distances ** (-zeta))

        denom = np.reshape(elems, (self.num_cities, self.num_cities - 1))
        denom = np.sum(denom, axis = 1)
        denom = np.repeat(denom, self.num_cities-1)

        return elems / denom


    def get_errors(self, varlist, full_vars=False):
        """ Get the model and data trade share differences.

        Args:
            varlist (np.ndarray):it is composed of
                `[zeta, lng_guess, lat_guess, alpha]`.

        Returns:
            np.ndarray: the difference between data and model trade shares.
        """
        # Unpack arguments
        zeta = varlist[0]

        i = self.div_indices[full_vars]
        lng_guess = varlist[i['long_s']: i['long_e']]
        lat_guess = varlist[i['lat_s']: i['lat_e']]
        alpha = varlist[i['a_s']:]

        #assert len(lat_guess) == len(lng_guess)

        s_ij_model = self.s_ij_model(zeta,
                                     alpha,
                                     self.fetch_dist(lat_guess,
                                                     lng_guess,
                                                     full_vars)
                                    )
        return self.shares - s_ij_model


    def sqerr_sum(self, varlist, full_vars=False):
        """ Gets the sum of squared errors.

        This is the objective function.

        Returns:
            numpy.float64: The value of the objective function given the data
            and model trade shares.
        """
        errors = self.get_errors(varlist, full_vars)
        return np.dot(errors, errors)


    def sqerr_sum_gen(self, varlist, omega_inv, full_vars = False):
        """ Gets the weighted sum of squared errors.

        This is the objective function for the second stage estimation via
        Feasible Generalized Nonlinear Least Squares, as depicted by Cameron
        and Trivedi (2005), p. 156.

        Args:
            omega_inv (np.ndarray): The weighting matrix in the quadratic
                form.

        Returns:
            numpy.float64: The value of the quadratic form given the data, the
                model trade shares, and the inverse Omega matrix.
        """
        errors = self.get_errors(varlist, full_vars)
        part_1 = np.dot(errors, omega_inv)
        return np.dot(part_1, errors)


    def full_to_short_i(self):
        """ Get the indices of elements of short `varlist` from full `varlist`.

        Returns:
            np.ndarray: the indices to select short varlist from full
                varlist
        """
        i = self.div_indices[True]
        res = np.concatenate(([0],
                              range(i['long_unknown_s'], i['long_e']),
                              range(i['lat_unknown_s'], i['lat_e']),
                              range(i['a_s'], i['a_e']))
                            )
        return res


    def short_to_jhwi(self, short):
        """ Transform short varlist to varlist in Jhwi style

        Takes a varlist in short format,
        e.g.  v = (zeta, lng_unknown, lat_unknown, alphas)
        and transforms it to Jhwi format:
        v'
        = (zeta, useless, lng_known, lng_unknown, lat_known, lat_unknown, alpha)

        Args:
            short (np.ndarray): The varlist in short format.

        Returns:
            np.ndarray: The varlist in Jhwi format.

        Warning:
            zeta is not transformed into sigma (= 2 zeta)
        """
        # Extract components from short
        i = self.div_indices[False]
        zeta = short[0]
        lng_unknown = short[i['long_s']: i['long_e']].flatten()
        lat_unknown = short[i['lat_s']: i['lat_e']].flatten()
        alphas = short[i['a_s']: i['a_e']].flatten()

        # Get known coordinates
        lng_known = self.df_known['long_x'].values
        lat_known = self.df_known['lat_y'].values

        return np.concatenate(([zeta, 2],
                               lng_known,
                               lng_unknown,
                               lat_known,
                               lat_unknown,
                               alphas))


    def initial_cond(self,
                     len_sim=None,
                     perturb=None,
                     perturb_type='rigid',
                     full_vars=False,
                     suggestion=None):
        """ Gets initial condition(s) for `IPOPT`.

        Args:
            len_sim (int): Specifies the number of initial conditions to draw.
            perturb (float): A percentage deviation from the default initial
                value given in `self.df_coordinates`.
            perturb_type (str): Type of perturbation on the default initial
                value. If `'rigid'` then the whole initial value vector is
                multiplied by a scalar. If `'flexible'` then each element of
                the initial value vector is multiplied by a different scalar.
                Default is `'rigid'`.
            suggestion (np.ndarray): A custom initial value. It must be
                coherent with `full_vars`.

        Returns:
            np.ndarray: The default initial condition if perturb is not
                specified, and an array with `len_sim` perturbed initial
                conditions.
        """
        if suggestion is None:
            # Form default initial value
            zeta = [2.0]
            alphas = np.ones(self.num_cities)
            if full_vars:
                # add tilde_delta0
                zeta = zeta + [2.0]
                lats = self.df_coordinates['lat_y'].values
                longs = self.df_coordinates['long_x'].values
            else:
                lats = self.df_unknown['lat_y'].values
                longs = self.df_unknown['long_x'].values
            x0 = np.concatenate((zeta, longs, lats, alphas))
        else:
            x0 = suggestion

        # Perturb it
        if perturb != None:
            x0 = np.tile(x0, (len_sim, 1))

            if perturb_type == 'rigid':
                p = np.random.uniform(1-perturb, 1+perturb, size=(len_sim, 1))
            elif perturb_type == 'flexible':
                p = np.random.uniform(1-perturb,
                                      1+perturb,
                                      size=(len_sim, x0.shape[1]))
            #print(p)
            x0 = x0*p

        return x0


    def get_variance_gmm(self, varlist, full_vars=False):
        """ Get the GMM variance-covariance matrix of the estimators

        Applies standard GMM formula. This function needs to be revised.
        """
        errors = self.get_errors(varlist, full_vars)

        # Make column vector
        errors = np.expand_dims(errors, 1)

        # Evaluate errors jacobian at estimated parameter.
        jac = self.jac_errors(varlist)

        # Remove fixed a.
        jac = np.delete(jac, self.index_normalized, axis=1)

        # Remove fixed a
        index_norm = self.div_indices[full_vars]['a_s'] + self.id_normalized
        jac = np.delete(jac, index_norm, axis=1)

        #assert np.shape(jac) == (650, 48)

        # Build variance-covariance matrix
        bread_top = np.linalg.inv(np.dot(np.transpose(jac), jac))
        ham = np.linalg.multi_dot((np.transpose(jac),
                                   errors,
                                   np.transpose(errors),
                                   jac))
        bread_bottom = bread_top

        return np.linalg.multi_dot((bread_top, ham, bread_bottom))


    def get_jacobian(self,
                     varlist,
                     var_type='white',
                     zeta_fixed=False,
                     full_vars=False):
        """ Compute the variance-covariance matrix of the estimators.

        It can be computed according to the White formula, or with
        homoskedasticity.

        Args:
            var_type (str): One of 'white' or 'homo', or 'gmm'.

        Returns:
            np.ndarray: The variance-covariance matrix of the estimators.
        """
        if full_vars:
            i = self.div_indices[True]
            jac = self.jac_errors_full_vars(varlist)
            jac = pd.DataFrame(jac)
            jac = jac.drop(columns = ([1]
                           + range(i['long_s'], i['long_unknown_s'])
                           + range(i['lat_s'], i['lat_unknown_s'])))
            jac = jac.values
        else:
            # Evaluate errors jacobian at estimated parameter.
            jac = self.jac_errors(varlist)

        # Remove fixed a
        index_norm = self.div_indices[False]['a_s'] + self.id_normalized
        jac = np.delete(jac, index_norm, axis=1)

        # If zeta is fixed, remove it.
        if zeta_fixed:
            jac = np.delete(jac, 0, axis=1)

        return jac


    def get_variance(self,
                     varlist,
                     var_type='white',
                     zeta_fixed=False,
                     full_vars=False):
        """ Compute the variance-covariance matrix of the estimators.

        It can be computed according to the White formula, or with
        homoskedasticity.

        Args:
            var_type (str): One of 'white' or 'homo', or 'gmm'.

        Returns:
            np.ndarray: The variance-covariance matrix of the estimators.
        """
        if full_vars:
            i = self.div_indices[True]
            jac = self.jac_errors_full_vars(varlist)
            jac = pd.DataFrame(jac)
            jac = jac.drop(columns = ([1]
                           + range(i['long_s'], i['long_unknown_s'])
                           + range(i['lat_s'], i['lat_unknown_s'])))
            jac = jac.values
        else:
            # Evaluate errors jacobian at estimated parameter.
            jac = self.jac_errors(varlist)

        # Remove fixed a
        index_norm = self.div_indices[False]['a_s'] + self.id_normalized
        jac = np.delete(jac, index_norm, axis=1)

        # If zeta is fixed, remove it.
        if zeta_fixed:
            jac = np.delete(jac, 0, axis=1)

        errors = self.get_errors(varlist, full_vars)

        # Build variance-covariance matrix, according to var_type
        scale = 1.0
        bread = np.linalg.inv(scale * np.dot( np.transpose(jac), jac ))
        if var_type == 'gmm':
            #middle = np.matmul(scale*np.expand_dims(errors, 1),
            #                   scale*np.expand_dims(errors, 0))
            middle = np.outer(scale*errors, scale*errors)
            ham = np.linalg.multi_dot((np.transpose(jac), middle, jac))
            return np.linalg.multi_dot((bread, ham, bread))

        elif var_type == 'white':
            middle = np.diag((scale*errors)**2)
            ham = np.linalg.multi_dot((np.transpose(jac), middle, jac))
            ## Make column vector to multiply row-wise
            #errors = np.expand_dims(errors, 1)
            #ham = np.dot(np.transpose(jac * errors), jac * errors)
            return np.linalg.multi_dot((bread, ham, bread))

        elif var_type == 'homo':
            return scale * (np.sum(errors**2) / len(errors)) * bread

        else:
            raise ValueError("Please specify the variance type to be one of "
                    + "'white' or 'homo'")


    def simulate_contour_data(self,
                              varlist,
                              size=20000,
                              var_type='white',
                              full_vars=False):
        """ Simulates contour data using the estimation results.

        Draws values from a normal distribution with mean equal to the
        estimated parameters and variance-covariance matrix given by
        `self.get_variance`.

        Args:
            varlist (np.ndarray): The mean. It should be the estimated
                vector of parameters.
            size (int): The number of draws from the normal distribution to
                get.

        Returns:
            np.ndarray
        """
        cov = self.get_variance(varlist,
                                var_type=var_type,
                                full_vars=full_vars)

        mean = varlist.copy()
        if full_vars:
            mean = mean[self.full_to_short_i()]
        # Remove fixed a from mean
        index_norm = self.div_indices[False]['a_s'] + self.id_normalized
        mean = np.delete(mean, index_norm)

        sims = np.random.multivariate_normal(mean, cov, size)

        # Select only unknown coordinates
        i = self.div_indices[False]
        sims = sims[ :, range(i['long_s'], i['long_e'])
                        + range(i['lat_s'], i['lat_e']) ]
        mean = mean[ range(i['long_s'], i['long_e'])
                     + range(i['lat_s'], i['lat_e']) ]

        # add id_lng; id_lat headers
        ids = self.df_unknown['id'].tolist()
        id_header = [i + '_lng' for i in ids] + [i + '_lat' for i in ids]

        df = pd.DataFrame( sims, columns = id_header )
        df.to_csv('./estim_results/plot_data_' + var_type + '.csv',
                  index=False)

        return df


    def get_size(self, varlist, theta=4.0):
        """ Retrieve the fundamental size of cities.

        Recall Size_i is proportional to L_i T_i^(1/theta).

        Args:
            theta (float): The trade elasticity parameter that is assumed away.

        Returns:
            np.ndarray: The fundamental size of cities
        """
        # Unpack arguments
        zeta = varlist[0]
        i = self.div_indices[True]
        lng_guess = varlist[i['long_s']: i['long_e']]
        lat_guess = varlist[i['lat_s']: i['lat_e']]
        alpha = varlist[i['a_s']:]

        distances = self.fetch_dist(lat_guess, lng_guess, True)

        factor_1 = alpha**(1 + 1.0/theta)

        ## Build summation
        # This part draws from self.s_ij_model()
        a = self.tile_nodiag(alpha, default = True)
        elems = a * (distances ** (-zeta))
        elems = np.reshape(elems, (self.num_cities, self.num_cities - 1))
        # Add within-city component. Assumed within-city distance: 30 km.
        own_factor = (30 ** (-zeta)) * alpha
        factor_2 = np.sum(elems, axis = 1).flatten() + own_factor

        sizes = factor_1 * factor_2
        return sizes


    def get_size_variance(self,
                          varlist,
                          var_type='white',
                          zeta_fixed=False):
        """ Get the fundamental size variance-covariance matrix.

        Applies Delta Method to get the variance-covariance matrix of the city
        size estimates.

        Returns:
            np.ndarray: The variance-covariance matrix of city sizes.
        """
        def size_for_grad(v):
            """ get_size function for autograd """
            return self.get_size(v)

        # Get Jacobian
        jac_size = jacobian(size_for_grad)
        # Evaluate
        j = jac_size(varlist)

        # Remove variables that are fixed
        i = self.div_indices[True]
        j = pd.DataFrame(j)
        j = j.drop(columns = ([1]
                       + range(i['long_s'], i['long_unknown_s'])
                       + range(i['lat_s'], i['lat_unknown_s'])))
        j = j.values
        # Remove fixed a
        index_norm = self.div_indices[False]['a_s'] + self.id_normalized
        j = np.delete(j, index_norm, axis=1)

        # If zeta is fixed, remove it.
        if zeta_fixed:
            j = np.delete(j, 0, axis=1)

        var = self.get_variance(varlist,
                                var_type=var_type,
                                zeta_fixed = zeta_fixed,
                                full_vars=True)

        return np.linalg.multi_dot((j, var, np.transpose(j)))


    def get_size_variance2(self,
                           varlist,
                           cov,
                           zeta_fixed=False):
        """ Get the fundamental size variance-covariance matrix.

        Applies Delta Method to get the variance-covariance matrix of the city
        size estimates.

        Returns:
            np.ndarray: The variance-covariance matrix of city sizes.
        """
        def size_for_grad(v):
            """ get_size function for autograd """
            return self.get_size(v)

        # Get Jacobian
        jac_size = jacobian(size_for_grad)
        # Evaluate
        j = jac_size(varlist)

        # Remove variables that are fixed
        i = self.div_indices[True]
        j = pd.DataFrame(j)
        j = j.drop(columns = ([1]
                       + range(i['long_s'], i['long_unknown_s'])
                       + range(i['lat_s'], i['lat_unknown_s'])))
        j = j.values
        # Remove fixed a
        index_norm = self.div_indices[False]['a_s'] + self.id_normalized
        j = np.delete(j, index_norm, axis=1)

        # If zeta is fixed, remove it.
        if zeta_fixed:
            j = np.delete(j, 0, axis=1)

        print(j)
        print(cov)
        print('--------------------')

        return np.linalg.multi_dot((j, cov, np.transpose(j)))


    def export_results(self,
                       varlist,
                       zeta_fixed = False,
                       loc = './estim_results/ancient/'):
        """ Export the estimation results.

        Exports zeta.csv, coordinates.csv, cities.csv, simulation.csv

        Args:
            varlist (np.ndarray): it is in jhwi format:
        `(zeta, useless, long_known, long_unknown, lat_known, lat_unknown, a)`
            loc (str): the directory to save results.
        """
        # 0. Cast varlist as np.float64. See numpy bug in
        # `self.haversine_approx`.
        varlist = np.float64(varlist)

        # 1. Fetch standard error of estimates
        varlist_cov_gmm = self.get_variance(varlist,
                                              var_type='gmm',
                                              zeta_fixed = zeta_fixed,
                                              full_vars=True)
        varlist_cov_white = self.get_variance(varlist,
                                              var_type='white',
                                              zeta_fixed = zeta_fixed,
                                              full_vars=True)
        varlist_cov_homo = self.get_variance(varlist,
                                             var_type='homo',
                                             zeta_fixed = zeta_fixed,
                                             full_vars=True)
        size_cov_gmm = self.get_size_variance(varlist,
                                                zeta_fixed = zeta_fixed,
                                                var_type='gmm')
        size_cov_white = self.get_size_variance(varlist,
                                                zeta_fixed = zeta_fixed,
                                                var_type='white')
        size_cov_homo = self.get_size_variance(varlist,
                                               zeta_fixed = zeta_fixed,
                                               var_type='homo')
        print(varlist_cov_white.shape)

        print(np.prod(np.diag(varlist_cov_white) > 0.0))
        print(np.prod(np.diag(varlist_cov_homo) > 0.0))
        print(np.prod(np.diag(size_cov_white) > 0.0))
        print(np.prod(np.diag(size_cov_white) > 0.0))

        varlist_sd_gmm = np.sqrt( np.diag(varlist_cov_gmm) )
        varlist_sd_white = np.sqrt( np.diag(varlist_cov_white) )
        varlist_sd_homo = np.sqrt( np.diag(varlist_cov_homo) )
        size_sd_gmm = np.sqrt( np.diag(size_cov_gmm) )
        size_sd_white = np.sqrt( np.diag(size_cov_white) )
        size_sd_homo = np.sqrt( np.diag(size_cov_homo) )

        ## Add 0 sd for zeta if zeta is fixed
        if zeta_fixed:
            print('ENTERED')
            varlist_sd_gmm = np.concatenate(([0.0], varlist_sd_gmm))
            varlist_sd_white = np.concatenate(([0.0], varlist_sd_white))
            varlist_sd_homo = np.concatenate(([0.0], varlist_sd_homo))

        # 2. Unpack varlist arguments
        zeta = varlist[0]
        i = self.div_indices[True]
        lng_estim = varlist[i['long_unknown_s']: i['long_e']]
        lat_estim = varlist[i['lat_unknown_s']: i['lat_e']]
        alpha = varlist[i['a_s']:]

        # 3. Save zeta.csv
        df_zeta = pd.DataFrame([[zeta,
                                 varlist_sd_gmm[0],
                                 varlist_sd_white[0],
                                 varlist_sd_homo[0]]],
                               columns=['zeta',
                                        'zeta_sd_gmm',
                                        'zeta_sd_white',
                                        'zeta_sd_homo']
                              )
        df_zeta.to_csv(loc + 'zeta.csv', index=False)

        # 4. Save estimated coordinates + standard errors
        ## Unpack arguments
        j = self.div_indices[False]

        lng_gmm = varlist_sd_gmm[j['long_s']: j['long_e']]
        lng_white = varlist_sd_white[j['long_s']: j['long_e']]
        lng_homo = varlist_sd_homo[j['long_s']: j['long_e']]

        lat_gmm = varlist_sd_gmm[j['lat_s']: j['lat_e']]
        lat_white = varlist_sd_white[j['lat_s']: j['lat_e']]
        lat_homo = varlist_sd_homo[j['lat_s']: j['lat_e']]

        ## Fetch IDs
        ids_coord = self.df_unknown['id'].values
        coord_array = np.column_stack((ids_coord,
                                       lng_estim,
                                       lng_gmm,
                                       lng_white,
                                       lng_homo,
                                       lat_estim,
                                       lat_gmm,
                                       lat_white,
                                       lat_homo))
        cols = ['id',
                'longitude',
                'longitude_sd_gmm',
                'longitude_sd_white',
                'longitude_sd_homo',
                'latitude',
                'latitude_sd_gmm',
                'latitude_sd_white',
                'latitude_sd_homo']
        coordinates = pd.DataFrame(coord_array, columns = cols)
        coordinates = coordinates.merge(self.df_id, how='left', on='id')
        coordinates = coordinates[['city_name'] + cols]
        coordinates.to_csv(loc + 'coordinates.csv', index=False)

        # 5. Save sizes and alphas (+ standard errors)
        size = self.get_size(varlist)

        alpha = varlist[i['a_s']:]
        alpha_gmm = varlist_sd_gmm[j['a_s']:]
        alpha_white = varlist_sd_white[j['a_s']:]
        alpha_homo = varlist_sd_homo[j['a_s']:]

        ## Insert 0 s.e. for fixed alpha (for compat. with jhwi)
        alpha_gmm = np.insert(alpha_gmm, self.id_normalized, 0.0)
        alpha_white = np.insert(alpha_white, self.id_normalized, 0.0)
        alpha_homo = np.insert(alpha_homo, self.id_normalized, 0.0)

        ids_city = self.df_coordinates['id'].values
        city_array = np.column_stack((ids_city,
                                      alpha,
                                      alpha_gmm,
                                      alpha_white,
                                      alpha_homo,
                                      size,
                                      size_sd_gmm,
                                      size_sd_white,
                                      size_sd_homo))
        cols = ['id',
                'alpha',
                'alpha_sd_gmm',
                'alpha_sd_white',
                'alpha_sd_homo',
                'size',
                'size_sd_gmm',
                'size_sd_white',
                'size_sd_homo'
                ]
        cities = pd.DataFrame(city_array, columns = cols)
        cities = cities.merge(self.df_id, how='left', on='id')
        cities = cities[['city_name'] + cols]
        cities.to_csv(loc + 'cities.csv', index=False)

        ## 6. Generate and store contour data, white and homo
        #for v in ['white', 'homo']:
        #    self.simulate_contour_data(varlist,
        #                               var_type=v,
        #                               full_vars=True)


    def format_jhwi(self, cities, coordinates, zeta, loc):
        """ Format results as Jhwi.

        Args:
            cities (pandas.DataFrame): The cities dataframe.
            coordinates (pandas.DataFrame): The coordinates dataframe.
            zeta (pandas.DataFrame): The zeta dataframe.

        Saves csv in jhwi format.
        """
        jhwi = cities.copy()
        jhwi = jhwi.merge(self.df_coordinates, on='id', how='left')

        # Correct type of id_jhwi
        jhwi['id_jhwi'] = jhwi['id_jhwi'].astype(int)

        jhwi = jhwi.merge(coordinates, on='id', how='left')

        # Add known coordinates to latitude and longitude
        jhwi.loc[ pd.isnull(jhwi['longitude']), 'longitude'] = jhwi['long_x']
        jhwi.loc[ pd.isnull(jhwi['latitude']), 'latitude'] = jhwi['lat_y']

        jhwi = jhwi[['id_jhwi',
                     'cert',
                     'long_x',
                     'lat_y',
                     'validity',
                     'city_name_x',
                     'alpha',
                     'alpha_sd_white',
                     'alpha_sd_homo',
                     'longitude',
                     'longitude_sd_white',
                     'longitude_sd_homo',
                     'latitude',
                     'latitude_sd_white',
                     'latitude_sd_homo',
                     'size',
                     'size_sd_white',
                     'size_sd_homo']]

        # Rename
        jhwi = jhwi.rename({'id_jhwi': 'id',
                            'city_name_x': 'name',
                            'longitude': 'varphi_est',
                            'latitude': 'lambda_est',
                            'size': 'T_one_over_vartheta'}, axis = 1)

        dic = {'white': 'homo', 'homo': 'white'}
        for se in dic.keys():
            result_jhwi = jhwi.drop(['alpha_sd_' + dic[se],
                                     'longitude_sd_' + dic[se],
                                     'latitude_sd_' + dic[se],
                                     'size_sd_' + dic[se]], axis = 1)
            result_jhwi = result_jhwi.rename({
                'alpha_sd_' + se: 'alpha_se',
                'longitude_sd_' + se: 'varphi_se',
                'latitude_sd_' + se: 'lambda_se',
                'size_sd_' + se: 'T_one_over_vartheta_se'}, axis = 1)

            print(result_jhwi)
            result_jhwi.insert(12, 'sigma_est_se', np.nan)
            result_jhwi['sigma_est_se'].iloc[0] = zeta['zeta'].iloc[0]/2
            result_jhwi['sigma_est_se'].iloc[1] = zeta['zeta_sd_'+se].iloc[0]/2
            result_jhwi = result_jhwi.fillna(0)

            result_jhwi.to_csv(loc + 'report_table_' + se + 'se.csv',
                               index=False)




class EstimateAncient(EstimateBase):
    """ Class for estimation procedures on ancient dataset.

    Initializes the data used to be used:

    * loads processed datasets.
    * sets coordinates in degrees.
    * sets known and unknown coordinates datasets as separate attributes.
    * saves the number of known and unknown cities
    * saves the gradient of the objective function as attribute, to avoid
        calling autograd multiple times.
    * saves the jacobian of errors
    * saves the dividing indices to go from variable array to individual
        components.
    * saves other data to speed up self.tile_nodiag and self.get_errors

    Args:
        build_type (str): One of "directional" or "non-directional".
        lat (tuple): Contains assumed lower and upper latitude bounds.
        lng (tuple): Contains assumed lower and upper longitude bounds.
        rand_lost_cities (int): Optional. The number of random cities to lose.
        lng_estimated (np.ndarray): The estimated longitudes of lost cities
            in original exercise.
        lat_estimated (np.ndarray): The estimated latitudes of lost cities
            in original exercise.
    """

    def __init__(self,
                 build_type,
                 lat = (36, 42),
                 lng = (27, 45),
                 rand_lost_cities = None,
                 lng_estimated = None,
                 lat_estimated = None,
                 cities_to_drop = [],
                 cities_to_known = [],
                 cities_to_unknown = [],
                 omega = None):
        EstimateBase.__init__(self, build_type, omega = omega)
        self.lat = lat
        self.lng = lng
        self.cities_to_drop = cities_to_drop
        self.cities_to_known = cities_to_known
        self.cities_to_unknown = cities_to_unknown

        ## Paths
        root = config.get('paths', 'root')
        process = config.get('paths', 'process_a')

        # Load processed datasets
        self.df_id = pd.read_csv(root + process + 'id.csv')
        if build_type == 'directional':
            self.df_iticount = pd.read_csv(root + process
                                           + 'iticount_dir.csv')
            self.df_coordinates = pd.read_csv(root + process
                                              + 'coordinates_dir.csv')
            self.df_constr_dyn = pd.read_csv(root + process
                                             + 'constraints_dir_dyn.csv')
            self.df_constr_stat = pd.read_csv(root + process
                                              + 'constraints_dir_stat.csv')
        elif build_type == 'non_directional':
            self.df_iticount = pd.read_csv(root + process
                                           + 'iticount_nondir.csv')
            self.df_coordinates = pd.read_csv(root + process
                                              + 'coordinates_nondir.csv')
            self.df_constr_dyn = pd.read_csv(root + process
                                             + 'constraints_nondir_dyn.csv')
            self.df_constr_stat = pd.read_csv(root + process
                                              + 'constraints_nondir_stat.csv')
        else:
            raise ValueError("Initialize class with 'directional' or "
                             + "'non_directional'")


        # Drop the cities that don't participate
        for city in self.cities_to_drop:
            self.df_coordinates = self.df_coordinates.loc[
                    self.df_coordinates['id'] != city
                    ]
            self.df_iticount = self.df_iticount.loc[
                    (self.df_iticount['id_i'] != city)
                    & (self.df_iticount['id_j'] != city)
                    , ['cert_i', 'cert_j', 'id_i', 'id_j', 'N_ij']]
            # Recompute trade shares
            for status in ['i', 'j']:
                N = (self.df_iticount.groupby('id_'+status)
                                     .sum()['N_ij']
                                     .rename('N_'+status)
                    )

                # Add this info
                self.df_iticount = self.df_iticount.join(N, on='id_'+status)

            # Add s_ij
            self.df_iticount['s_ij'] = (
                    self.df_iticount['N_ij'] / self.df_iticount['N_j']
                    )


        # Save number of cities
        self.num_cities = len(self.df_coordinates)

        # Set coordinates in degrees (this should be done in process)
        for v in ['long_x', 'lat_y']:
            self.df_coordinates[v] = np.rad2deg(self.df_coordinates[v].values,
                                                dtype='float64')

        # Set additional known cities as known
        for c in cities_to_known:
            self.df_coordinates.loc[self.df_coordinates['id'] == c, 'cert'] = 2
            self.df_iticount.loc[self.df_iticount['id_i'] == c, 'cert_i'] = 2
            self.df_iticount.loc[self.df_iticount['id_j'] == c, 'cert_j'] = 2

        for c in cities_to_unknown:
            self.df_coordinates.loc[self.df_coordinates['id'] == c, 'cert'] = 3
            self.df_iticount.loc[self.df_iticount['id_i'] == c, 'cert_i'] = 3
            self.df_iticount.loc[self.df_iticount['id_j'] == c, 'cert_j'] = 3

        # Reindex datasets
        if rand_lost_cities is not None:
            lost_i = self.df_coordinates['cert'] == 3
            self.df_coordinates.loc[lost_i, 'long_x'] = lng_estimated
            self.df_coordinates.loc[lost_i, 'lat_y'] = lat_estimated
            certs = load.random_certs(self.num_cities, rand_lost_cities)
        else:
            certs = self.df_coordinates['cert']

        load = LoaderNew()

        self.df_coordinates = load.reindex_coordinates(certs,
                                                       self.df_coordinates)
        self.df_iticount = load.reindex_iticount(self.df_iticount,
                                                 self.df_coordinates)
        self.df_constr_dyn = load.reindex_constr(self.df_constr_dyn,
                                                 self.df_coordinates)
        self.df_constr_stat = load.reindex_constr(self.df_constr_stat,
                                                  self.df_coordinates)
        # Save Kanes' position in cities
        try:
            self.id_normalized = (
                    self.df_coordinates.loc[self.df_coordinates['id'] == 'ka01',
                                            'id_shuff'].values[0]
                    )
        except IndexError:
            self.id_normalized = (
                    self.df_coordinates.loc[self.df_coordinates['id'] == 'ka02',
                                            'id_shuff'].values[0]
                    )
        #else:
        #    self.id_normalized = int(
        #            self.df_coordinates.loc[self.df_coordinates['id'] == 'ka01',
        #                                    'id_jhwi'].values[0]
        #            ) - 1

        # Known cities
        self.df_known = self.df_coordinates.loc[
            self.df_coordinates['cert'] < 3,
            ['id', 'long_x', 'lat_y']
        ]
        # Lost cities
        self.df_unknown = self.df_coordinates.loc[
                self.df_coordinates['cert'] == 3,
            ['id', 'long_x', 'lat_y']
        ]

        self.num_cities_known = len(self.df_known)
        self.num_cities_unknown = len(self.df_unknown)

        # Automatic differentiation of objective and errors

        ## Objective
        ### Using all vars
        def objective_full_vars(varlist):
            """ This is the formulation for autograd. """
            return self.sqerr_sum(varlist, full_vars=True)
        self.grad_full_vars = grad(objective_full_vars)

        ## Errors
        ### Using all vars
        def error_autograd_full_vars(v):
            return self.get_errors(v, full_vars=True)
        self.jac_errors_full_vars = jacobian(error_autograd_full_vars)

        # Save indices to unpack argument of objective and gradient
        # There is a set of indices if we are using the full set of coordinates
        # as arguments, or if we use only the only the unknown cities
        # coordinates as arguments.
        self.div_indices = {
            True: {'long_unknown_s': 2 + self.num_cities_known,
                   'long_s': 2,
                   'long_e': 2 + self.num_cities,
                   'lat_unknown_s': 2 + self.num_cities + self.num_cities_known,
                   'lat_s': 2 + self.num_cities,
                   'lat_e': 2 + 2*self.num_cities,
                   'a_s': 2 + 2*self.num_cities,
                   'a_e': 2 + 3*self.num_cities},
            False: {'long_s': 1,
                    'long_e': 1 + self.num_cities_unknown,
                    'lat_s': 1 + self.num_cities_unknown,
                    'lat_e': 1 + 2*self.num_cities_unknown,
                    'a_s': 1 + 2*self.num_cities_unknown,
                    'a_e': 1 + 2*self.num_cities_unknown + self.num_cities}
        }

        # Save array index that views array of size len(self.df_coordinates)
        # and selects off-diagonal elements. See self.tile_nodiag.
        i = np.repeat(np.arange(1, self.num_cities), self.num_cities)
        self.index_nodiag = i + np.arange(self.num_cities*(self.num_cities - 1))

        # Save trade shares and trade counts (to speed up objective)
        self.shares = self.df_iticount['s_ij'].values
        self.counts = self.df_iticount['N_ij'].values


    def replace_id_coord(self, constr, drop_wahsusana=False, no_constr=False):
        """ Replaces the city id with its coordinates in the constraints data.

        Args:
            constr (DataFrame): Specifies upper and lower bounds for
                coordinates of cities.
            no_constr (bool): Whether or not to ignore the constraints.
                Default is False.

        Returns:
            The constraints data with substituted coordinates.
        """
        constr = constr.copy()
        v = ['lb_lambda', 'ub_lambda', 'lb_varphi', 'ub_varphi']

        if no_constr:
            constr[v] = np.nan

        if drop_wahsusana:
            constr[v] = constr[v].replace('wa01', np.nan)

        (ids, lats, lngs) = (self.df_coordinates['id'].values,
                              self.df_coordinates['lat_y'].values,
                              self.df_coordinates['long_x'].values
                            )

        tracker = {'lb_lambda': self.lat[0], 'ub_lambda': self.lat[1]}
        for var in tracker.keys():
            # An empty column is transformed to float64, thus raising error
            try:
                constr[var] = (constr[var].replace(np.nan, tracker[var])
                                          .replace(ids, lats)
                              )
            except TypeError:
                constr[var] = constr[var].replace(np.nan, tracker[var])


        tracker = {'lb_varphi': self.lng[0], 'ub_varphi': self.lng[1]}
        for var in tracker.keys():
            try:
                constr[var] = (constr[var].replace(np.nan, tracker[var])
                                          .replace(ids, lngs)
                              )
            except TypeError:
                constr[var] = constr[var].replace(np.nan, tracker[var])

        return constr


    def get_bounds(self, constr, full_vars=False, set_elasticity=None):
        """ Fetch the upper and lower bounds for all entries in `varlist`.

        Args:
            constr (DataFrame): Specifies upper and lower bounds for
                coordinates of cities.
            set_elasticity (float): An imposed distance elasticity of trade.
                Optional.

        Returns:
            tuple: (lb, ub), where lb and ub are of type `list` for the bounds.
        """
        # Build specs: Ursu does not participate if directional, Kanes is in 
        # second position in coordinates dataframe if directional, third
        # otherwise.
        #if self.build_type == 'directional':
        #    constr = constr[constr['id'] != 'ur01']

        if full_vars:
            num_vars = 2 + 3*self.num_cities
        else:
            num_vars = 1 + 2*self.num_cities_unknown + self.num_cities

        lb = num_vars * [-1.0e20]
        ub = num_vars * [1.0e20]

        print(len(lb), len(ub))

        if set_elasticity != None:
            lb[0] = set_elasticity
            ub[0] = set_elasticity
        else:
            # zeta should be larger than zero
            lb[0] = 0.0

        print(len(lb), len(ub))
        i = self.div_indices[full_vars]
        dit = {'long':('varphi', 'long_x'), 'lat': ('lambda', 'lat_y')}

        if full_vars:
            lb[1] = 0
            ub[1] = 0
            for c in dit.keys():
                # Known locations are given
                lb[i[c+'_s']: i[c+'_unknown_s']] = self.df_known[dit[c][1]].tolist()
                ub[i[c+'_s']: i[c+'_unknown_s']] = self.df_known[dit[c][1]].tolist()
                # Unknown location constraints
                lb[i[c+'_unknown_s']: i[c+'_e']] = constr['lb_'+dit[c][0]].tolist()
                ub[i[c+'_unknown_s']: i[c+'_e']] = constr['ub_'+dit[c][0]].tolist()
        else:
            for c in dit.keys():
                # Unknown location constraints
                print(lb[i[c+'_s']: i[c+'_e']])
                print(constr['lb_'+dit[c][0]].tolist())
                assert len(lb[i[c+'_s']: i[c+'_e']]) == len(constr['lb_'+dit[c][0]].tolist())
                assert len(ub[i[c+'_s']: i[c+'_e']]) == len(constr['ub_'+dit[c][0]].tolist())
                lb[i[c+'_s']: i[c+'_e']] = constr['lb_'+dit[c][0]].tolist()
                ub[i[c+'_s']: i[c+'_e']] = constr['ub_'+dit[c][0]].tolist()
        lb[i['a_s']:] = self.num_cities * [0.0]

        # Normalization
        lb[i['a_s'] + self.id_normalized] = 100.0
        ub[i['a_s'] + self.id_normalized] = 100.0

        #print(self.id_normalized)

        return (lb, ub)


    def initial_cond_uniform(self, len_sim=None):
        """ Gets uniform draws of initial condition(s) for `IPOPT`.

        Returns:
            np.ndarray: An array with `len_sim` perturbed initial
                conditions.
        """
        # Specify lower and upper bounds for every variable (full_vars = False)
        lower_bound = ( [0]
                        + self.num_cities_unknown * [self.lng[0]]
                        + self.num_cities_unknown * [self.lat[0]]
                        + self.num_cities * [0]
                      )
        upper_bound =  ( [15]
                        + self.num_cities_unknown * [self.lng[1]]
                        + self.num_cities_unknown * [self.lat[1]]
                        + self.num_cities * [200]
                      )
        n_params = 1 + 2*self.num_cities_unknown + self.num_cities

        return np.random.uniform(lower_bound,
                                 upper_bound,
                                 size=(len_sim, n_params))


    def solve(self,
              x0,
              constraint_type = 'static',
              max_iter = 25000,
              full_vars = False,
              solver = 'ma57',
              set_elasticity = None,
              no_constr = False,
              scale = None):
        """ Solve the sum of squared distances minimization problem with IPOPT.

        Args:
            x0 (list): The initial value.
            max_iter (int): Maximum iterations before IPOPT stops.
            constraint_type (str): One of 'static' or 'dynamic'.
            solver (str): Linear solver. 'ma57' is the default. If not
                available, use 'mumps'.

        Returns:
            A one-row dataframe with optimization information.
        """
        print('scale:')
        print(scale)
        # Cast x0 as np.float64. See numpy bug in `self.haversine_approx`.
        x0 = np.float64(x0)

        # Set bounds
        if constraint_type == 'static':
            constr = self.replace_id_coord(self.df_constr_stat,
                                           no_constr=no_constr)
        elif constraint_type == 'dynamic':
            constr = self.replace_id_coord(self.df_constr_dyn,
                                           no_constr=no_constr)
        else:
            raise ValueError("Please specify the constraint type to be "
                             + "'static' or 'dynamic'")
        bounds = self.get_bounds(constr, full_vars, set_elasticity)

        print('Low bound:')
        print(bounds[0])
        print(len(bounds[0]))
        print('High bound:')
        print(bounds[1])
        print(len(bounds[1]))
        #print('max_iter:' + str(max_iter))
        assert len(bounds[0]) == len(x0)
        assert len(bounds[1]) == len(x0)

        nlp = ipopt.problem( n=len(x0),
                             m=0,
                             problem_obj=OptimizerAncient(
                                 build_type = self.build_type,
                                 omega = self.omega,
                                 cities_to_known = self.cities_to_known,
                                 cities_to_unknown = self.cities_to_unknown,
                                 cities_to_drop = self.cities_to_drop,
                                 full_vars = full_vars,
                                 scale = scale),
                             lb=bounds[0],
                             ub=bounds[1] )


        # Add IPOPT options (some jhwi options were default)
        option_specs = { 'hessian_approximation': 'limited-memory',
                         'print_frequency_iter': 5,
                         'linear_solver': solver,
                         'limited_memory_max_history': 100,
                         'limited_memory_max_skipping': 1,
                         'mu_strategy': 'adaptive',
                         'tol': 1e-8,
                         'acceptable_tol': 1e-7,
                         'acceptable_iter': 100,
                         'max_iter': max_iter}
        for option in option_specs.keys():
            nlp.addOption(option, option_specs[option])

        (x, info) = nlp.solve(x0)

        # Set up variable names
        alphas = ['{0}_a'.format(i) for i in self.df_coordinates['id'].tolist()]
        if full_vars:
            longs = ['{0}_lng'.format(i) for i in self.df_coordinates['id'].tolist()]
            lats = ['{0}_lat'.format(i) for i in self.df_coordinates['id'].tolist()]
            headers = ['zeta', 'useless'] + longs + lats + alphas
        else:
            longs = ['{0}_lng'.format(i) for i in self.df_unknown['id'].tolist()]
            lats = ['{0}_lat'.format(i) for i in self.df_unknown['id'].tolist()]
            headers = ['zeta'] + longs + lats + alphas

        df = pd.DataFrame(data = [x],
                          columns = headers)
        df['obj_val'] = info['obj_val']
        df['status'] = info['status']
        df['status_msg'] = info['status_msg']
        df['status_msg'] = df['status_msg'].str.replace(';', '')

        #Add initial condition
        for ival in range(len(x0)):
            df['x0_'+str(ival)] = x0[ival]

        return df


    def gen_data(self,
                 x0,
                 rank = None,
                 max_iter = 25000,
                 full_vars = False,
                 set_elasticity=None,
                 no_constr=False,
                 solver = 'ma57',
                 scale = None):
        """ Run `self.solve` for many initial values.

        This function is the one called when running estimation in parallel.

        Args:
            x0 (np.ndarray): The array of initial conditions. Each row is an
                initial condition.
            rank (int): Process number in parallelized computing.

        Returns:
            DataFrame: simulation dataframe sorted by objective value

        Warning:
            Make sure `full_vars` is consistent with `x0`.
        """
        print('Unknown Cities:')
        print(np.unique(self.df_unknown))

        print('Known Cities:')
        print(np.unique(self.df_known))

        data = self.solve( x0[0, :],
                           max_iter=max_iter,
                           full_vars=full_vars,
                           set_elasticity=set_elasticity,
                           no_constr=no_constr,
                           solver = solver,
                           scale = scale)
        len_sim = x0.shape[0]
        for i in range(1, len_sim):
            data = data.append( self.solve(x0[i, :],
                                           max_iter = max_iter,
                                           full_vars = full_vars,
                                           set_elasticity = set_elasticity,
                                           no_constr = no_constr,
                                           solver = solver,
                                           scale = scale) )

        if rank != None:
            data['process'] = rank

        # Sort
        return data.sort_values('obj_val')


    def check_bounds_hold(self, varlist, bounds):
        """ Check whether coordinates in `arg` satisfy bounds.

        Args:
            bounds (tuple): the output of `self.get_bounds`.
            arg (np.ndarray): the argument in short format (as appears in
                IPOPT output).

        Returns:
            bool: True if bounds hold, False otherwise.
        """
        # Unpack coordinates
        i = self.div_indices[False]
        lng = varlist[i['long_s']: i['long_e']]
        lat = varlist[i['lat_s']: i['lat_e']]

        # Get corresponding bounds
        lng_lb = bounds[0][i['long_s']: i['long_e']]
        lat_lb = bounds[0][i['lat_s']: i['lat_e']]
        lng_ub = bounds[1][i['long_s']: i['long_e']]
        lat_ub = bounds[1][i['lat_s']: i['lat_e']]

        bools_lat_lb = np.greater_equal(lat, lat_lb)
        bools_lat_ub = np.greater_equal(lat_ub, lat)
        bools_lng_lb = np.greater_equal(lng, lng_lb)
        bools_lng_ub = np.greater_equal(lng_ub, lng)

        return np.prod( np.concatenate((bools_lat_lb,
                                        bools_lat_ub,
                                        bools_lng_lb,
                                        bools_lng_ub))
                      )




class EstimateToblerWeinburg(EstimateAncient):
    """ Class for estimation procedures à la Tobler and Weinburg (1971).
    """

    def __init__(self,
                 build_type = 'non_directional',
                 lat = (36, 42),
                 lng = (27, 45),
                 rand_lost_cities = None,
                 lng_estimated = None,
                 lat_estimated = None,
                 cities_to_drop = [],
                 cities_to_known = [],
                 cities_to_unknown = [],
                 omega = None):
        EstimateAncient.__init__(self,
                                 build_type,
                                 lat,
                                 lng,
                                 rand_lost_cities = rand_lost_cities,
                                 lng_estimated = lng_estimated,
                                 lat_estimated = lat_estimated,
                                 cities_to_drop = cities_to_drop,
                                 cities_to_known = cities_to_known,
                                 cities_to_unknown = cities_to_unknown,
                                 omega = omega)

        # Add inverted gravity term
        self.df_iticount['inv_grav'] = (
                self.df_iticount['N_i'] * self.df_iticount['N_j']
                / self.df_iticount['N_ij']
                )

        # Save this to speed up objective
        self.inv_grav = self.df_iticount['inv_grav'].values
        self.inv_grav = self.inv_grav[self.df_iticount['N_ij'] != 0]


        # Automatic differentiation of objective and errors

        ## Objective
        def objective(varlist):
            """ This is the formulation for autograd. """
            return self.sqerr_sum(varlist)
        self.grad = grad(objective)

        ## Errors
        ### Using all vars
        def error_autograd(v):
            return self.get_errors(v)
        self.jac_errors = jacobian(error_autograd)


    def get_bounds(self, constr):
        """ Fetch the upper and lower bounds for all entries in `varlist`.

        Args:
            constr (DataFrame): Specifies upper and lower bounds for
                coordinates of cities.
            set_elasticity (float): An imposed distance elasticity of trade.
                Optional.

        Returns:
            tuple: (lb, ub), where lb and ub are of type `list` for the bounds.
        """
        num_vars = 1 + 2*self.num_cities_unknown

        lb = num_vars * [-1.0e20]
        ub = num_vars * [1.0e20]

        print(len(lb), len(ub))

        i = self.div_indices[False]
        dit = {'long':('varphi', 'long_x'), 'lat': ('lambda', 'lat_y')}

        for c in dit.keys():
            # Unknown location constraints
            print(lb[i[c+'_s']: i[c+'_e']])
            print(constr['lb_'+dit[c][0]].tolist())
            assert len(lb[i[c+'_s']: i[c+'_e']]) == len(constr['lb_'+dit[c][0]].tolist())
            assert len(ub[i[c+'_s']: i[c+'_e']]) == len(constr['ub_'+dit[c][0]].tolist())
            lb[i[c+'_s']: i[c+'_e']] = constr['lb_'+dit[c][0]].tolist()
            ub[i[c+'_s']: i[c+'_e']] = constr['ub_'+dit[c][0]].tolist()

        return (lb, ub)


    def initial_cond_uniform(self, len_sim=None):
        """ Gets uniform draws of initial condition(s) for `IPOPT`.

        Returns:
            np.ndarray: An array with `len_sim` perturbed initial
                conditions.
        """
        # Specify lower and upper bounds for every variable (full_vars = False)
        lower_bound = ( [0]
                        + self.num_cities_unknown * [self.lng[0]]
                        + self.num_cities_unknown * [self.lat[0]]
                      )
        upper_bound =  ( [15]
                        + self.num_cities_unknown * [self.lng[1]]
                        + self.num_cities_unknown * [self.lat[1]]
                      )
        n_params = 1 + 2*self.num_cities_unknown

        return np.random.uniform(lower_bound,
                                 upper_bound,
                                 size=(len_sim, n_params))


    def initial_cond(self,
                     suggestion,
                     len_sim=None,
                     perturb=None,
                     perturb_type='rigid'):
        """ Gets initial condition(s) for `IPOPT`.

        Args:
            len_sim (int): Specifies the number of initial conditions to draw.
            perturb (float): A percentage deviation from the default initial
                value given in `self.df_coordinates`.
            perturb_type (str): Type of perturbation on the default initial
                value. If `'rigid'` then the whole initial value vector is
                multiplied by a scalar. If `'flexible'` then each element of
                the initial value vector is multiplied by a different scalar.
                Default is `'rigid'`.
            suggestion (np.ndarray): A custom initial value. It must be
                coherent with `full_vars`.

        Returns:
            np.ndarray: The default initial condition if perturb is not
                specified, and an array with `len_sim` perturbed initial
                conditions.
        """
        x0 = suggestion

        # Perturb it
        if perturb != None:
            x0 = np.tile(x0, (len_sim, 1))

            if perturb_type == 'rigid':
                p = np.random.uniform(1-perturb, 1+perturb, size=(len_sim, 1))
            elif perturb_type == 'flexible':
                p = np.random.uniform(1-perturb,
                                      1+perturb,
                                      size=(len_sim, x0.shape[1]))
            #print(p)
            x0 = x0*p

        return x0


    def get_errors(self, varlist, full_vars):
        """ Get the model and data difference in the gravity equation.

        Args:
            varlist (np.ndarray): it is composed of
                `[alpha_inv, lng_guess, lat_guess]`.

        Returns:
            np.ndarray: the Tobler-Weinburg errors.
        """
        # Unpack arguments
        alpha_inv = varlist[0]

        i = self.div_indices[full_vars]
        lng_guess = varlist[i['long_s']: i['long_e']]
        lat_guess = varlist[i['lat_s']: i['lat_e']]

        assert len(lat_guess) == len(lng_guess)

        distances = self.fetch_dist(lat_guess, lng_guess, full_vars)
        distances = distances[self.df_iticount['N_ij'] != 0]

        return self.inv_grav - alpha_inv * (distances ** 2)


    def solve(self,
              x0,
              constraint_type = 'static',
              max_iter = 25000,
              full_vars = False,
              solver = 'ma57',
              set_elasticity = None,
              no_constr = False,
              scale = None):
        """ Solve the sum of squared distances minimization problem with IPOPT.

        Args:
            x0 (list): The initial value.
            max_iter (int): Maximum iterations before IPOPT stops.
            constraint_type (str): One of 'static' or 'dynamic'.
            solver (str): Linear solver. 'ma57' is the default. If not
                available, use 'mumps'.

        Returns:
            A one-row dataframe with optimization information.
        """
        # Cast x0 as np.float64. See numpy bug in `self.haversine_approx`.
        x0 = np.float64(x0)

        # Set bounds
        if constraint_type == 'static':
            constr = self.replace_id_coord(self.df_constr_stat,
                                           no_constr=no_constr)
        elif constraint_type == 'dynamic':
            constr = self.replace_id_coord(self.df_constr_dyn,
                                           no_constr=no_constr)
        else:
            raise ValueError("Please specify the constraint type to be "
                             + "'static' or 'dynamic'")

        bounds = self.get_bounds(constr)

        print('Low bound:')
        print(bounds[0])
        print(len(bounds[0]))
        print('High bound:')
        print(bounds[1])
        print(len(bounds[1]))
        #print('max_iter:' + str(max_iter))
        assert len(bounds[0]) == len(x0)
        assert len(bounds[1]) == len(x0)

        nlp = ipopt.problem( n=len(x0),
                             m=0,
                             problem_obj=OptimizerToblerWeinburg(
                                 omega = self.omega,
                                 cities_to_known = self.cities_to_known,
                                 cities_to_unknown = self.cities_to_unknown,
                                 cities_to_drop = self.cities_to_drop),
                             lb=bounds[0],
                             ub=bounds[1] )


        # Add IPOPT options (some jhwi options were default)
        option_specs = { 'hessian_approximation': 'limited-memory',
                         'print_frequency_iter': 5,
                         'linear_solver': solver,
                         'limited_memory_max_history': 100,
                         'limited_memory_max_skipping': 1,
                         'mu_strategy': 'adaptive',
                         'tol': 1e-8,
                         'acceptable_tol': 1e-7,
                         'acceptable_iter': 100,
                         'max_iter': max_iter}
        for option in option_specs.keys():
            nlp.addOption(option, option_specs[option])

        (x, info) = nlp.solve(x0)

        # Set up variable names
        longs = ['{0}_lng'.format(i) for i in self.df_unknown['id'].tolist()]
        lats = ['{0}_lat'.format(i) for i in self.df_unknown['id'].tolist()]
        headers = ['alpha_inv'] + longs + lats

        df = pd.DataFrame(data = [x], columns = headers)
        df['obj_val'] = info['obj_val']
        df['status'] = info['status']
        df['status_msg'] = info['status_msg']
        df['status_msg'] = df['status_msg'].str.replace(';', '')

        #Add initial condition
        for ival in range(len(x0)):
            df['x0_'+str(ival)] = x0[ival]

        return df


    def get_jacobian(self,
                     varlist,
                     var_type='white',
                     zeta_fixed=False):
        """ Compute the variance-covariance matrix of the estimators.

        It can be computed according to the White formula, or with
        homoskedasticity.

        Args:
            var_type (str): One of 'white' or 'homo', or 'gmm'.

        Returns:
            np.ndarray: The variance-covariance matrix of the estimators.
        """
        # Evaluate errors jacobian at estimated parameter.
        jac = self.jac_errors(varlist)

        # If zeta (i.e. alpha_inv) is fixed, remove it.
        if zeta_fixed:
            jac = np.delete(jac, 0, axis=1)

        return jac


    def get_variance(self,
                     varlist,
                     var_type='white',
                     zeta_fixed=False):
        """ Compute the variance-covariance matrix of the estimators.

        It can be computed according to the White formula, or with
        homoskedasticity.

        Args:
            var_type (str): One of 'white' or 'homo', or 'gmm'.

        Returns:
            np.ndarray: The variance-covariance matrix of the estimators.
        """
        # Evaluate errors jacobian at estimated parameter.
        jac = self.jac_errors(varlist)

        # If zeta is fixed, remove it.
        if zeta_fixed:
            jac = np.delete(jac, 0, axis=1)

        errors = self.get_errors(varlist, full_vars = False)

        # Build variance-covariance matrix, according to var_type
        scale = 1.0
        bread = np.linalg.inv(scale * np.dot( np.transpose(jac), jac ))
        if var_type == 'gmm':
            #middle = np.matmul(scale*np.expand_dims(errors, 1),
            #                   scale*np.expand_dims(errors, 0))
            middle = np.outer(scale*errors, scale*errors)
            ham = np.linalg.multi_dot((np.transpose(jac), middle, jac))
            return np.linalg.multi_dot((bread, ham, bread))

        elif var_type == 'white':
            middle = np.diag((scale*errors)**2)
            ham = np.linalg.multi_dot((np.transpose(jac), middle, jac))
            ## Make column vector to multiply row-wise
            #errors = np.expand_dims(errors, 1)
            #ham = np.dot(np.transpose(jac * errors), jac * errors)
            return np.linalg.multi_dot((bread, ham, bread))

        elif var_type == 'homo':
            return scale * (np.sum(errors**2) / len(errors)) * bread

        else:
            raise ValueError("Please specify the variance type to be one of "
                    + "'white' or 'homo'")




class EstimateAncientMLE(EstimateAncient):

    def __init__(self,
                 build_type,
                 lat = (36, 42),
                 lng = (27, 45),
                 lng_estimated = None,
                 lat_estimated = None):
        EstimateAncient.__init__(self,
                                 build_type,
                                 lat,
                                 lng,
                                 lng_estimated,
                                 lat_estimated)

        ## Objective (assume full_vars is False)
        def objective(varlist):
            """ This is the formulation for autograd. """
            return self.mle_objective(varlist)
        self.grad = grad(objective)

        ## Errors
        def increments_autograd(v):
            return self.log_L_increments(v)
        self.jac_increments = jacobian(increments_autograd)
        self.hess_increments2 = hessian(increments_autograd)

        def hess(v):
            return self.jac_increments(v)
        self.hess_increments = jacobian(hess)


        ## Using all vars
        def increments_autograd_full_vars(v):
            return self.log_L_increments(v, full_vars=True)
        self.jac_increments_full_vars = jacobian(increments_autograd_full_vars)

        def hess_full_vars(v):
            return self.jac_increments_full_vars(v)
        self.hess_increments_full_vars = jacobian(hess_full_vars)


    def log_L_increments(self, varlist, full_vars=False):
        """ Get the log likelihood increments.

        Args:
            varlist (np.ndarray): it is composed of
                `[zeta, lng_guess, lat_guess, alpha]`.

        Returns:
            np.ndarray: the one-dimensional array of log likelihood
            increments.
        """
        # Unpack arguments
        zeta = varlist[0]

        i = self.div_indices[full_vars]
        lng_guess = varlist[i['long_s']: i['long_e']]
        lat_guess = varlist[i['lat_s']: i['lat_e']]
        alpha = varlist[i['a_s']:]

        #assert len(lat_guess) == len(lng_guess)

        s_ij_model = self.s_ij_model(zeta,
                                     alpha,
                                     self.fetch_dist(lat_guess,
                                                     lng_guess,
                                                     full_vars)
                                    )

        #print('Max s_ij_model:')
        #print(np.max(s_ij_model))
        #print('Min s_ij_model:')
        #print(np.min(s_ij_model))

        # Scale shares
        #s_ij_model = 1.0e+50 * s_ij_model

        return self.counts * np.log(s_ij_model)


    def mle_objective(self, varlist, full_vars=False):
        """ Get the log objective specification.

        Gets the sum of data shares times log(model shares).

        Returns:
            float: The negative of the likelihood function evaluated at
                varlist.
        """
        increments = self.log_L_increments(varlist, full_vars = full_vars)

        return - np.sum(increments)


    def solve(self,
              x0,
              constraint_type = 'static',
              max_iter = 25000,
              full_vars = False,
              solver='ma57',
              set_elasticity=None,
              no_constr=False):
        """ Solve the sum of squared distances minimization problem with IPOPT.

        Args:
            x0 (list): The initial value.
            max_iter (int): Maximum iterations before IPOPT stops.
            constraint_type (str): One of 'static' or 'dynamic'.
            solver (str): Linear solver. 'ma57' is the default. If not
                available, use 'mumps'.

        Returns:
            A one-row dataframe with optimization information.
        """
        # Cast x0 as np.float64. See numpy bug in `self.haversine_approx`.
        x0 = np.float64(x0)

        # Set bounds
        if constraint_type == 'static':
            constr = self.replace_id_coord(self.df_constr_stat,
                                           no_constr=no_constr)
        elif constraint_type == 'dynamic':
            constr = self.replace_id_coord(self.df_constr_dyn,
                                           no_constr=no_constr)
        else:
            raise ValueError("Please specify the constraint type to be "
                             + "'static' or 'dynamic'")
        bounds = self.get_bounds(constr, full_vars, set_elasticity)

        assert len(bounds[0]) == len(x0)
        assert len(bounds[1]) == len(x0)
        #print(x0)
        #print(len(x0))
        #print('Low bound:')
        #print(bounds[0])
        #print('High bound:')
        #print(bounds[1])

        nlp = ipopt.problem( n=len(x0),
                             m=0,
                             problem_obj=OptimizerAncientMLE(build_type=self.build_type,
                                                                   full_vars=full_vars),
                             lb=bounds[0],
                             ub=bounds[1] )

        # Add IPOPT options (some jhwi options were default)
        option_specs = { 'hessian_approximation': 'limited-memory',
                         'linear_solver': solver,
                         'limited_memory_max_history': 100,
                         'limited_memory_max_skipping': 1,
                         'mu_strategy': 'adaptive',
                         'tol': 1e-10,
                         'acceptable_tol': 1e-10,
                         'acceptable_iter': 100,
                         'max_iter': max_iter}
        for option in option_specs.keys():
            nlp.addOption(option, option_specs[option])

        (x, info) = nlp.solve(x0)

        # Set up variable names
        alphas = ['{0}_a'.format(i) for i in self.df_coordinates['id'].tolist()]
        if full_vars:
            longs = ['{0}_lng'.format(i) for i in self.df_coordinates['id'].tolist()]
            lats = ['{0}_lat'.format(i) for i in self.df_coordinates['id'].tolist()]
            headers = ['zeta', 'useless'] + longs + lats + alphas
        else:
            longs = ['{0}_lng'.format(i) for i in self.df_unknown['id'].tolist()]
            lats = ['{0}_lat'.format(i) for i in self.df_unknown['id'].tolist()]
            headers = ['zeta'] + longs + lats + alphas

        df = pd.DataFrame(data = [x],
                          columns = headers)
        df['obj_val'] = info['obj_val']
        df['status'] = info['status']
        df['status_msg'] = info['status_msg']
        df['status_msg'] = df['status_msg'].str.replace(';', '')

        #Add initial condition
        for ival in range(len(x0)):
            df['x0_'+str(ival)] = x0[ival]

        return df


    def get_variance(self,
                     varlist,
                     var_type='white',
                     zeta_fixed=False,
                     full_vars=False,
                     numerical=False):
        """ Compute the variance-covariance matrix of the estimators.

        This overwrites the analogous function in EstimateBase.

        Args:
            var_type (str): One of 'white' or 'homo'. This argument is useless
                and is provided only for compatibility.
            numerical (bool): If True, use jacobian that is numerically
                approximated. Otherwise, go for the exact jacobian. I assume
                that full_vars == False if True.

        Returns:
            np.ndarray: The variance-covariance matrix of the estimators.
        """
        if full_vars:
            if numerical:
                print("--------------------------------------------------")
                print("IT IS NUMERICAL")
                print("--------------------------------------------------")
                jac = self.numerical_jac(varlist, h=1.0e-05, full_vars=True)
            else:
                jac = self.jac_increments_full_vars(varlist)
            i = self.div_indices[True]
            jac = pd.DataFrame(jac)
            jac = jac.drop(columns = ([1]
                           + range(i['long_s'], i['long_unknown_s'])
                           + range(i['lat_s'], i['lat_unknown_s'])))
            jac = jac.values
        else:
            if numerical:
                jac = self.numerical_jac(varlist, h=1.0e-05)
            else:
                jac = self.jac_increments(varlist)

        # Remove fixed a
        index_norm = self.div_indices[False]['a_s'] + self.id_normalized
        jac = np.delete(jac, index_norm, axis=1)

        # If zeta is fixed, remove it.
        if zeta_fixed:
            jac = np.delete(jac, 0, axis=1)

        #print("Jacobian shape:")
        #print(jac.shape)
        jac_A = np.expand_dims(jac, 2)
        jac_B = np.expand_dims(jac, 1)

        #print("Before getting dat outer product")
        #print(jac_A.shape)
        #print(jac_B.shape)
        #print(len(jac))

        outer_prods = np.matmul(jac_A, jac_B)
        A0 = np.sum(outer_prods, axis = 0)

        return np.linalg.inv(A0)


    def get_variance2(self,
                      varlist,
                      var_type='white',
                      zeta_fixed=False,
                      full_vars=False):
        """ Compute the variance-covariance matrix of the estimators.

        Another way to compute the asymptotic variance. Consistency test for
        self.get_variance

        Args:
            var_type (str): One of 'white' or 'homo'. This argument is useless
                and is provided only for compatibility.

        Returns:
            np.ndarray: The variance-covariance matrix of the estimators.
        """
        if full_vars:
            i = self.div_indices[True]
            jac = self.jac_increments_full_vars(varlist)
            jac = pd.DataFrame(jac)
            jac = jac.drop(columns = ([1]
                           + range(i['long_s'], i['long_unknown_s'])
                           + range(i['lat_s'], i['lat_unknown_s'])))
            jac = jac.values
        else:
            # Evaluate errors jacobian at estimated parameter.
            jac = self.jac_increments(varlist)

        # Remove fixed a
        index_norm = self.div_indices[False]['a_s'] + self.id_normalized
        jac = np.delete(jac, index_norm, axis=1)

        # If zeta is fixed, remove it.
        if zeta_fixed:
            jac = np.delete(jac, 0, axis=1)

        outer_prods = np.empty((len(jac), 48, 48))
        for i in range(len(jac)):
            outer_prods[i] = np.outer(jac[i], jac[i])

        return np.linalg.inv(np.sum(outer_prods, axis = 0))


    def numerical_grad(self, varlist, h):
        """ Compute the numerical gradient for step size `h`.

        Args:
            h (float): The step size.

        Returns:
            np.ndarray: The gradient of ``self.mle_objective`` evaluated at
                ``varlist`` for step size ``h``.
        """
        # Form inputs
        H = h * np.eye(len(varlist))
        v_plus = varlist + H
        v_minus = varlist - H

        # ``self.mle_objective`` must be evaluated one arg at a time... loop.
        grads = np.empty(len(varlist))

        for i in range(len(varlist)):
            obj_plus = self.mle_objective(v_plus[i])
            obj_minus = self.mle_objective(v_minus[i])
            grads[i] = (obj_plus - obj_minus) / (2 * h)

        return grads


    def numerical_jac(self, varlist, h, full_vars = False):
        """ Compute the numerical jacobian for step size `h`.

        Args:
            h (float): The step size.

        Returns:
            np.ndarray: The jacobian of ``self.log_L_increments`` evaluated
                at ``varlist`` for step size ``h``.
        """
        # Form inputs
        H = h * np.eye(len(varlist))
        v_plus = varlist + H
        v_minus = varlist - H

        # ``self.mle_objective`` must be evaluated one arg at a time... loop.
        jac = np.empty((len(self.shares), len(varlist)))

        for col in range(len(varlist)):
            obj_plus = self.log_L_increments(v_plus[col], full_vars)
            obj_minus = self.log_L_increments(v_minus[col], full_vars)
            jac[:, col] = (obj_plus - obj_minus) / (2 * h)

        return jac




class EstimateModernProof(EstimateBase):
    """ Class for estimation procedures on modern dataset, proof of concept.

    Initializes the data used to be used:

    * loads processed datasets.
    * sets coordinates in degrees.
    * sets known and unknown coordinates datasets as separate attributes.
    * saves the number of known and unknown cities
    * saves the gradient of the objective function as attribute, to avoid
        calling autograd multiple times.
    * saves the jacobian of errors
    * saves the dividing indices to go from variable array to individual
        components.
    * saves other data to speed up self.tile_nodiag and self.get_errors

    Args:
        build_type (str): One of "directional" or "non-directional".
        lat (tuple): Contains assumed lower and upper latitude bounds.
        lng (tuple): Contains assumed lower and upper longitude bounds.
        rand_lost_cities (int): Optional. The number of random cities to lose.
        lng_estimated (np.ndarray): The estimated longitudes of lost cities
            in original exercise.
        lat_estimated (np.ndarray): The estimated latitudes of lost cities
            in original exercise.
    """

    def __init__(self,
                 rand_lost_cities,
                 lat = (35, 43),
                 lng = (25, 45)):
        EstimateBase.__init__(self, build_type='directional')

        self.lat = lat
        self.lng = lng

        ## Paths
        root = config.get('paths', 'root')
        process = config.get('paths', 'process_p')

        # Load processed datasets
        self.df_coordinates = pd.read_csv(root + process + 'modern'
                                          + '/modern_cities_all_coordinates.csv')
        self.df_iticount = pd.read_csv(root + process + 'modern'
                                       + '/estimation_data_directional.csv')

        # Save number of cities
        self.num_cities = len(self.df_coordinates)

        # Rename coordinates columns for consistency
        self.df_coordinates.rename({'name': 'city_name',
                                    'x_c': 'long_x',
                                    'y_c': 'lat_y'}, axis = 1)

        # Separate coordinates data from id-names data
        self.df_id = self.df_coordinates[['city_name', 'id']]
        self.df_coordinates = self.df_coordinates[['id', 'long_x', 'lat_y']]

        # Loose cities
        load = Loader()
        certs = load.random_certs(self.num_cities, rand_lost_cities)
        self.df_coordinates = load.reindex_coordinates(certs,
                                                       self.df_coordinates)
        self.df_iticount = load.reindex_iticount(self.df_iticount,
                                                 self.df_coordinates,
                                                 notation='jhwi')
        # Save position of city that must be normalized to 100.
        self.id_normalized = (
                self.df_coordinates.loc[self.df_coordinates['id'] == 38,
                                        'id_shuff'].values[0]
                )

        # Known cities
        self.df_known = self.df_coordinates.loc[
            self.df_coordinates['cert'] < 3,
            ['id', 'long_x', 'lat_y']
        ]
        # Lost cities
        self.df_unknown = self.df_coordinates.loc[
                self.df_coordinates['cert'] == 3,
            ['id', 'long_x', 'lat_y']
        ]

        self.num_cities_known = len(self.df_known)
        self.num_cities_unknown = len(self.df_unknown)


        # Automatic differentiation of objective and errors
        ## Errors
        ### Using all vars
        def error_autograd_full_vars(v):
            return self.get_errors(v, full_vars=True)
        self.jac_errors_full_vars = jacobian(error_autograd_full_vars)

        # Save indices to unpack argument of objective and gradient
        # There is a set of indices if we are using the full set of coordinates
        # as arguments, or if we use only the only the unknown cities
        # coordinates as arguments.
        self.div_indices = {
            True: {'long_unknown_s': 2 + self.num_cities_known,
                   'long_s': 2,
                   'long_e': 2 + self.num_cities,
                   'lat_unknown_s': 2 + self.num_cities + self.num_cities_known,
                   'lat_s': 2 + self.num_cities,
                   'lat_e': 2 + 2*self.num_cities,
                   'a_s': 2 + 2*self.num_cities,
                   'a_e': 2 + 3*self.num_cities},
            False: {'long_s': 1,
                    'long_e': 1 + self.num_cities_unknown,
                    'lat_s': 1 + self.num_cities_unknown,
                    'lat_e': 1 + 2*self.num_cities_unknown,
                    'a_s': 1 + 2*self.num_cities_unknown,
                    'a_e': 1 + 2*self.num_cities_unknown + self.num_cities}
        }

        # Save array index that views array of size len(self.df_coordinates)
        # and selects off-diagonal elements. See self.tile_nodiag.
        i = np.repeat(np.arange(1, self.num_cities), self.num_cities)
        self.index_nodiag = i + np.arange(self.num_cities*(self.num_cities - 1))

        # Save trade shares (to speed up self.get_errors)
        self.shares = self.df_iticount['s_ij'].values


    def get_bounds(self, full_vars=False):
        """ Fetch the upper and lower bounds for all entries in `varlist`.

        Returns:
            tuple: (lb, ub), where lb and ub are of type `list` for the bounds.
        """
        if full_vars:
            num_vars = 2 + 3*self.num_cities
        else:
            num_vars = 1 + 2*self.num_cities_unknown + self.num_cities

        lb = num_vars * [-1.0e20]
        ub = num_vars * [1.0e20]

        # zeta should be larger than zero
        lb[0] = 0.0

        i = self.div_indices[full_vars]
        dit = {'long': 'long_x', 'lat': 'lat_y'}
        unknown_lb = { 'long': self.num_cities_unknown * [self.lng[0]],
                       'lat': self.num_cities_unknown * [self.lat[0]] }
        unknown_ub = { 'long': self.num_cities_unknown * [self.lng[1]],
                       'lat': self.num_cities_unknown * [self.lat[1]] }
        if full_vars:
            lb[1] = 0
            ub[1] = 0
            for c in dit.keys():
                # Known locations are given
                lb[i[c+'_s']: i[c+'_unknown_s']] = self.df_known[dit[c]].tolist()
                ub[i[c+'_s']: i[c+'_unknown_s']] = self.df_known[dit[c]].tolist()
                # Unknown location constraints
                lb[i[c+'_unknown_s']: i[c+'_e']] = unknown_lb[c]
                ub[i[c+'_unknown_s']: i[c+'_e']] = unknown_ub[c]
        else:
            for c in dit.keys():
                # Unknown location constraints
                lb[i[c+'_s']: i[c+'_e']] = unknown_lb[c]
                ub[i[c+'_s']: i[c+'_e']] = unknown_ub[c]

        # alphas should be positive
        lb[i['a_s']:] = self.num_cities * [0.0]

        # Normalization
        lb[i['a_s'] + self.id_normalized] = 100.0
        ub[i['a_s'] + self.id_normalized] = 100.0

        return (lb, ub)


    def solve(self,
              x0,
              max_iter = 25000,
              full_vars = False,
              solver='ma57'):
        """ Solve the sum of squared distances minimization problem with IPOPT.

        Args:
            x0 (list): The initial value.
            max_iter (int): Maximum iterations before IPOPT stops.
            solver (str): Linear solver. 'ma57' is the default. If not
                available, use 'mumps'.

        Returns:
            A one-row dataframe with optimization information.
        """
        # Cast x0 as np.float64. See numpy bug in `self.haversine_approx`.
        x0 = np.float64(x0)

        bounds = self.get_bounds(full_vars)

        assert len(bounds[0]) == len(x0)
        #print('Low bound:')
        #print(bounds[0])
        #print('High bound:')
        #print(bounds[1])
        #print('max_iter:' + str(max_iter))

        nlp = ipopt.problem( n=len(x0),
                             m=0,
                             problem_obj=OptimizerAncient(build_type=self.build_type,
                                                   full_vars=full_vars),
                             lb=bounds[0],
                             ub=bounds[1] )

        # Add IPOPT options (some jhwi options were default)
        option_specs = { 'hessian_approximation': 'limited-memory',
                         'linear_solver': solver,
                         'limited_memory_max_history': 100,
                         'limited_memory_max_skipping': 1,
                         'mu_strategy': 'adaptive',
                         'tol': 1e-8,
                         'acceptable_tol': 1e-7,
                         'acceptable_iter': 100,
                         'max_iter': max_iter}
        for option in option_specs.keys():
            nlp.addOption(option, option_specs[option])

        (x, info) = nlp.solve(x0)

        # Set up variable names
        alphas = ['{0}_a'.format(i) for i in self.df_coordinates['id'].tolist()]
        if full_vars:
            longs = ['{0}_lng'.format(i) for i in self.df_coordinates['id'].tolist()]
            lats = ['{0}_lat'.format(i) for i in self.df_coordinates['id'].tolist()]
            headers = ['zeta', 'useless'] + longs + lats + alphas
        else:
            longs = ['{0}_lng'.format(i) for i in self.df_unknown['id'].tolist()]
            lats = ['{0}_lat'.format(i) for i in self.df_unknown['id'].tolist()]
            headers = ['zeta'] + longs + lats + alphas

        df = pd.DataFrame(data = [x],
                          columns = headers)
        df['obj_val'] = info['obj_val']
        df['status'] = info['status']
        df['status_msg'] = info['status_msg']
        df['status_msg'] = df['status_msg'].str.replace(';', '')

        #Add initial condition
        for ival in range(len(x0)):
            df['x0_'+str(ival)] = x0[ival]

        return df


    def gen_data(self,
                 x0,
                 rank = None,
                 max_iter = 25000,
                 full_vars = False):
        """ Run `self.solve` for many initial values.

        This function is the one called when running estimation in parallel.

        Args:
            x0 (np.ndarray): The array of initial conditions. Each row is an
                initial condition.
            rank (int): Process number in parallelized computing.

        Returns:
            DataFrame: simulation dataframe sorted by objective value

        Warning:
            Make sure `full_vars` is consistent with `x0`.
        """
        data = self.solve( x0[0, :],
                           max_iter=max_iter,
                           full_vars=full_vars)
        len_sim = x0.shape[0]
        for i in range(1, len_sim):
            i_val = x0[i, :]
            data = data.append( self.solve(i_val,
                                           max_iter = max_iter,
                                           full_vars = full_vars) )

        if rank != None:
            data['process'] = rank

        # Sort
        return data.sort_values('obj_val')




class EstimateModern(EstimateBase):
    """ Class for estimation procedures on modern dataset.

    Initializes the data used to be used:

    * loads processed datasets.
    * saves the dividing indices to go from variable array to individual
        components.
    * saves other data to speed up self.tile_nodiag and self.get_errors

    Args:
        build_type (str): One of "directional" or "non-directional".
        source (str): One of "all", "ancient_system" or "matched".
    """

    def __init__(self, build_type, source):
        EstimateBase.__init__(self, build_type)

        self.source = source

        ## Paths
        root = config.get('paths', 'root')
        process = config.get('paths', 'process_m')

        # Load processed datasets
        self.df_coordinates = pd.read_csv(root + process + self.source
                                          + '/coordinates.csv')
        if build_type == 'directional':
            self.df_iticount = pd.read_csv(root + process + self.source
                                           + '/estimation_data_directional.csv')
        elif build_type == 'non_directional':
            self.df_iticount = pd.read_csv(root + process + self.source
                                           + '/estimation_data_nondirectional.csv')
        else:
            raise ValueError("Initialize class with 'directional' or "
                             + "'non_directional'")

        # Save number of cities
        self.num_cities = len(self.df_coordinates)

        # Save indices to unpack argument of objective and gradient
        ## Note the useless variable is no longer included.
        self.div_indices = {'a_s': 1, 'a_e': 1 + self.num_cities}

        # Save trade shares (to speed up self.get_errors)
        self.shares = self.df_iticount['s_ij'].values

        # Save index (in varlist) of city that must be normalized to 100.
        self.id_normalized = {'all': 38, 'ancient_system': 12, 'matched': 7}
        self.index_normalized = (self.div_indices['a_s']
                                 + self.id_normalized[self.source]
                                 - 1)

        # Save array index that views array of size len(self.df_coordinates)
        # and selects off-diagonal elements. See self.tile_nodiag.
        i = np.repeat(np.arange(1, self.num_cities), self.num_cities)
        self.index_nodiag = i + np.arange(self.num_cities*(self.num_cities - 1))


    def get_errors(self, varlist, full_vars=False):
        """ Get the model and data trade share differences.

        Args:
            varlist (np.ndarray):it is composed of
                `[zeta, alpha]`.

        Returns:
            np.ndarray: the difference between data and model trade shares.
        """
        # Unpack arguments
        zeta = varlist[0]

        i = self.div_indices
        alpha = varlist[i['a_s']:]

        #assert len(lat_guess) == len(lng_guess)

        s_ij_model = self.s_ij_model(zeta,
                                     alpha,
                                     self.df_iticount['dist'].values
                                    )
        return self.shares - s_ij_model


    def get_bounds(self, set_elasticity=None):
        """ Fetch the upper and lower bounds for all entries in `varlist`.

        Args:
            set_elasticity (float): An imposed distance elasticity of trade.
                Optional.

        Returns:
            tuple: (lb, ub), where lb and ub are of type `list` for the bounds.
        """
        num_vars = 1 + self.num_cities

        # All variables should be positive
        lb = num_vars * [0.0]
        ub = num_vars * [1.0e20]

        if set_elasticity != None:
            lb[0] = set_elasticity
            ub[0] = set_elasticity

        # Normalize the city given in id_normalized to 100
        lb[self.index_normalized] = 100.0
        ub[self.index_normalized] = 100.0

        return (lb, ub)


    def initial_cond(self,
                     len_sim=None,
                     perturb=None,
                     perturb_type='rigid'):
        """ Gets initial condition(s) for `IPOPT`.

        Args:
            len_sim (int): Specifies the number of initial conditions to draw.
            perturb (float): A percentage deviation from the default initial
                value.
            perturb_type (str): Type of perturbation on the default initial
                value. If `'rigid'` then the whole initial value vector is
                multiplied by a scalar. If `'flexible'` then each element of
                the initial value vector is multiplied by a different scalar.
                Default is `'rigid'`.

        Returns:
            np.ndarray: The default initial condition if perturb is not
                specified, and an array with `len_sim` perturbed initial
                conditions.
        """
        # Form default initial value
        zeta = [20.0]
        alphas = 100*np.ones(self.num_cities)
        x0 = np.concatenate((zeta, alphas))
        #print(x0)

        # Perturb it
        if perturb != None:
            x0 = np.tile(x0, (len_sim, 1))
            if perturb_type == 'rigid':
                p = np.random.uniform(1-perturb,
                                      1+perturb,
                                      size=(len_sim, 1))
            elif perturb_type == 'flexible':
                p = np.random.uniform(1-perturb,
                                      1+perturb,
                                      size=(len_sim, x0.shape[1]))
            #print(p)
            x0 = x0*p

        return x0


    def solve(self,
              x0,
              max_iter = 25000,
              solver='ma57',
              set_elasticity=None):
        """ Solve the sum of squared distances minimization problem with IPOPT.

        Args:
            x0 (list): The initial value.
            max_iter (int): Maximum iterations before IPOPT stops.
            solver (str): Linear solver. 'ma57' is the default. If not
                available, use 'mumps'.

        Returns:
            A one-row dataframe with optimization information.
        """
        # Cast x0 as np.float64. See numpy bug in `self.haversine_approx`.
        x0 = np.float64(x0)

        # Set bounds
        bounds = self.get_bounds(set_elasticity)

        assert len(bounds[0]) == len(x0)
        #print('Low bound:')
        #print(bounds[0])
        #print('High bound:')
        #print(bounds[1])
        #print('max_iter:' + str(max_iter))

        nlp = ipopt.problem( n=len(x0),
                             m=0,
                             problem_obj = (
                                 OptimizerModern(build_type = self.build_type,
                                                 source = self.source)
                                 ),
                             lb=bounds[0],
                             ub=bounds[1] )

        # Add IPOPT options (some jhwi options were default)
        option_specs = { 'hessian_approximation': 'limited-memory',
                         'linear_solver': solver,
                         'limited_memory_max_history': 100,
                         'limited_memory_max_skipping': 1,
                         'mu_strategy': 'adaptive',
                         'tol': 1e-8,
                         'acceptable_tol': 1e-7,
                         'acceptable_iter': 100,
                         'max_iter': max_iter}
        for option in option_specs.keys():
            nlp.addOption(option, option_specs[option])

        (x, info) = nlp.solve(x0)

        # Set up variable names
        alphas = ['{0}_a'.format(i) for i in self.df_coordinates['id'].tolist()]
        headers = ['zeta'] + alphas

        df = pd.DataFrame(data = [x], columns = headers)
        df['obj_val'] = info['obj_val']
        df['status'] = info['status']
        df['status_msg'] = info['status_msg']
        df['status_msg'] = df['status_msg'].str.replace(';', '')

        #Add initial condition
        for ival in range(len(x0)):
            df['x0_'+str(ival)] = x0[ival]

        return df


    def gen_data(self,
                 x0,
                 rank = None,
                 max_iter = 25000,
                 set_elasticity=None):
        """ Run `self.solve` for many initial values.

        This function is the one called when running estimation in parallel.

        Args:
            x0 (np.ndarray): The array of initial conditions. Each row is an
                initial condition.
            rank (int): Process number in parallelized computing.

        Returns:
            DataFrame: simulation dataframe sorted by objective value
        """
        data = self.solve( x0[0, :],
                           max_iter=max_iter,
                           set_elasticity=set_elasticity)
        len_sim = x0.shape[0]
        for i in range(1, len_sim):
            i_val = x0[i, :]
            data = data.append( self.solve(i_val,
                                           max_iter = max_iter,
                                           set_elasticity = set_elasticity) )

        if rank != None:
            data['process'] = rank

        # Sort
        return data.sort_values('obj_val')


    def get_variance_gmm(self, varlist):
        """ Get the GMM variance-covariance matrix of the estimators

        Applies standard GMM formula. This function needs to be revised.
        """
        errors = self.get_errors(varlist)

        # Make column vector
        errors = np.expand_dims(errors, 1)

        # Evaluate errors jacobian at estimated parameter.
        jac = self.jac_errors(varlist)

        # Remove fixed a.
        jac = np.delete(jac, self.index_normalized, axis=1)

        #assert np.shape(jac) == (650, 48)

        # Build variance-covariance matrix
        bread_top = np.linalg.inv(np.dot(np.transpose(jac), jac))
        ham = np.linalg.multi_dot((np.transpose(jac),
                                   errors,
                                   np.transpose(errors),
                                   jac))
        bread_bottom = bread_top

        return np.linalg.multi_dot((bread_top, ham, bread_bottom))


    def get_variance(self, varlist, var_type='white'):
        """ Compute the variance-covariance matrix of the estimators.

        It can be computed according to the White formula, or with
        homoskedasticity.

        Args:
            var_type (str): One of 'white' or 'homo'.

        Returns:
            np.ndarray: The variance-covariance matrix of the estimators.
        """
        errors = self.get_errors(varlist)

        # Make column vector
        errors = np.expand_dims(errors, 1)

        # Evaluate errors jacobian at estimated parameter.
        jac = self.jac_errors(varlist)

        # Remove fixed a
        jac = np.delete(jac, self.index_normalized, axis=1)

        # Build variance-covariance matrix, according to var_type
        bread = np.linalg.inv(np.dot( np.transpose(jac), jac ))

        if var_type == 'white':
            ham = np.dot(np.transpose(jac * errors), jac * errors)
            return np.linalg.multi_dot((bread, ham, bread))

        elif var_type == 'homo':
            return (np.sum(errors**2) / len(errors)) * bread

        else:
            raise ValueError("Please specify the variance type to be one of "
                    + "'white' or 'homo'")


    def get_size(self, varlist, theta=4.0):
        """ Retrieve the fundamental size of cities.

        Recall Size_i is proportional to L_i T_i^(1/theta).

        Args:
            theta (float): The trade elasticity parameter that is assumed away.

        Returns:
            np.ndarray: The fundamental size of cities
        """
        # Unpack arguments
        zeta = varlist[0]
        i = self.div_indices
        alpha = varlist[i['a_s']:]

        distances = self.df_iticount['dist'].values

        factor_1 = alpha**(1 + 1.0/theta)

        ## Build summation
        # This part draws from self.s_ij_model()
        a = self.tile_nodiag(alpha, default = True)
        elems = a * (distances ** (-zeta))
        elems = np.reshape(elems, (self.num_cities, self.num_cities - 1))
        # Add within-city component. Assumed within-city distance: 30 km.
        own_factor = (30 ** (-zeta)) * alpha
        factor_2 = np.sum(elems, axis = 1).flatten() + own_factor

        sizes = factor_1 * factor_2
        return sizes


    def get_size_variance(self, varlist, var_type='white'):
        """ Get the fundamental size variance-covariance matrix.

        Applies Delta Method to get the variance-covariance matrix of the city
        size estimates.

        Returns:
            np.ndarray: The variance-covariance matrix of city sizes.
        """
        def size_for_grad(v):
            """ get_size function for autograd """
            return self.get_size(v)

        # Get Jacobian
        jac_size = jacobian(size_for_grad)
        # Evaluate
        j = jac_size(varlist)

        ## Remove fixed a.
        j = np.delete(j, self.index_normalized, axis=1)

        var = self.get_variance(varlist, var_type=var_type)

        return np.linalg.multi_dot((j, var, np.transpose(j)))


    def export_results(self, varlist):
        """ Export the estimation results.

        Exports zeta.csv, coordinates.csv, cities.csv, simulation.csv

        Args:
            varlist (np.ndarray): it is in jhwi format: `(zeta, a)`
        """
        # 1. Fetch standard error of estimates
        varlist_cov_white = self.get_variance(varlist,
                                              var_type='white')
        varlist_cov_homo = self.get_variance(varlist,
                                             var_type='homo')
        size_cov_white = self.get_size_variance(varlist, var_type='white')
        size_cov_homo = self.get_size_variance(varlist, var_type='homo')

        varlist_sd_white = np.sqrt( np.diag(varlist_cov_white) )
        varlist_sd_homo = np.sqrt( np.diag(varlist_cov_homo) )
        size_sd_white = np.sqrt( np.diag(size_cov_white) )
        size_sd_homo = np.sqrt( np.diag(size_cov_homo) )

        # 2. Unpack varlist arguments
        zeta = varlist[0]
        i = self.div_indices
        alpha = varlist[i['a_s']:]

        # 3. Save zeta.csv
        df_zeta = pd.DataFrame([[zeta,
                                 varlist_sd_white[0],
                                 varlist_sd_homo[0]]], columns=['zeta',
                                                                'zeta_sd_white',
                                                                'zeta_sd_homo']
                              )
        df_zeta.to_csv('./estim_results/modern/'+self.source+'/zeta.csv', index=False)

        # 4. Save sizes and alphas (+ standard errors)
        size = self.get_size(varlist)

        alpha = varlist[i['a_s']:]
        alpha_white = varlist_sd_white[i['a_s']:]
        alpha_homo = varlist_sd_homo[i['a_s']:]

        ## Insert missing s.e. for fixed city in alpha
        alpha_white = np.insert(alpha_white,
                                self.id_normalized[self.source] - 1,
                                np.nan)
        alpha_homo = np.insert(alpha_homo,
                               self.id_normalized[self.source] - 1,
                               np.nan)

        ids_city = self.df_coordinates['id'].values
        city_array = np.column_stack((ids_city,
                                      alpha,
                                      alpha_white,
                                      alpha_homo,
                                      size,
                                      size_sd_white,
                                      size_sd_homo))
        cities = pd.DataFrame( city_array,
                               columns = ['id',
                                          'alpha',
                                          'alpha_sd_white',
                                          'alpha_sd_homo',
                                          'size',
                                          'size_sd_white',
                                          'size_sd_homo']
                             )
        cities.to_csv('./estim_results/modern/'+self.source+'/cities.csv', index=False)




# Now define optimization problems for IPOPT
class OptimizerAncient(EstimateAncient):

    def __init__(self,
                 build_type,
                 omega = None,
                 cities_to_known = [],
                 cities_to_unknown = [],
                 cities_to_drop = [],
                 full_vars = False,
                 scale = None):
        EstimateAncient.__init__(self,
                                 build_type,
                                 omega = omega,
                                 cities_to_known = cities_to_known,
                                 cities_to_unknown = cities_to_unknown,
                                 cities_to_drop = cities_to_drop)
        self.full_vars = full_vars

        if scale is None:
            self.scale = 1.0
        else:
            self.scale = scale

        # Set objective and gradient depending on whether we have an omega.
        if omega is None:
            def obj_fn(varlist):
                return self.sqerr_sum(varlist / self.scale,
                                      full_vars = self.full_vars)
            ## Grad here depends on whether full_vars is on or off.
            #if self.full_vars:
            #    self.grad_fn = self.grad_full_vars
            #else:
            #    self.grad_fn = self.grad
        else:
            def obj_fn(varlist):
                return self.sqerr_sum_gen(varlist / self.scale,
                                          self.omega_inv,
                                          full_vars = self.full_vars)
            #self.grad_fn = self.grad_gen

        self.obj_fn = obj_fn
        self.grad_fn = grad(self.obj_fn)


    def objective(self, varlist):
        return self.obj_fn(varlist)


    def gradient(self, varlist):
        return self.grad_fn(varlist)




class OptimizerToblerWeinburg(EstimateToblerWeinburg):

    def __init__(self,
                 build_type = 'non_directional',
                 omega = None,
                 cities_to_known = [],
                 cities_to_unknown = [],
                 cities_to_drop = []):
        EstimateToblerWeinburg.__init__(self,
                                 build_type = build_type,
                                 omega = omega,
                                 cities_to_known = cities_to_known,
                                 cities_to_unknown = cities_to_unknown,
                                 cities_to_drop = cities_to_drop)

        def obj_fn(varlist):
            return self.sqerr_sum(varlist)

        self.obj_fn = obj_fn
        self.grad_fn = grad(self.obj_fn)


    def objective(self, varlist):
        return self.obj_fn(varlist)


    def gradient(self, varlist):
        return self.grad_fn(varlist)




class OptimizerAncientMLE(EstimateAncientMLE):

    def __init__(self, build_type, full_vars):
        EstimateAncientMLE.__init__(self, build_type)
        self.full_vars = full_vars

    def objective(self, varlist):
        return self.mle_objective(varlist, full_vars = self.full_vars)

    def gradient(self, varlist):
        #print(varlist)
        if self.full_vars:
            return self.grad_full_vars(varlist)
        else:
            return self.grad(varlist)




class OptimizerModern(EstimateModern):

    def __init__(self, build_type, source):
        EstimateModern.__init__(self, build_type, source)

    def objective(self, varlist):
        return self.sqerr_sum(varlist)

    def gradient(self, varlist):
        #print(varlist)
        return self.grad(varlist)
