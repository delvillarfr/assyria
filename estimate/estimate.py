'''
Model Parameter Estimation
'''

import os
import ConfigParser


import pandas as pd
import autograd.numpy as np
from autograd import grad
import ipopt

import sys



# Configuration


config = ConfigParser.ConfigParser()

## file keys.ini should be in process.py parent directory.
config.read(os.path.dirname(os.path.dirname(__file__)) + '/keys.ini')

## Paths
root = config.get('paths', 'root')
root_jhwi = config.get('paths', 'root_jhwi')
process = config.get('paths', 'process')



class Estimate(object):

    def __init__(self, build_type, lat = (36, 42), lng = (27, 45)):
        '''
        build_type: str. One of "directional" or "non-directional".
        lat: 2-element tuple. Contains assumed lower and upper latitude bounds
        lng: 2-element tuple. Contains assumed lower and upper longitude bounds

        Loads all processed datasets.

            Sets coordinates in degrees.
        Sets known and unknown coordinates datasets as separate attributes.

        Adds selected vars from trade data with known coordinates as
        attribute. This will be the main dataset to work with further down.

        Saves the gradient of the objective function as attribute, to avoid
        calling autograd multiple times.
        '''
        self.lat = lat
        self.lng = lng
        self.build_type = build_type

        # Load processed datasets
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

        # Save number of cities
        self.num_cities = len(self.df_coordinates)

        # Set coordinates in degrees (this should be done in process)
        for v in ['long_x', 'lat_y']:
            self.df_coordinates[v] = np.rad2deg(self.df_coordinates[v].values,
                                                dtype='float64')

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

        # Add known locations to trade data
        self.df_main = self.df_iticount[['id_j', 'id_i', 's_ij']]
        for status in ['i', 'j']:
            self.df_main = (self.df_main.merge(self.df_known,
                                              how = 'left',
                                              left_on = 'id_'+status,
                                              right_on = 'id'
                                             )
                            .rename(columns={'long_x': 'long_x_'+status,
                                             'lat_y': 'lat_y_'+status})
                           )
        self.df_main = self.df_main.drop(['id_x', 'id_y'], axis = 1)

        # Save Gradient as attribute
        def objective(varlist):
            ''' This is the formulation for autograd. '''
            return self.sqerr_sum(varlist)
        self.grad = grad(objective)

        def objective_full_vars(varlist):
            ''' This is the formulation for autograd. '''
            return self.sqerr_sum(varlist, full_vars=True)
        self.grad_full_vars = grad(objective_full_vars)

        # Save array index that views array of size len(self.df_coordinates)
        # and selects off-diagonal elements. See self.tile_nodiag.
        i = np.repeat(np.arange(1, self.num_cities), self.num_cities)
        self.index_nodiag = i + np.arange(self.num_cities*(self.num_cities - 1))

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
                   'a_s': 2 + 2*self.num_cities},
            False: {'long_s': 1,
                    'long_e': 1 + self.num_cities_unknown,
                    'lat_s': 1 + self.num_cities_unknown,
                    'lat_e': 1 + 2*self.num_cities_unknown,
                    'a_s': 1 + 2*self.num_cities_unknown,
                    'a_e': 1 + 2*self.num_cities_unknown + self.num_cities}
        }

        # Save trade shares located in df_main
        self.df_shares = self.df_iticount['s_ij'].values



    def haversine_approx(self, coord_i, coord_j):
        '''
        coord_i, coord_j: np.array. 2 columns, len(iticount) rows. First column
        (column 0) is latitude, second column (column 1) is longitude.

        Returns the approximation of the Haversine formula described in the
        estimation section of the paper.
        '''
        factor_out = 10000.0/90
        factor_in = np.cos(37.9 * np.pi / 180)

        lat_diff = coord_j[:, 0] - coord_i[:, 0]
        lng_diff = coord_j[:, 1] - coord_i[:, 1]

        diff = np.column_stack((lat_diff, factor_in * lng_diff))

        return factor_out * np.sqrt( np.sum(diff**2, axis=1) )



    def tile_nodiag(self, arr):
        '''
        arr: np.array. A 1-dim array of length self.num_cities.

        Returns an array repeating arr the number of times given by
        self.num_cities, but extracting value in index j on repetition j.

        example: If arr = np.array([1, 2, 3]) then self.tile_nodiag(arr) returns
        np.array([2, 3, 1, 3, 1, 2]).
        '''
        arr_tiled = np.tile(arr, self.num_cities)

        return arr_tiled[self.index_nodiag]


    def get_coordinate_pairs(self, lat_guess, lng_guess, full_vars=False):
        '''
        full_vars: bool. If True, the known coordinates are included as
        variables of the objective and gradient.

        This is an alternative implementation of the fetching distance process,

        Leverages the fact that the iticount data is sorted according to
        id_jhwi_j first, then by id_jhwi_i, and the coordinates are sorted
        according to id_jhwi.
        '''
        if full_vars:
            lats = lat_guess
            longs = lng_guess
        else:
            lats = np.concatenate((self.df_known['lat_y'].values, lat_guess))
            longs = np.concatenate((self.df_known['long_x'].values, lng_guess))

        coord_j = np.column_stack((
            np.repeat(lats, self.num_cities-1),
            np.repeat(longs, self.num_cities-1)
        ))
        #assert len(lats) == len(longs)
        coord_i = np.column_stack((
            self.tile_nodiag(lats),
            self.tile_nodiag(longs)
        ))

        return (coord_i, coord_j)


    def fetch_dist(self, lat_guess, lng_guess, full_vars=False):
        '''
        Wrapper
        '''
        coords = self.get_coordinate_pairs(lat_guess, lng_guess, full_vars)
        return self.haversine_approx( coords[0], coords[1] )


    def s_ij_model(self, zeta, alpha, distances):
        '''
        zeta: float. The distance elasticity of trade.
        alpha: np.array. One for each importing city.
        distances: np.array. Contains distances between all j, i pairs,
        excluding j, j pairs.

        Idea: cast elements as matrix, add over axis=0,
        repeat (number of cities - 1), divide elements by this new 1-dim array.

        Returns np.array: the model-predicted trade shares
        '''
        a = self.tile_nodiag(alpha)
        elems = a * (distances ** (-zeta))

        denom = np.reshape(elems, (self.num_cities, self.num_cities - 1))
        denom = np.sum(denom, axis = 1)
        denom = np.repeat(denom, self.num_cities-1)

        return elems / denom


    def sqerr_sum(self, varlist, full_vars=False):
        '''
        varlist = np.array([zeta, alpha, lat_guess, lng_guess]).

        Returns the value of the objective function given the data and the
        model trade shares.
        '''
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
        diff = self.df_shares - s_ij_model
        return np.dot(diff, diff)


    def replace_id_coord(self, constr, drop_wahsusana=False):
        '''
        constr: pd.DataFrame. Specifies upper and lower bounds for
        coordinates of cities.

        Replaces the city id with its corresponding latitudes and longitudes in
        the constraints datasets.
        '''
        constr = constr.copy()

        if drop_wahsusana:
            v = ['lb_lambda', 'ub_lambda', 'lb_varphi', 'ub_varphi']
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


    def get_bounds(self, constr, full_vars=False):
        '''
        Returns (lb, ub), where lb and ub are lists for the bounds.
        '''
        # Build specs: Ursu does not participate if directional, Kanes is in 
        # second position in coordinates dataframe if directional, third
        # otherwise.
        if self.build_type == 'directional':
            constr = constr[constr['id'] != 'ur01']
            kanes_loc = 1
        else:
            kanes_loc = 2

        if full_vars:
            num_vars = 2 + 3*self.num_cities
        else:
            num_vars = 1 + 2*self.num_cities_unknown + self.num_cities

        lb = num_vars * [-1.0e20]
        ub = num_vars * [1.0e20]

        # zeta should be larger than zero
        lb[0] = 0.0


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
                lb[i[c+'_s']: i[c+'_e']] = constr['lb_'+dit[c][0]].tolist()
                ub[i[c+'_s']: i[c+'_e']] = constr['ub_'+dit[c][0]].tolist()

        lb[i['a_s']:] = self.num_cities * [0.0]

        # Kanes' alphas are normalized to 100
        lb[i['a_s'] + kanes_loc] = 100.0
        ub[i['a_s'] + kanes_loc] = 100.0

        return (lb, ub)


    def initial_cond(self, len_sim=None, perturb=None, full_vars=False) :
        '''
        len_sim: int. Specifies the number of draws to take.
        perturb: float. Specifies a percentage deviation from the default
        initial value.

        Returns default initial condition if perturb is not specified, and an
        array of perturbed values of dimension (len_sim, numvars)
        '''
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

        # Perturb it
        if perturb != None:
            x0 = np.tile(x0, (len_sim, 1))
            p = np.random.uniform(1-perturb, 1+perturb, size=(len_sim, 1))
            x0 = x0*p

        return x0


    def solve(self, x0, constraint_type='static', full_vars=False):
        '''
        x0: list. It is the initial value.
        constraint_type: str. One of 'static' or 'dynamic'.
        Returns a one-row dataframe with optimization information.
        '''
        # Set bounds
        if constraint_type == 'static':
            constr = self.replace_id_coord(self.df_constr_stat)
        elif constraint_type == 'dynamic':
            constr = self.replace_id_coord(self.df_constr_dyn)
        else:
            raise ValueError("Please specify the constraint type to be "
                             + "'static' or 'dynamic'")
        bounds = self.get_bounds(constr, full_vars)

        print(len(bounds))
        print(len(x0))
        assert len(bounds[0]) == len(x0)

        nlp = ipopt.problem( n=len(x0),
                             m=0,
                             problem_obj=Optimizer(self.build_type),
                             lb=bounds[0],
                             ub=bounds[1] )

        # Add IPOPT options (some jhwi options were default)
        option_specs = { 'hessian_approximation': 'limited-memory',
                         'linear_solver': 'ma57',
                         'limited_memory_max_history': 100,
                         'limited_memory_max_skipping': 1,
                         'mu_strategy': 'adaptive',
                         'tol': 1e-8,
                         'acceptable_tol': 1e-7,
                         'acceptable_iter': 100,
                         'max_iter': 25000 }
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


    def gen_data(self, len_sim, perturb, rank=None, full_vars=False):
        '''
        rank: int. Process number in parallelized computing.
        Returns simulation dataframe sorted by objective value
        '''
        # Get initial values
        x0 = self.initial_cond(len_sim, perturb, full_vars)

        data = self.solve( x0[0, :], full_vars=full_vars )
        for i in range(1, len_sim):
            i_val = x0[i, :]
            data = data.append( self.solve(i_val, full_vars=full_vars) )

        if rank != None:
            data['process'] = rank

        # Sort
        return data.sort_values('obj_val')


    def get_best_result(self, results):
        '''
        results: pd.DataFrame. It is the output of the parallelized execution.
        returns the row with minimum objective function value.
        '''
        r = results.sort_values('obj_val')

        # Discard results that are result in invalid numbers
        r = r.loc[ r['status'] != -13, :]

        return r.head(1)


    def resolve(self, result):
        '''
        result: pd.DataFrame. Output of self.get_best_result

        Recursively digs into the coordinates results if the maximum number of
        iterations was reached. Otherwise it returns the best solution.
        '''
        if result['status'].iloc[0] == -1:
            names = (['zeta']
                + ['{0}_lng'.format(i) for i in self.df_unknown['id'].tolist()]
                + ['{0}_lat'.format(i) for i in self.df_unknown['id'].tolist()]
                + ['{0}_a'.format(i) for i in self.df_coordinates['id'].tolist()]
                )
            init_val = result[names].values.flatten()

            return self.resolve(self.solve(init_val))

        else:
            return result


    def input_to_jhwi(self, x):
        '''
        x: pd.DataFrame. arguments for my function.

        Returns the initial value to evaluate the MATLAB objective function.
        '''
        x = x.values.flatten()

        i = self.div_indices[False]
        res = np.concatenate(([x[0]/2.0, 4],
                              self.df_known['long_x'].values,
                              x[i['long_s']: i['long_e']],
                              self.df_known['lat_y'].values,
                              x[i['lat_s']: i['lat_e']],
                              x[i['a_s']: i['a_e'] + 1]))
        pd.Series(res).to_csv('input_to_jhwi.csv', index=False)


# Now define optimization problem
class Optimizer(Estimate):

    def __init__(self, build_type):
        Estimate.__init__(self, build_type)

    def objective(self, varlist):
        return self.sqerr_sum(varlist)

    def gradient(self, varlist):
        return self.grad(varlist)
