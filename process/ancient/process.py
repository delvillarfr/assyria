'''
This module contains all processing functions to get the data used for
estimation.
'''

import os
import ConfigParser
import pandas as pd
import numpy as np

import sys



# Configuration


config = ConfigParser.ConfigParser()

## file keys.ini should be in process.py parent directory.
config.read('../../keys.ini')

## Paths
root = config.get('paths', 'root')
root_jhwi = config.get('paths', 'root_jhwi_a')
raw_iticount = config.get('paths', 'raw_iticount_a')
raw_city_name = config.get('paths', 'raw_city_name_a')
raw_coord = config.get('paths', 'raw_coord_a')
raw_constr_dynamic = config.get('paths', 'raw_constr_dynamic_a')
raw_constr_static = config.get('paths', 'raw_constr_static_a')
raw_constr_format = config.get('paths', 'raw_constr_format_a')
process = config.get('paths', 'process_a')





# 1. Filter and construct indices


class Process(object):
    """ Parent class for directional and non-directional data generation """

    def __init__(self, build_type):
        '''
        build_type: str. One of "directional" or "non-directional".

        Loads all raw datasets.
        Stores all filenames of datasets that can be produced.
        Saves the filtering and variable-building methods as attributes.
        '''
        # Load raw datasets
        self.df_iticount = pd.read_csv(root + raw_iticount)
        self.df_coordinates = pd.read_csv(root + raw_coord)
        self.df_constr_dyn = pd.read_csv(root + raw_constr_dynamic)
        self.df_constr_stat = pd.read_csv(root + raw_constr_static)
        self.df_constr_format = pd.read_csv(root + raw_constr_format)
        self.df_city_name = pd.read_csv(root + raw_city_name, header = None)
        self.df_city_name = self.df_city_name.rename(columns = {0: 'city_name'})

        self.id_path = root + process + 'id.csv'

        if build_type == 'directional':
            self.coordinates_path = root + process + 'coordinates_dir.csv'
            self.iticount_path = root + process + 'iticount_dir.csv'
            self.id_equiv_path = root + process + 'id_equivalence_dir.csv'
            self.constr_path = root + process + 'constraints_dir_'

            # Methods as attributes
            self.filter = self.filter_positive_imports
            self.build_vars = self.build_vars_directional


        elif build_type == 'non_directional':
            self.coordinates_path = root + process + 'coordinates_nondir.csv'
            self.iticount_path = root + process + 'iticount_nondir.csv'
            self.id_equiv_path = root + process + 'id_equivalence_nondir.csv'
            self.constr_path = root + process + 'constraints_nondir_'

            # Methods as attributes
            self.filter = self.filter_positive_activity
            self.build_vars = self.build_vars_nondirectional

        else:
            raise ValueError("Initialize class with 'directional' or "
                             + "'non_directional'")


    def create_id(self, names):
        """
        names: pd.Series. A series of city names.

        The id to use will be the first two letters of the city followed by
        their place in the list. e.g. Kanes is the first of two cities starting
        with "Ka". Its id is "ka01".

        Assumes that the city_names dataset is our universe.
        Returns a pd.Series with the created id's.
        """
        names = names.to_frame('name')
        names['id_head'] = (names['name'].str[:2]
                                         .str.lower()
                           )
        names['id_tail'] = 0
        for n in names['id_head'].unique():
            size = len( names.loc[ names['id_head'] == n, : ] )
            names.loc[ names['id_head'] == n, 'id_tail' ] = np.arange(1, size+1)

        ## Now concatenate
        names['id_tail'] = names['id_tail'].astype(str)
        names['id'] = names['id_head'].str.cat(names['id_tail'], sep='0')

        return names['id']


    def fetch_df_id(self):
        """
        Genrates id's for our universe, which is assumed to be the cities in
        the city_name dataset.
        """
        names = self.df_city_name.copy()
        names['id'] = self.create_id(names['city_name'])
        names['id_old'] = names.index + 1

        return names


    def filter_positive_imports(self):
        """
        Keeps rows in iticount data where importing (id_city1) and exporting
        (id_city2) cities import at least once.

        Returns 1-dim np.array with id's of participating cities.
        """
        # Get N_j
        N_j = (self.df_iticount.groupby('id_city2')
                       .sum()['iti_joint']
                       .rename('N_j')
              )

        # Add this info keep obs with positive aggregate imports
        iticount = self.df_iticount.join(N_j, on='id_city2')
        iticount = iticount.loc[ iticount['N_j'] > 0 ]

        return iticount['id_city2'].unique()


    def filter_positive_activity(self):
        """
        Keeps rows in iticount data where importing (id_city1) and exporting
        (id_city2) cities import or export at least once.

        Returns 1-dim np.array with id's of participating cities.
        """
        # Add N_ij and N_ji for all i, j.
        iticount = self.df_iticount.sort_values(['id_city2', 'id_city1'])
        iticount_reversed = iticount.sort_values(['id_city1', 'id_city2'])
        iticount['iti_joint'] = (iticount['iti_joint'].values
                                + iticount_reversed['iti_joint'].values
                               )

        # Get N_j
        N_j = (iticount.groupby('id_city2')
                       .sum()['iti_joint']
                       .rename('N_j')
              )

        # Add this info keep obs with positive aggregate imports
        iticount = iticount.join(N_j, on='id_city2')

        iticount = iticount.loc[ iticount['N_j'] > 0 ]

        return iticount['id_city2'].unique()


    def convert_id(self, ids_left):
        """
        ids_left: list-like. It is the id of cities to consider.

        Returns ID conversion table for all cities in ids_left.
        """
        # Create old id
        city_name = self.df_city_name.copy()
        city_name['id_old'] = city_name.index + 1

        # Keep only filtered cities
        city_name = city_name.loc[ city_name['id_old'].isin(ids_left) ]

        # Get Joonhwi's id (it's not alphabetically ordered...)
        df = self.df_coordinates.merge(city_name,
                                       how='left',
                                       left_on='name',
                                       right_on='city_name'
                                      )
        df = (df.loc[ df['id_old'].isin(ids_left) ]
                                  .reset_index(drop = True)
                      )
        df['id_jhwi'] = df.index + 1
        df['id_old'] = df['id_old'].astype(int)
        df = df[['city_name', 'id_old', 'id_jhwi']].sort_values('id_old')

        df = df.merge(self.fetch_df_id()[['id', 'city_name']],
                      how='left',
                      on='city_name')

        cols = [ 'city_name',
                 'id_old',
                 'id_jhwi',
                 'id' ]

        return df[cols]


    def fetch_df_id_equiv(self):
        """
        Generates an id's equivalence table for reference.
        """
        ids_left = self.filter()
        result = self.convert_id(ids_left)

        return result


    def select_trade_data(self):
        """
        Selects trade data according to the filtering function and adds ids.
        """
        df_id = self.fetch_df_id_equiv()
        df_id = df_id[['id_old', 'id_jhwi', 'id']]

        iticount = self.df_iticount.copy()
        d = {'i': '1', 'j': '2'}
        for status in d.keys():
            iticount = iticount.merge(df_id,
                                      how='right',
                                      left_on='id_city'+d[status],
                                      right_on='id_old')
            iticount = iticount.rename(columns = {'id_old': 'id_old_'+status,
                                                  'id_jhwi': 'id_jhwi_'+status,
                                                  'id': 'id_'+status}
                                      )

        # This sorting is useful for testing
        iticount = (iticount.sort_values(['id_old_j', 'id_old_i'])
                            .reset_index(drop=True)
                   )

        return iticount


    def build_vars_directional(self, df):
        """
        df: DataFrame. It is the filtered iticount DataFrame, given by
        self.select_trade_data().

        Returns the DataFrame with all variables used in the directional
        analysis. The dataset level is ordered city-pairs.
        """
        # Rename variables
        df = df.rename(columns = {'iti_joint': 'N_ij',
                                  'certainty1': 'cert_i',
                                  'certainty2': 'cert_j'}
                      )

        # Add N_i, N_j
        for status in ['i', 'j']:
            N = (df.groupby('id_'+status)
                         .sum()['N_ij']
                         .rename('N_'+status)
                 )

            # Add this info
            df = df.join(N, on='id_'+status)

        # Add s_ij ( adding s_ji is overkill: the data level is (i, j) )
        df[ 's_ij' ] = df['N_ij'] / df['N_j']

        return df


    def build_vars_nondirectional(self, df):
        """
        df: DataFrame. It is the filtered iticount DataFrame, given by
        self.select_trade_data().

        Returns the DataFrame with all variables used in the non-directional
        analysis. Notice the database level is unordered city-pairs.
        """
        # Add N_ij and N_ji for all i, j.
        df = df.sort_values(['id_j', 'id_i'])
        df_reversed = df.sort_values(['id_i', 'id_j'])
        df['iti_joint'] = df['iti_joint'].values + df_reversed['iti_joint'].values

        # From this point on, the procedure equals that of the directional.
        return self.build_vars_directional(df)


    def fetch_df_iticount(self, jhwi=True):
        """
        jhwi: bool. Include jhwi's id.

        Generates the trade dataset to be used in estimation.
        """
        df = self.select_trade_data()
        df = self.build_vars(df)

        df = (df.sort_values(['id_jhwi_j', 'id_jhwi_i'])
                .reset_index(drop=True)
             )

        # Select columns
        cols = ['cert_i',
                'cert_j',
                'id_i',
                'id_j',
                'N_ij',
                'N_i',
                'N_j',
                's_ij']
        if jhwi == True:
            cols += ['id_jhwi_i', 'id_jhwi_j']

        df = df[cols]

        return df


    def fetch_df_coordinates(self):
        '''
        Returns the coordinates table with the appropriate id.
        '''
        df_id = self.fetch_df_id_equiv()

        coordinates = self.df_coordinates.drop('id', axis=1)
        coordinates = coordinates.merge(df_id,
                                        how='right',
                                        left_on='name',
                                        right_on='city_name'
                                       )
        #print(coordinates)
        #coordinates = coordinates.loc[ pd.notnull(coordinates['id']) ]
        #print(coordinates)

        # Add id_jhwi for testing
        coordinates = coordinates[['id',
                                   'id_jhwi',
                                   'long_x',
                                   'lat_y',
                                   'cert',
                                   'validity']]

        return coordinates


    def fetch_df_constr(self, dtype, id_used = 'id'):
        """
        dtype: str. One of "static" or "dynamic".
        id_used: str. Specifies which id to replace with. Used for testing.

        Updates the ids in the constraints datasets
        """
        if dtype == "dynamic":
            constr = self.df_constr_dyn.copy()
        elif dtype == "static":
            constr = self.df_constr_stat.copy()
        else:
            raise ValueError("Must be called with dtype equal to one of "
                             + "'static' or 'dynamic'")

        if id_used == 'id':
            df_equiv = self.fetch_df_id()
            id_old = df_equiv['id_old'].values
            id_new = df_equiv[id_used].values

        else:
            df_equiv = self.fetch_df_id_equiv()
            id_old = df_equiv['id_old'].values
            id_new = df_equiv[id_used].values

        variables = ['upper_y', 'lower_y', 'upper_x', 'lower_x']
        constr[variables] = (constr[variables].replace(0, np.nan)
                                              .replace(id_old, id_new)
                            )
        constr = constr.merge(self.fetch_df_id(),
                              how='left',
                              left_on='anccity',
                              right_on='city_name'
                             )
        constr = (constr[['id', 'certainty'] + variables]
                        .rename(columns = {'upper_y': 'ub_lambda',
                                           'lower_y': 'lb_lambda',
                                           'upper_x': 'ub_varphi',
                                           'lower_x': 'lb_varphi'}
                               )
                 )

        return constr


    def refresh_all(self):
        self.fetch_df_id().to_csv(self.id_path, index=False)
        self.fetch_df_id_equiv().to_csv(self.id_equiv_path, index=False)
        self.fetch_df_iticount(jhwi=False).to_csv(self.iticount_path,
                                                  index=False)
        self.fetch_df_coordinates().to_csv(self.coordinates_path, index=False)
        self.fetch_df_constr('static').to_csv(self.constr_path
                                              + 'stat.csv', index=False)
        self.fetch_df_constr('dynamic').to_csv(self.constr_path
                                               + 'dyn.csv', index=False)


## Test all data except iticount is the same.
#
#p = Process('directional')
#m_i = p.df_iticount['id_city1'] == 20
#m_e = p.df_iticount['id_city2'] == 20
#df_id_equiv = p.fetch_df_id_equiv()
#df_id = p.fetch_df_id()
#df_iticount = p.fetch_df_iticount()
#df_coordinates = p.fetch_df_coordinates()
#coords = p.fetch_df_coordinates()
#for d in ['id',
#          'constraints_dir_dyn',
#          'constraints_dir_stat',
#          'constraints_nondir_dyn',
#          'constraints_nondir_stat']:
#    #      'coordinates_dir',
#    #      'coordinates_nondir']:
#    p_new = pd.read_csv(d + '.csv')
#    p_old = pd.read_csv('../../process_old/ancient/' + d + '.csv')
#
#    pd.testing.assert_frame_equal(p_new, p_old)
