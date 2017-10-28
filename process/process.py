'''
This module contains all processing functions to get the data used for
estimation.
'''

import os
import configparser
import pandas as pd
import numpy as np

import sys



# 0. configuration


config = configparser.ConfigParser()

## file keys.ini should be in process.py parent directory.
config.read(os.path.dirname(os.path.dirname(__file__)) + '/keys.ini')

## Paths
root = config['paths']['root']
root_jhwi = config['paths']['root_jhwi']
raw_iticount = config['paths']['raw_iticount']
raw_city_name = config['paths']['raw_city_name']
raw_coord = config['paths']['raw_coord']
raw_constr_dynamic = config['paths']['raw_constr_dynamic']
raw_constr_static = config['paths']['raw_constr_static']
raw_constr_format = config['paths']['raw_constr_format']
process = config['paths']['process']





# 1. Filter and construct indices


class Process(object):
    """ Parent class for directional and non-directional data generation """

    def __init__(self, build_type): 
        '''
        build_type: str. One of "directional" and "non-directional".

        Stores all filenames of datasets that can be produced.
        Loads all raw datasets. 
        Saves the filering and variable-building methods as attributes.
        '''
        # Load raw datasets
        self.df_iticount = pd.read_csv(root + raw_iticount)
        self.df_coordinates = pd.read_csv(root + raw_coord)
        self.df_constr_dyn = pd.read_csv(root + raw_constr_dynamic)
        self.df_constr_stat = pd.read_csv(root + raw_constr_static)
        self.df_constr_format = pd.read_csv(root + raw_constr_format)
        self.df_city_name = pd.read_csv(root + raw_city_name, header = None)
        self.df_city_name = self.df_city_name.rename(columns = {0: 'city_name'})
        

        if build_type == 'directional':
            self.coordinates_path = root + process + 'coordinates_dir.csv'
            self.main_path = root + process + 'main_dir.csv'
            self.id_path = root + process + 'id_equivalence_dir.csv'

            # Methods as attributes
            self.filter = self.filter_positive_imports
            self.build_vars = self.build_vars_directional


        elif build_type == 'non_directional':
            self.id_path = root + process + 'id_equivalence_nondir.csv'
            self.main_path = root + process + 'main_nondir.csv'
            self.coordinates_path = root + process + 'coordinates_nondir.csv'

            # Methods as attributes
            self.filter = self.filter_positive_activity
            self.build_vars = self.build_vars_nondirectional

        else:
            raise ValueError("Initialize class with 'directional' or "
                             + "'non_directional'")


    def filter_positive_imports(self):
        """
        Keeps rows in iticount data where importing (anccityid1) and exporting
        (anccityid2) cities import at least once.

        Returns 1-dim np.array with id's of participating cities.
        """
        # Get N_j
        N_j = (self.df_iticount.groupby('anccityid2')
                       .sum()['iticount']
                       .rename('N_j')
              )

        # Add this info keep obs with positive aggregate imports
        iticount = self.df_iticount.join(N_j, on='anccityid2')
        iticount = iticount.loc[ iticount['N_j'] > 0 ]
        
        return iticount['anccityid2'].unique()


    def filter_positive_activity(self):
        """
        Keeps rows in iticount data where importing (anccityid1) and exporting
        (anccityid2) cities import or export at least once.

        Returns 1-dim np.array with id's of participating cities.
        """
        # Add N_ij and N_ji for all i, j.
        iticount = self.df_iticount.sort_values(['anccityid2', 'anccityid1'])
        iticount_reversed = iticount.sort_values(['anccityid1', 'anccityid2'])
        iticount['iticount'] = (iticount['iticount'].values
                                + iticount_reversed['iticount'].values
                               )

        # Get N_j
        N_j = (iticount.groupby('anccityid2')
                       .sum()['iticount']
                       .rename('N_j')
              )

        # Add this info keep obs with positive aggregate imports
        iticount = iticount.join(N_j, on='anccityid2')
        
        iticount = iticount.loc[ iticount['N_j'] > 0 ]
        
        return iticount['anccityid2'].unique()


    def create_id(self, ids_left):
        """
        ids_left: list-like. It is the id of cities to consider.

        Generates ID conversion table for all cities in ids_left.

        The id to use will be the first two letters of the city followed by their
        place in the list. e.g. Kanes is the first of two cities starting with
        "Ka". Its id is "ka01".
        """
        # Create old id
        city_name = self.df_city_name.copy()
        city_name['id_old'] = city_name.index + 1

        # Keep only filtered cities
        city_name = city_name.loc[ city_name['id_old'].isin(ids_left) ]

        # Get Joonhwi's id (it's not alphabetically ordered...)
        coordinates = self.df_coordinates.merge(city_name,
                                                how='left',
                                                left_on='name',
                                                right_on='city_name'
                                               )
        coordinates = (coordinates.loc[ coordinates['id_old'].isin(ids_left) ]
                                  .reset_index(False)
                      )
        coordinates['id_jhwi'] = coordinates.index + 1
        coordinates['id_old'] = coordinates['id_old'].astype(int)
        main = coordinates[['city_name',
                            'id_old',
                            'id_jhwi']].sort_values('id_old')

        # Create new id.
        main['id_head'] = (main['city_name'].str[:2]
                                             .str.lower()
                           )
        main['id_tail'] = 0
        for n in main['id_head'].unique():
            size = len( main.loc[ main['id_head'] == n, : ] )
            main.loc[ main['id_head'] == n, 'id_tail' ] = np.arange(1, size+1)

        ## Now concatenate
        main['id_tail'] = main['id_tail'].astype(str)
        main['id'] = main['id_head'].str.cat(main['id_tail'], sep='0')

        cols = [ 'city_name',
                 'id_old',
                 'id_jhwi',
                 'id' ]

        return main[cols]


    def fetch_df_id(self):
        """
        Generates an id's equivalence table for reference.
        """
        ids_left = self.filter()
        result = self.create_id(ids_left)

        return result


    def select_trade_data(self):
        """
        Selects trade data according to the filtering function and adds ids.
        """
        df_id = self.fetch_df_id()
        df_id = df_id[['id_old', 'id_jhwi', 'id']]
        
        iticount = self.df_iticount.copy()
        d = {'i': '1', 'j': '2'}
        for status in d.keys():
            iticount = iticount.merge(df_id,
                                      how='right',
                                      left_on='anccityid'+d[status],
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
        df = df.rename(columns = {'iticount': 'N_ij',
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
        df['iticount'] = df['iticount'].values + df_reversed['iticount'].values

        # From this point on, the procedure equals that of the directional.
        return self.build_vars_directional(df)


    def fetch_df_iticount(self, jhwi=True):
        """
        jhwi: bool. Include jhwi's id.

        Generates the trade dataset to be used in estimation.
        """
        df = self.select_trade_data()
        df = self.build_vars(df)

        # Select columns
        cols = ['cert_i',
                'cert_j',
                'id_i',
                'id_j',
                'N_ij',
                'N_i',
                'N_j',
                's_ij']
        if jhwi == 1:
            cols += ['id_jhwi_i', 'id_jhwi_j']

        df = df[cols]
        df['id'] = df['id_i'].str.cat(df['id_j'])

        return df


    def fetch_df_coordinates(self):
        '''
        Returns the coordinates table with the appropriate id.
        '''
        df_id = self.fetch_df_id()

        coordinates = self.df_coordinates.copy()
        coordinates = coordinates.drop('id', axis=1)
        coordinates = coordinates.merge(df_id,
                                        how='left',
                                        left_on='name',
                                        right_on='city_name'
                                       )
        coordinates = coordinates.loc[ pd.notnull(coordinates['id']) ]

        coordinates = coordinates[['id',
                                   'long_x',
                                   'lat_y',
                                   'cert',
                                   'validity']]

        return coordinates


    def update_id_cstr(self, dtype, id_used = 'id'):
        """
        dtype: str. One of "static" or "dynamic".
        id_used: str. Specifies which id to replace with. Used for testing.

        Updates the ids in the constraints datasets
        """
        df_id = self.fetch_df_id()

        if dtype == "dynamic":
            cstr = self.df_constr_dyn
        elif dtype == "static":
            cstr = self.df_constr_stat
        else:
            raise ValueError("Must be called with dtype equal to one of "
                             + "'static' or 'dynamic'")

        coordinates = self.df_coordinates.drop('id', axis=1)
        id_equiv = coordinates.merge(df_id,
                                     how='left',
                                     left_on='name',
                                     right_on='city_name'
                                    )
        print(id_equiv)

        #sys.exit()
        id_old = df_id['id_old'].values
        id_new = df_id[id_used].values

        variables = ['upper_y', 'lower_y', 'upper_x', 'lower_x']
        cstr[variables] = (cstr[variables].replace(0, np.nan)
                                          .replace(id_old, id_new)
                          )
        #cstr = (cstr[['id_city', 'certainty'] + variables]
        cstr = (cstr
                .rename(columns = {'id_city': 'id',
                                   'upper_y': 'ub_lambda',
                                   'lower_y': 'lb_lambda',
                                   'upper_x': 'ub_varphi',
                                   'lower_x': 'lb_varphi'}
                       )
               )

        return cstr


    def refresh_all():
        fetch_df_id()
        fetch_df_id(False)
        fetch_df_iticount()
        fetch_df_iticount(False)
        fetch_coordinates()
        fetch_coordinates(False)
