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


def filter_positive_imports(iticount):
    """
    iticount: DataFrame. It is the raw commerce data.
    
    Keeps rows in iticount data where exporting (anccityid1) and exporting
    (anccityid2) cities import at least once.

    Returns 1-dim np.array with id's of participating cities.
    """
    # Get N_j
    N_j = (iticount.groupby('anccityid2')
                   .sum()['iticount']
                   .rename('N_j')
          )

    # Add this info keep obs with positive aggregate imports
    iticount = iticount.join(N_j, on='anccityid2')
    iticount = iticount.loc[ iticount['N_j'] > 0 ]
    
    return iticount['anccityid2'].unique()


def filter_positive_activity(iticount):
    """
    iticount: DataFrame. It is the raw commerce data.
    
    Keeps rows in iticount data where exporting (anccityid1) and exporting
    (anccityid2) cities import or export at least once.

    Returns 1-dim np.array with id's of participating cities.
    """
    # Add N_ij and N_ji for all i, j.
    iticount = iticount.sort_values(['anccityid2', 'anccityid1'])
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


def create_id(cities, coordinates, ids_left):
    """
    cities: DataFrame. Its only column is 'city_name'.
    coordinates: DataFrame. I only use it to recover Jhwi's id.
    ids_left: list-like. It is the id of cities to consider.

    Generates ID conversion table for all cities in ids_left.

    The id to use will be the first two letters of the city followed by their
    place in the list. e.g. Kanes is the first of two cities starting with
    "Ka". Its id is "ka01".
    """
    # Create old id
    cities['id_old'] = cities.index + 1

    # Keep only filtered cities
    cities = cities.loc[ cities['id_old'].isin(ids_left) ]

    # Get Joonhwi's id (it's not alphabetically ordered...)
    coordinates = coordinates.merge(cities,
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


def fetch_id_df(directional = True):
    """
    Uses a list-selection function and a create_id() to generate a table with
    all id's for reference.
    """
    city_name = pd.read_csv(root + raw_city_name, header=None)

    city_name = city_name.rename(columns = {0: 'city_name'})
    coordinates = pd.read_csv(root + raw_coord)

    if directional == True:
        ids_left = filter_positive_imports(pd.read_csv(root + raw_iticount))
    else:
        ids_left = filter_positive_activity(pd.read_csv(root + raw_iticount))

    result = create_id(city_name, coordinates, ids_left)

    # Save
    if directional == True:
        result.to_csv(root + process + 'id_equivalence_dir.csv',
                      index=False)
    else:
        result.to_csv(root + process + 'id_equivalence_nondir.csv',
                      index=False)

    return result


def select_trade_data(directional = True):
    """ Selects trade data according to the filtering function """

    iticount = pd.read_csv(root + raw_iticount)

    if directional == True:
        ids_left = filter_positive_imports(iticount)
        id_df = fetch_id_df()
    else:
        ids_left = filter_positive_activity(iticount)
        id_df = fetch_id_df(False)

    id_df = id_df[['id_old', 'id_jhwi', 'id']]
    
    d = {'i': '1', 'j': '2'}
    for status in d.keys():
        iticount = iticount.merge(id_df,
                                  how='right',
                                  left_on='anccityid'+d[status],
                                  right_on='id_old')
        iticount = iticount.rename(columns = {'id_old': 'id_old_'+status,
                                              'id_jhwi': 'id_jhwi_'+status,
                                              'id': 'id_'+status}
                                  )

    iticount = (iticount.sort_values(['id_old_j', 'id_old_i'])
                        .reset_index(drop=True)
               )

    return iticount


def build_vars_directional(df):
    """
    df: DataFrame. It is the filtered DataFrame.

    Returns the DataFrame with all variables used in the directional analysis.
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

        # Add this info keep obs with positive aggregate imports
        df = df.join(N, on='id_'+status)

    # Add s_ij ( adding s_ji is overkill: the data level is (i, j) )
    df[ 's_ij' ] = df['N_ij'] / df['N_j']

    return df


def build_vars_nondirectional(df):
    """
    df: DataFrame. It is the filtered DataFrame.

    Returns the DataFrame with all variables used in the non-directional
    analysis. Notice the database level is unordered city pairs.
    """
    # Add N_ij and N_ji for all i, j.
    df = df.sort_values(['id_j', 'id_i'])
    df_reversed = df.sort_values(['id_i', 'id_j'])
    df['iticount'] = df['iticount'].values + df_reversed['iticount'].values

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

        # Add this info keep obs with positive aggregate imports
        df = df.join(N, on='id_'+status)

    # Add s_ij ( adding s_ji is overkill: the data level is (i, j) )
    df[ 's_ij' ] = df['N_ij'] / df['N_j']

    return df


def fetch_trade_data(directional = True, jhwi=True):
    """
    directional: bool. Specifies which data to fetch.
    jhwi: bool. Include that id.

    Generates the trade dataset to be used in estimation.
    """
    if directional == True:
        df = select_trade_data()
        df = build_vars_directional(df)
    else:
        df = select_trade_data(directional = False)
        df = build_vars_nondirectional(df)

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

    # Save
    if directional == True:
        df.to_csv(root + process + 'main_dir.csv', index=False)
    else:
        df.to_csv(root + process + 'main_nondir.csv', index=False)

    return df


def fetch_coordinates(directional = True):
    # get id table, merge with coordinates on name, select vars
    
    if directional == True:
        id_df = fetch_id_df(filter_positive_imports)
    else:
        id_df = fetch_id_df(filter_positive_activity)

    coordinates = pd.read_csv(root + raw_coord)
    coordinates = coordinates.drop('id', axis=1)
    coordinates = coordinates.merge(id_df,
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

    # Save
    if directional == True:
        coordinates.to_csv(root + process + 'coordinates_dir.csv',
                           index = False)
    else:
        coordinates.to_csv(root + process + 'coordinates_nondir.csv',
                           index = False)

    return coordinates


def update_id_cstr(directional = True, dynamic = True, id_used = 'id'):
    """
    dynamic: bool. One of "static" or "dynamic"
    id_used: str. Specifies which id to replace with. Used for testing.

    Updates the ids in the constraints datasets
    """
    # Fetch data equivalence table and constraints
    if directional == True:
        id_df = fetch_id_df(filter_positive_imports)
    else:
        id_df = fetch_id_df(filter_positive_activity)
    if dynamic == True:
        cstr = pd.read_csv(root + raw_constr_dynamic)
    else:
        cstr = pd.read_csv(root + raw_constr_static)

    coordinates = pd.read_csv(root + raw_coord)
    coordinates = coordinates.drop('id', axis=1)
    id_equiv = coordinates.merge(id_df,
                                 how='left',
                                 left_on='name',
                                 right_on='city_name'
                                )
    print(id_equiv)

    #sys.exit()
    id_old = id_df['id_old'].values
    id_new = id_df[id_used].values

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
    direct = {True: '_dir', False: '_nondir'}
    dyn = {True: '_dyn', False: '_stat'}
    cstr.to_csv(root
                + process
                + 'constraints'
                + direct[directional]
                + dyn[dynamic]
                + '.csv',
                index = False
               )
    
    return cstr


a = update_id_cstr(dynamic = False, id_used = 'id_jhwi')





def refresh_all():
    fetch_id_df()
    fetch_id_df(False)
    fetch_trade_data()
    fetch_trade_data(False)
    fetch_coordinates()
    fetch_coordinates(False)
