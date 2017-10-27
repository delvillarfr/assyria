import pandas as pd
import numpy as np
import pytest
import process

@pytest.fixture
def iticount_data():
    return pd.read_csv(process.root + process.raw_iticount)


def test_iticount_data_has_city_pairs_as_uuid():
    """ Check that city pairs are a unique identifier. """
    test_data = iticount_data()[['anccityid1', 'anccityid2']]
    no_dups = test_data.drop_duplicates()
    pd.testing.assert_frame_equal(test_data, no_dups)


def test_same_filtered_oldid_directional():
    """ oldid taken from line 54 (directional) """
    uuid_jhwi = np.array([3,
                          5,
                          6,
                          7,
                          8,
                          9,
                          10,
                          11,
                          15,
                          16,
                          18,
                          19,
                          20,
                          22,
                          23,
                          24,
                          25,
                          26,
                          29,
                          31,
                          32,
                          33,
                          36,
                          37,
                          38,
                          39])
    uuid_fdvom = process.fetch_id_df()
    np.testing.assert_array_equal(uuid_jhwi, uuid_fdvom['id_old'].values)


def test_same_filtered_oldid_non_directional():
    """ oldid taken from line 54 (directional) """
    uuid_jhwi = np.array([1,
                          3,
                          5,
                          6,
                          7,
                          8,
                          9,
                          10,
                          11,
                          15,
                          16,
                          18,
                          19,
                          20,
                          22,
                          23,
                          24,
                          25,
                          26,
                          29,
                          31,
                          32,
                          33,
                          34,
                          36,
                          37,
                          38,
                          39])
    uuid_fdvom = process.fetch_id_df(False)
    print(uuid_fdvom['id_old'].values)
    np.testing.assert_array_equal(uuid_jhwi, uuid_fdvom['id_old'].values)


def test_merge_gets_only_filtered_data_directional():
    df_merged = process.select_trade_data()
    df_merged = df_merged[['id_old_i', 'id_old_j']]


    df_other = iticount_data()
    filtered_obs = process.filter_positive_imports(df_other)
    df_other = df_other.rename(columns = {'anccityid1': 'id_old_i',
                                          'anccityid2': 'id_old_j'}
                              )
    df_other = df_other[['id_old_i', 'id_old_j']]
    df_other = (df_other.loc[ df_other['id_old_i'].isin(filtered_obs), :]
                        .loc[ df_other['id_old_j'].isin(filtered_obs), :]
                        .sort_values(['id_old_j', 'id_old_i'])
                        .reset_index(drop=True)
               )
    pd.testing.assert_frame_equal(df_merged, df_other)


def test_same_trade_data_directional():
    df_mine = process.fetch_trade_data()
    cols = ['id_jhwi_i',
            'id_jhwi_j',
            'cert_i',
            'cert_j',
            'N_ij',
            'N_j',
            's_ij']
    df_mine = (df_mine[cols].sort_values(['id_jhwi_j', 'id_jhwi_i'])
                           .reset_index(drop=True)
              )
    df_jhwi = pd.read_csv(process.root_jhwi
                          + 'estimation_directional/'
                          + 'ppml_estimation_directional_data.csv'
                         )
    df_jhwi = (df_jhwi.sort_values(['j_id', 'i_id'])
                      .rename(columns = {'i_cert': 'cert_i',
                                         'j_cert': 'cert_j',
                                         'N_j_sum': 'N_j',
                                         'i_id': 'id_jhwi_i',
                                         'j_id': 'id_jhwi_j'}
                             )
                      .reset_index(drop=True)
              )
    df_jhwi = df_jhwi[cols]
    
    pd.testing.assert_frame_equal(df_mine, df_jhwi)


def test_same_trade_data_non_directional():
    df_mine = process.fetch_trade_data(directional=False)
    cols = ['id_jhwi_i',
            'id_jhwi_j',
            'cert_i',
            'cert_j',
            'N_ij',
            'N_j',
            's_ij']
    df_mine = (df_mine[cols].sort_values(['id_jhwi_j', 'id_jhwi_i'])
                           .reset_index(drop=True)
              )
    df_jhwi = pd.read_csv(process.root_jhwi
                          + 'estimation_nondirectional/'
                          + 'ppml_estimation_nondirectional_data.csv'
                         )
    df_jhwi = (df_jhwi.sort_values(['j_id', 'i_id'])
                      .rename(columns = {'i_cert': 'cert_i',
                                         'j_cert': 'cert_j',
                                         'N_j_sum': 'N_j',
                                         'i_id': 'id_jhwi_i',
                                         'j_id': 'id_jhwi_j'}
                             )
                      .reset_index(drop=True)
              )
    df_jhwi = df_jhwi[cols]
    
    pd.testing.assert_frame_equal(df_mine, df_jhwi)


def test_that_coordinates_match():
    pass


#def jhwi_for_constraints(direct, dyn):
#    df_jhwi = pd.read_csv(process.root_jhwi
#                          + 'estimation_'
#                          + direct + '/'
#                          + 'constraints_'
#                          + dyn
#                          + '.csv'
#                         )
#    #df_jhwi = df_jhwi.drop('name', axis=1)
#    return df_jhwi
#
#
#def test_same_constraints_directional_dynamic():
#    df_jhwi = jhwi_for_constraints('directional', 'dynamic')
#    df_mine = process.update_id_cstr(True, True, id_used = 'id_jhwi')
#
#    df_mine = df_mine.drop('certainty', axis=1)
#
#    print(df_mine)
#    print(df_jhwi)
#
#    pd.testing.assert_frame_equal(df_mine, df_jhwi)
#
#
#def test_same_constraints_directional_static():
#    df_jhwi = jhwi_for_constraints('directional', 'static')
#    df_mine = process.update_id_cstr(True, False, id_used = 'id_jhwi')
#
#    df_mine = df_mine.drop('certainty', axis=1)
#
#    print(df_mine)
#    print(df_jhwi)
#
#    pd.testing.assert_frame_equal(df_mine, df_jhwi)
