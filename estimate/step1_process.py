import pandas as pd
import numpy as np
import sys

import estimate

pd.options.display.max_rows = 999




def iticounts_of_known_cities(e_instance):
    """ Gets the subset of iticount data of known cities.
    """
    df = e_instance.df_iticount.copy()
    coords = e_instance.df_coordinates.loc[e_instance.df_coordinates['cert'] == 3]

    # Add distances
    df['distances'] = e_instance.fetch_dist(coords['lat_y'].values,
                                            coords['long_x'].values)

    # Select known cities
    df = df[['id_i', 'id_j', 'N_ij', 'distances']].loc[
            (df['cert_i'] < 3) & (df['cert_j'] < 3)
            ]

    return df




def fetch_sij(iticounts):
    """ Get trade shares data from trade counts.
    """
    # Add N_i, N_j
    for status in ['i', 'j']:
        N = (iticounts.groupby('id_'+status)
                     .sum()['N_ij']
                     .rename('N_'+status)
             )

        # Add this info
        iticounts = iticounts.join(N, on='id_'+status)

    # Add s_ij
    iticounts[ 's_ij' ] = iticounts['N_ij'] / iticounts['N_j']

    return iticounts




def find_isolated_cities(iticounts, i_cities = []):
    """ Gets trade shares such that all cities import somesing.

    Args:
        iticounts (pd.DataFrame): The iticounts data prior to computing trade
            shares.
        i_cities (list): The list of isolated cities I have to remove.

    Returns:
        list: The isolated cities. All other cities form a trade network where
            each city imports something from some other city.
    """
    # Drop the cities given in i_cities
    iticounts_new = iticounts.copy()
    for i_city in i_cities:
        iticounts_new = iticounts_new.loc[(iticounts_new['id_i'] != i_city)
                                          & (iticounts_new['id_j'] != i_city)]
    iticounts_new = fetch_sij(iticounts_new)

    # Test: are there cities for which s_ij is nan?
    na_indices = pd.isna(iticounts_new['s_ij'])
    if na_indices.sum() == 0:
        return i_cities
    else:
        # Get cities
        isolated = iticounts_new.loc[na_indices, 'id_j'].unique()
        i_cities += isolated.tolist()
        return find_isolated_cities(iticounts, i_cities)




def wrap(e_instance):
    """ Fetch the step1 data.
    """
    iticounts = iticounts_of_known_cities(e_instance)
    i_cities = find_isolated_cities(iticounts, i_cities = [])
    print(i_cities)

    for i_city in i_cities:
        iticounts = iticounts.loc[(iticounts['id_i'] != i_city)
                                  & (iticounts['id_j'] != i_city)]
    iticounts = fetch_sij(iticounts)

    # Headers
    iticounts = iticounts[['id_i', 'id_j', 'N_j', 'N_ij', 's_ij', 'distances']]

    # Get number of cities
    n_cities = len(iticounts['id_i'].unique())
    ## Check this is true: n_cities * (n_cities - 1) = len(iticounts)
    assert n_cities == np.sqrt(len(iticounts) + 0.25) + 0.5

    # Add sending and receiving dummies
    origin_dummies = pd.get_dummies(iticounts['id_i'])
    origin_dummies.columns = ['origin_dummy' + str(i) for i in range(1, n_cities + 1)]
    destination_dummies = pd.get_dummies(iticounts['id_j'],
                                         prefix='destination_dummy')
    destination_dummies.columns = ['destination_dummy' + str(i) for i in range(1, n_cities + 1)]

    iticounts = pd.concat([iticounts, origin_dummies, destination_dummies], axis=1)

    return iticounts





# Generate step1 data for all variations to run




# 1. Without dropping Qattara

# 1.1. Neither Mamma nor Hahhum known

e = estimate.EstimateAncient('directional')
df = wrap(e)
df.to_csv('./results/ancient/twostep/noneDrop/base/step1/step1_processed.csv')


# 1.2. Mamma known, Hahhum unknown

e = estimate.EstimateAncient('directional', cities_to_known = ['ma02'])
df = wrap(e)
df.to_csv('./results/ancient/twostep/noneDrop/ma02Known/step1/step1_processed.csv')


# 1.3. Mamma known, Hahhum known

e = estimate.EstimateAncient('directional', cities_to_known = ['ma02', 'ha01'])
df = wrap(e)
df.to_csv('./results/ancient/twostep/noneDrop/ma02ha01Known/step1/step1_processed.csv')


# 1.4. Mamma unknown, Hahhum known

e = estimate.EstimateAncient('directional', cities_to_known = ['ha01'])
df = wrap(e)
df.to_csv('./results/ancient/twostep/noneDrop/ha01Known/step1/step1_processed.csv')




# 2. Dropping Qattara


# 2.1. Neither Mamma nor Hahhum known

e = estimate.EstimateAncient('directional', cities_to_drop=['qa01'])
df = wrap(e)
df.to_csv('./results/ancient/twostep/qa01Drop/base/step1/step1_processed.csv')


# 2.2. Mamma known, Hahhum unknown

e = estimate.EstimateAncient('directional',
                             cities_to_drop = ['qa01'],
                             cities_to_known = ['ma02'])
df = wrap(e)
df.to_csv('./results/ancient/twostep/qa01Drop/ma02Known/step1/step1_processed.csv')


# 2.3. Mamma known, Hahhum known

e = estimate.EstimateAncient('directional',
                             cities_to_drop = ['qa01'],
                             cities_to_known = ['ma02', 'ha01'])
df = wrap(e)
df.to_csv('./results/ancient/twostep/qa01Drop/ma02ha01Known/step1/step1_processed.csv')


# 2.4. Mamma unknown, Hahhum known

e = estimate.EstimateAncient('directional',
                             cities_to_drop = ['qa01'],
                             cities_to_known = ['ha01'])
df = wrap(e)
df.to_csv('./results/ancient/twostep/qa01Drop/ha01Known/step1/step1_processed.csv')



# 2. Drop every known city, one at a time.
## Known cities, except Hahhum
cities = ["Hattus",
          "Kanes",
          "Karahna",
          "Tapaggas",
          "Hanaknak",
          "Hurama",
          "Malitta",
          "Salatuwar",
          "Samuha",
          "Timelkiya",
          "Ulama",
          "Unipsum",
          "Wahsusana",
          "Zimishuna"]
city_ids = pd.read_csv('../process/ancient/id.csv')

for city in cities:
    city_id = city_ids.loc[ city_ids['city_name'] == city, 'id'].values[0]
    print(city_id)
    e = estimate.EstimateAncient('directional', cities_to_drop=[city_id])
    df = wrap(e)
    target = './results/ancient/twostep/' + city_id + 'Drop/step1/step1_processed.csv'
    print(target)
    df.to_csv(target)




# TESTS
#e = estimate.EstimateAncient('directional', cities_to_known = ['ma02'])
#
#df = wrap(e)
#df.to_csv('tmp.csv')
#
#df_other = pd.read_csv('./results/ancient/twostep/test/original/step1/step1_processed.csv')
#df_other = df_other.drop('Unnamed: 0', axis = 1)
#
#np.testing.assert_almost_equal(df.values[:, 2:], df_other.values[:, 2:])
