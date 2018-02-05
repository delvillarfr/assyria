import estimate
import pandas as pd
import numpy as np

pd.options.display.max_columns = 999

def append_ancient_runs():
    e = estimate.EstimateAncient('directional')
    path = './estim_results/ancient/'

    runs = ['par_estimation_dir_01022018.csv',
            'par_estimation_dir_01032018_2.csv',
            'par_estimation_dir_01112018_2.csv',
            'par_estimation_dir_11202017.csv',
            'par_estimation_dir_rigid_12262017.csv',
            'par_estimation_dir_rigid_12192017.csv']
    dfs = []
    for run in runs:
        dfs.append(pd.read_csv(path + run))

    df = pd.concat(dfs)
    df = df.drop('Unnamed: 0', axis=1)

    # Drop invalid arguments
    df = (df.loc[df['status'] != -13]
            .sort_values('obj_val')
            .reset_index()
            )
    return df


def run_ancient():
    e = estimate.EstimateAncient('directional')
    path = './estim_results/ancient/'

    sims = append_ancient_runs()
    sims = sims.drop('index', axis=1)

    # Note there are 49 variables
    varlist = np.array(sims.iloc[0, :49].tolist())
    v = e.short_to_jhwi(varlist)

    e.export_results(v,
                     zeta_fixed = True,
                     loc = path)


def run_todo1_z2():
    e = estimate.EstimateAncient('directional')

    path = './estim_results/ancient/todo1_z2/'

    sims = pd.read_csv(path + 'todo1_z2_appended.csv')
    sims = sims.sort_values('obj_val')

    # Note there are 49 variables
    varlist = np.array(sims.iloc[0, :49].tolist())
    v = e.short_to_jhwi(varlist)

    e.export_results(v,
                     zeta_fixed = True,
                     loc = path)
