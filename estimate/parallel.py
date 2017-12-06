# Script for parallel execution of objective function minimization



import pandas as pd
from mpi4py import MPI
import estimate



# Configuration. User input goes here.

## Number of simulations (for one process)
simulations = 5

## Maximum number of iterations before IPOPT stops.
iters = 100

## Estimation type
e_type = 'directional'

## Name of resulting data including full path (in server)
path = '/home/delvillar/assyria/fdv/estimate/par_estimation_dir.csv'



# Initialization

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

estimator = estimate.Estimate(e_type)



# Generate estimation data and aggregate across processes

est_data = estimator.gen_data(len_sim = simulations,
                              perturb = 0.2,
                              rank = rank+1,
                              max_iter = iters,
                              full_vars = True)

if rank > 0:
    comm.send( est_data, dest=0 )
    print("Process " + str(rank) + " terminated.")
else:
    ## Process 0 receives and consolidates
    for process in range(1, size):
        est_data = est_data.append( comm.recv(source=process) )
    est_data = est_data.reset_index(drop=True)

    est_data.to_csv(path)
