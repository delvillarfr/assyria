# Parallel implementation of exercise 2

import pandas as pd
from mpi4py import MPI
import estimate



# 1. Initialize


## Get MPI-related info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Instantiate Estimate
estimator = estimate.Estimate('directional')

## Number of initial values to draw (goal: 20000 simulations total)
simulations = 334



# 2. Generate estimation data


solution = estimator.gen_data(simulations, 0.2, rank+1)

if rank > 0:
    comm.send( solution, dest=0 )
    print("Process " + str(rank) + " terminated.")
else:
    # Receive and aggregate
    for process in range(1, size):
        solution = solution.append( comm.recv(source=process) )
    solution = solution.reset_index(drop=True)

    solution.to_csv('/home/delvillar/assyria/fdv/estimate/par_estimation_dir.csv')
