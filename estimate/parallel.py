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

## Number of initial values to draw
simulations = 2



# 2. Generate estimation data


proc_data = estimator.gen_data(simulations, 0.2, rank+1)
solution = proc_data.tail(n=1)

if rank > 0:
    comm.send( solution, dest=0 )
    print("Process " + str(rank) + " terminated.")

else:
    # Receive and aggregate
    for process in rank(1, processors):
        solution = solution.append( comm.recv(source=process) )
    solution = solution.reset_index(drop=True)
    print('And the solutions are...')
    print(solution)

    solution.to_csv('/home/delvillar/assyria/fdv/estimate/par_estimation.csv')
