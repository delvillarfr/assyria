#"qsub pythonsubmit.sh" will start the job 
#You will need to add "module load openmpi/1.10.7" to your .profile first

#!/bin/bash
#PBS -N hw_multiLogit
#PBS -j oe
#PBS -V
#PBS -l procs=10,mem=30gb

cd $PBS_O_WORKDIR


#mprirun will start $procs instances of script.py
#$PBS_NODEFILE tells mpirun which CPU's PBS reseved for the job
mpirun -n 10 -machinefile $PBS_NODEFILE python parallel.py
