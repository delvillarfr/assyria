#"qsub pythonsubmit.sh" will start the job 
#You will need to add "module load openmpi/1.10.7" to your .profile first

#!/bin/bash
#PBS -N assyria_massive_par_nondir
#PBS -j oe
#PBS -V
#PBS -l procs=3,mem=20gb

cd $PBS_O_WORKDIR


#mprirun will start $procs instances of script.py
#$PBS_NODEFILE tells mpirun which CPU's PBS reseved for the job
mpirun -n 3 -machinefile $PBS_NODEFILE python parallel.py
