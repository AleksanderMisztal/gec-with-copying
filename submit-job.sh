#!/bin/bash

#SBATCH -J apm81-gec
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

source venv/bin/activate

JOBID=$SLURM_JOB_ID
CMD="python3 $hpcapplication >./logs/log$JOBID.txt 2>&1"

echo -e "Running from `pwd`.\n"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo -e "\nExecuting command:\n$CMD\n"
eval $CMD
