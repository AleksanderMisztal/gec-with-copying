#!/bin/bash
rm slurm*
export hpcapplication=$1
sbatch ./submit-job.sh
