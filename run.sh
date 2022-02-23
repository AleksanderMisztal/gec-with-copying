#!/bin/bash
rm slurm*
export hpcapplication=$1
sbatch ./queue.wilkes3
