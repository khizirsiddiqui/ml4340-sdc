#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=DrowsyDriver_job0

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=4G
# the job will need 16GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=10:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-user=khizirsiddiqui@gmail.com
#SBATCH --mail-type=ALL


####
#c) Execute your file.
####


start_time=$(date +%s)
singularity exec --nv ~/sdc_gym_amd64.simg bash script.sh
t3=$(date +%s)
total_elapsed=$(( t3 - start_time ))
echo ">> Sending notification to tg.."
cd ~/util-scripts && bash send_tg.sh "Training and evaluation completed in $total_elapsed seconds."
echo "== Notification sent."

echo DONE!
