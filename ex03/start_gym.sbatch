#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=DrowsyDrivers_job0

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=8G
# the job will need 16GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=1-0
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out


####
#c) Execute your file.
####

start_time=$(date +%s)
singularity exec --nv ~/sdc_gym_amd64.simg bash -x script.sh
end_time=$(date +%s)
total_elapsed=$((end_time - start_time))
job_id=${SLURM_JOB_ID:-"N/A"}
job_name=${SLURM_JOB_NAME:-"N/A"}
job_desc=$(grep -m 1 'outdir:' job.${SLURM_JOB_ID}.out | awk '{print $2}')
job_desc="${job_desc/results\//}"
MESSAGE="SLURM Job Finished!
Job Desc: $job_desc
Job ID  : $job_id
Job Name: $job_name
Total Elapsed Time: $total_elapsed seconds
Last 20 lines of output:"
MESSAGE=$MESSAGE$(tail -n 20 job.${SLURM_JOB_ID}.out)
cd ~/util-scripts
bash send_tg.sh "$MESSAGE"
echo

echo DONE!
