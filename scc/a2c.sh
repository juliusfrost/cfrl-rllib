#!/bin/bash -l

# Set SCC project
#$ -P saenkog

# Submit an array job with 3 tasks
# -t 1-3

# Specify hard time limit for the job.
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=48:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m bea

# Give job a name
#$ -N train_a2c

# Combine output and error files into a single file
#$ -j y

# Specify the output file name
#$ -o train_a2c.qlog

# Request 8 CPUs
#$ -pe omp 8

# Request 1 GPU (the number of GPUs needed should be divided by the number of CPUs requested above)
#$ -l gpus=0.125

# Specify the minimum GPU compute capability
#$ -l gpu_c=3.5

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

# Use the SGE_TASK_ID environment variable to select the appropriate input file from bash array
# Bash array index starts from 0, so we need to subtract one from SGE_TASK_ID value
environments=(BreakoutNoFrameskip-v4 PongNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 MsPacmanNoFrameskip-v4 AsteroidsNoFrameskip-v4)
env_index=$(($SGE_TASK_ID-1))
env=${environments[$env_index]}

module load cuda/10.1
module load python3/3.6.9
module load pytorch/1.3
module load tensorflow/2.0.0

source /projectnb/saenkog/juliusf/envs/ray/bin/activate

python ../train.py --env ${env} --algo a2c
