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
# -m ea

# Give job a name
#$ -N minigrid_p2

# Combine output and error files into a single file
#$ -j y

# Specify the output file name
#$ -o minigrid_p2.qlog

# Request 8 CPUs
#$ -pe omp 8

# Request 1 GPU
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=3.7

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

# Use the SGE_TASK_ID environment variable to select the appropriate input file from bash array
config_file=${SGE_TASK_ID}.yaml

module load python3/3.7.7
module load tensorflow/2.1.0
module load pytorch/1.6.0

source /projectnb/saenkog/juliusf/cfrl-rllib/venv/bin/activate

python train.py --ray-num-gpus 1 --ray-num-cpus 8 -f config/minigrid_phase_2/${config_file}
