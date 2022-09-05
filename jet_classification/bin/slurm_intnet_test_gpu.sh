#!/bin/sh
#SBATCH --job-name=int_net_test
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00-00:15
#SBATCH --output=./logs/intnet_test_gpu_%j.out

# Folder where the data is located for the training of the AE.
# Change so it suits your configuration.
DATA_FOLDER=/work/deodagiu/ki_data/intnet_input

# Default parameters for running the intnet training.
norm=nonorm
test_events=-1
model_dir=./trained_intnets/test
seed=321

# Gather parameters given by user.
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Set up conda environment and cuda.
source /work/deodagiu/miniconda/bin/activate ki_intnets

# Run the script with print flushing instantaneous.
export PYTHONUNBUFFERED=TRUE
./intnet_test --data_folder $DATA_FOLDER --norm ${norm} --test_events ${test_events} --model_dir ${model_dir} --seed ${seed}
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
mv ./logs/intnet_test_gpu_${SLURM_JOBID}.out ${model_dir}
