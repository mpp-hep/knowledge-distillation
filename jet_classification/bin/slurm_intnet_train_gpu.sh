#!/bin/sh
#SBATCH --job-name=int_net_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=7-00:00
#SBATCH --output=./logs/intnet_gpu_%j.out

# Folder where the data is located for the training of the AE.
# Change so it suits your configuration.
DATA_FOLDER=/work/deodagiu/ki_data/intnet_input

# Default parameters for running the intnet training.
norm=nonorm
train_events=-1
lr=0.001
batch=100
epochs=100
valid_split=0.3
optimiser=adam
loss=categorical_crossentropy
metrics=categorical_accuracy
outdir=test
summation=false
type=dens
jet_seed=123
seed=127

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
if ${summation}
then
   ./intnet_train --data_folder $DATA_FOLDER --norm ${norm} --train_events ${train_events} --lr ${lr} --batch ${batch} --epochs ${epochs} --valid_split ${valid_split} --optimiser ${optimiser} --loss ${loss} --metrics ${metrics} --outdir ${outdir} --seed ${seed} --summation --intnet_type ${type} --jet_seed ${jet_seed}
else
   ./intnet_train --data_folder $DATA_FOLDER --norm ${norm} --train_events ${train_events} --lr ${lr} --batch ${batch} --epochs ${epochs} --valid_split ${valid_split} --optimiser ${optimiser} --loss ${loss} --metrics ${metrics} --outdir ${outdir} --seed ${seed} --intnet_type ${type} --jet_seed ${jet_seed}
fi
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
mv ./logs/intnet_gpu_${SLURM_JOBID}.out ./trained_intnets/${outdir} 
