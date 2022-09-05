#!/bin/sh
#SBATCH --job-name=student_net_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=24:00:00
#SBATCH --output=./logs/student_gpu_%j.out

# Folder where the data is located for the training of the AE.
# Change so it suits your configuration.
DATA_FOLDER=/work/deodagiu/ki_data/intnet_input

# Default parameters for running the student knowledge distillation training.
norm=nonorm
train_events=-1
lr=0.001
batch=128
epochs=100
optimizer=adam
student_loss=softmax_with_crossentropy
distill_loss=kl_divergence
metrics=categorical_accuracy
teacher=./trained_intnets/intnet_16const_robust
student_type=unistudent
alpha=0.1
temperature=10
outdir=test
seed=123

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
./student_train --data_folder $DATA_FOLDER --norm ${norm} --train_events ${train_events} --lr ${lr} --batch ${batch} --epochs ${epochs} --optimizer ${optimizer} --student_loss ${student_loss} --distill_loss ${distill_loss} --metrics ${metrics} --outdir ${outdir} --seed ${seed} --teacher ${teacher} --student_type ${student_type} --alpha ${alpha} --temperature ${temperature}
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
mv ./logs/student_gpu_${SLURM_JOBID}.out ./trained_students/${outdir}
