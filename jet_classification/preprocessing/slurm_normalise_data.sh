#!/bin/sh
#SBATCH --job-name=normalise_data
#SBATCH --account=t3
#SBATCH --mem=64000M
#SBATCH --time=0-01:00
#SBATCH --output=./logs/normalise_data_%j.out

# Folder where the data is located for the training of the AE.
# Change so it suits your configuration.
DATA_FOLDER=/work/deodagiu/ki_data/

# Default parameters for running the intnet training.
train_data=x_jet_images_c150_pt2.0_jedinet_train.npy
test_data=x_jet_images_c150_pt2.0_jedinet_test.npy
train_target=y_jet_images_c150_pt2.0_jedinet_train.npy
test_target=y_jet_images_c150_pt2.0_jedinet_test.npy
output_dir=intnet_input
norm=nonorm
test_split=0.33

# Gather parameters given by user.
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Set up conda environment and cuda.
source /work/deodagiu/miniconda/bin/activate ki_preprocessing

# Run the script with print flushing instantaneous.
export PYTHONUNBUFFERED=TRUE
python normalise_mlready_data.py --x_data_path_train ${DATA_FOLDER}/${train_data} --x_data_path_test ${DATA_FOLDER}/${test_data} --y_data_path_train ${DATA_FOLDER}/${train_target} --y_data_path_test ${DATA_FOLDER}/${test_target} --norm ${norm} --test_split ${test_split} --output_dir ${DATA_FOLDER}/${output_dir}
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
mv ./logs/intnet_gpu_${SLURM_JOBID}.out ./trained_intnets/${outdir}
