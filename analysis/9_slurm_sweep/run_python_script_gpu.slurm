#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --job-name=python_script
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --account=platiglab
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@60  # Send SIGUSR1 60 seconds before job ends

#. env.sh
export WANDB_DIR=/scratch/$USER/wandb
mkdir -p $WANDB_DIR

export PYTHONUNBUFFERED=1

# Load necessary modules
module load anaconda

# Activate the venv
: ${VENV_DIR:=/sfs/qumulo/qhome/kqt4tu/src/platiglab/.venv}
source $VENV_DIR/bin/activate

# Run the Python script
python $*

