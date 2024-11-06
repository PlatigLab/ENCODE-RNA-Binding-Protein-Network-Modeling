#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=python_script
#SBATCH --output=./logs/%x.out
#SBATCH --error=./logs/%x.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=standard
#SBATCH --account=platiglab
#SBATCH --signal=USR1@60  # Send SIGUSR1 60 seconds before job ends

export WANDB_DIR=/scratch/$USER/wandb
mkdir -p $WANDB_DIR

export PYTHONUNBUFFERED=1

module load miniforge
: ${VENV_DIR:=/home/$USER/src/platiglab/.venv}
source $VENV_DIR/bin/activate

# Run the Python script
python $*

