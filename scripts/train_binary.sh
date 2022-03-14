#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name=“training”
#SBATCH --output=%j.out

# only use the following if you want email notification
#SBATCH --mail-user=prathams@stanford.edu
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate local_nmt

cd ../
python data/walk-based-re/src/walk_re.py --config data/walk-based-re/configs/ace2005_params_l4.yaml --train --gpu 0
python data/walk-based-re/src/walk_re.py --config data/walk-based-re/configs/ace2005_params_l4.yaml --test --gpu 0
