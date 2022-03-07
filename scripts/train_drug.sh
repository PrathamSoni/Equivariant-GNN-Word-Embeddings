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
python main.py --mode train --model bilstm --pretrain_dataset pubmed --dir n_ary --encoder pubmed --epochs 100 --lr .0001  --dataset drug --batch_size 8
python main.py --mode test --model bilstm --pretrain_dataset pubmed --dir n_ary --encoder pubmed --dataset drug