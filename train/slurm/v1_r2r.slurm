#!/usr/bin/env bash
#SBATCH --job-name=stagevln-v1-r2r
#SBATCH --account=yshang
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/home/an221229/code/StageVLN-v2/slurm_logs/%x-%j.out
#SBATCH --error=/home/an221229/code/StageVLN-v2/slurm_logs/%x-%j.err
set -euo pipefail
bash /home/an221229/code/StageVLN-v2/train/slurm/run_r2r.sh v1
