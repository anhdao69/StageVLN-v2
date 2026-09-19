#!/usr/bin/env bash
#SBATCH --job-name=stagevln-v0-r2r
#SBATCH --account=yshang
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=28:00:00
#SBATCH --output=/home/an221229/code/StageVLN-v2/slurm_logs/%x-%j.out
#SBATCH --error=/home/an221229/code/StageVLN-v2/slurm_logs/%x-%j.err
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
bash "$PROJECT_ROOT/train/slurm/run_r2r.sh" v0
