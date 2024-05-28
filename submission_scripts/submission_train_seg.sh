#!/bin/env bash
#SBATCH --partition=shared-gpu
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --output=kraken-%j.out
#SBATCH --mem=24GB
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:24GB

module load CUDA/11.8.0 GCCcore/11.2.0 Python/3.9.6
source ~/kraken-env/bin/activate

OUTPUT_NAME="/home/users/c/cannavac/BnF_fr19152/models/trained_segmentation_model"
XML_FOLDER="/home/users/c/cannavac/BnF_fr19152/data"

echo "KETOS training"
srun ketos segtrain -o $OUTPUT_NAME \
	-f alto -d cuda:0 --resize new \
	--epochs 50 --schedule cosine \
	-i "/home/users/c/cannavac/BnF_fr19152/models/blla.mlmodel" \
	"${XML_FOLDER}/*.xml"
