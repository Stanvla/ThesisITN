#!/bin/bash
#SBATCH -J vs-audio-feat
#SBATCH -o vs-audio-feat.out				  # name of output file for this submission script
#SBATCH -e vs-audio-feat.err				  # name of error file for this submission script
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --constraint="gpuram16G|gpuram24G"
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

echo $HOSTNAME
source /lnet/express/work/people/stankov/python-venvs/2023-python3.8.14/bin/activate
python /lnet/express/work/people/stankov/alignment/Thesis/punc_restoration/create_audio_features.py
