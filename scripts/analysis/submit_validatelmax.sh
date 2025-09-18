#!/bin/bash
#$ -N lmax
#$ -l mfree=32G
#$ -pe serial 1
#$ -j y
#$ -o /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/analysis/choose_lmax_log

conda activate /net/beliveau/vol1/home/msforman/miniconda3/envs/otls-pipeline

cd /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/scripts

python3 -m analysis.validate_lmax