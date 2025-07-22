#!/bin/bash
#$ -N otls_pipeline
#$ -l mfree=128G
#$ -pe serial 1
#$ -j y
#$ -o /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs
#$ -V


conda activate <path_to_conda_environment>
cd <path_to_pipeline_folder>

snakemake --unlock
snakemake \
    --rerun-triggers mtime \
    --cores 1 \
    --latency-wait 60 \
    --use-conda \
    -p 