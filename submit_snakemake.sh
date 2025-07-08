#!/bin/bash
#$ -N snakemake_bigstitcher
#$ -l mfree=128G
#$ -pe serial 1
#$ -j y
#$ -o /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs
#$ -V


conda activate /net/beliveau/vol1/home/msforman/miniconda3/envs/snakemake
cd /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline

snakemake --unlock
snakemake \
    --rerun-triggers mtime \
    --rerun-incomplete \
    --cores 1 \
    --latency-wait 60 \
    --use-conda \
    -p 