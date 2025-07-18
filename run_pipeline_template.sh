#!/bin/bash
#$ -N otls_pipeline
#$ -l mfree=256G
#$ -pe serial 1
#$ -j y
#$ -o /net/beliveau/vol1/project/Nico/OTLS/snakemake.log
#$ -V

conda activate /net/beliveau/vol1/home/longnic/miniconda3/envs/OTLS
cd /net/beliveau/vol1/project/Nico/OTLS/lightsheet-analysis-pipeline

snakemake --unlock
snakemake \
    --rerun-triggers mtime \
    --cores 1 \
    --latency-wait 60 \
    --use-conda \
    -p


### ALWAYS RUN WITH -n FIRST TO SEE WHICH SNAKEMAKE RULES ARE BEING EXECUTED ###