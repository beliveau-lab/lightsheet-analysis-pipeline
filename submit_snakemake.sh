#!/bin/bash
#$ -N snakemake_bigstitcher
#$ -l mfree=128G
#$ -pe serial 1
#$ -j y
#$ -o /net/beliveau/vol1/project/VB_Segmentation/subprojects/OTLS-Analyzer/logs/snakemake.log
#$ -V
#$ -m ea


conda activate /net/beliveau/vol1/home/vbrow/miniconda3/envs/snakemake-spark
cd /net/beliveau/vol1/project/VB_Segmentation/subprojects/OTLS-Analyzer/

snakemake --unlock

snakemake \
    --latency-wait 60 \
    --rerun-incomplete \
    --use-conda \
    --rerun-triggers mtime \
    -p