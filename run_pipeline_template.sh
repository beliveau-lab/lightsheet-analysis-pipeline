#!/bin/bash
#$ -N otls_pipeline
#$ -l mfree=256G
#$ -pe serial 1
#$ -j y
#$ -o snakemake.log
#$ -V

set -euo pipefail

PROJECT_ROOT="/net/beliveau/vol1/project/VB_Segmentation/subprojects/lightsheet-analysis-pipeline"
cd "$PROJECT_ROOT"

conda activate /net/beliveau/vol1/project/bigly_conda/miniconda3/envs/otls-pipeline

# Optional: unlock if a previous run was interrupted
snakemake --unlock --profile profiles/sge || true

# Run with profile; pass-through args are supported
snakemake -p --profile profiles/sge "$@"

# Example usages:
#   ./run_pipeline.sh -n                    # dry run
#   ./run_pipeline.sh --until rechunk_to_blocks
#   ./run_pipeline.sh --configfile config.yaml