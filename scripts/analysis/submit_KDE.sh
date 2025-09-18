#!/bin/bash
#$ -N KDE
#$ -l mfree=64G
#$ -pe serial 1
#$ -j y
#$ -o /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/analysis/KDE_estimation

PROJECT_DIR="/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline" # where the script is located

FIGURE_DIR="/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/KDE_figures_skew/" # where the figures will be saved

CSV_PATH='/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_features_skew_alignment.csv' # where the og csv is located

OUTLIER_PATH='/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/outliers_skew.csv' # where a csv of outliers will be saved

DENSITY_THRESH=1e-10

N_SUBSAMPLE=10000 # can test with 1000 for a quick run

# Activate conda environment
conda activate /net/beliveau/vol1/home/msforman/miniconda3/envs/otls-pipeline

# Change to script directory
cd /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/scripts

python -m analysis.find_outliers \
    --csv_path "$CSV_PATH" \
    --figure_dir "$FIGURE_DIR" \
    --outlier_path "$OUTLIER_PATH" \
    --density_thresh $DENSITY_THRESH \
    --n_subsample $N_SUBSAMPLE \
    --n_rounds 5