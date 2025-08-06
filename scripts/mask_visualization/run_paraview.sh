#!/bin/bash
#$ -N paraview_job
#$ -l mfree=135G
#$ -l gpgpu=1
#$ -l cuda=1
#$ -pe serial 2
#$ -j y
#$ -o /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/figures/logs_dir/logs_8_5

# Define paths
PARAVIEW_BIN_DIR="/net/beliveau/vol1/home/msforman/ParaView-6.0.0-RC1-MPI-Linux-Python3.12-x86_64/bin/"
PROJECT_DIR="/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline"
SCRIPTS_DIR="$PROJECT_DIR/scripts/mask_visualization"
OUTPUT_DIR="$PROJECT_DIR/figures/mask_visualization"
SURFACE_DIR='/net/beliveau/vol2/instrument/E9.5_317/Zoom_317/surfaces_compressed_3'
OUTPUT_TAG='_4'

# SURFACE_DIR='/net/beliveau/vol2/instrument/E9.5_290/Zoom_290_subset_test/surfaces_compressed_scaled'
EMBRYONIC_NUMBER=317

# Microscope scaling factor (adjust based on your microscope settings)
export PYTHONUNBUFFERED=1

cd $PARAVIEW_BIN_DIR
mpirun -np $NSLOTS pvbatch $SCRIPTS_DIR/render_segmentations.py --embryonic_number $EMBRYONIC_NUMBER --output_dir $OUTPUT_DIR --surface_dir $SURFACE_DIR --num_frames 25 --output_tag $OUTPUT_TAG