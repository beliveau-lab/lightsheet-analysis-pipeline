# This script should produce a few summary statistics about the Snakemake workflow, and the data. 
# 1. .pdf showing a few key outputs from the .csv (or anndata):
#   a. Table with summary statistics for:
#       - Number of cells
#       - Cells called as GFP+ / RFP+ / Both / Neg
#       - Distribution of cell morphological/intensity features (ridgeplot/violin?)
#       
# 2. Spatial map showing distribution of features (scatter plot?)
#   
# 3. Spherical harmonics shape modes (PCA)
#
# 4. PCA / UMAP on all cell morpho features, colored by intensity / class
#
# 5. some random cell images and their S.H reconstructions
#


import numpy as np
import matplotlib.pyplot as plt 