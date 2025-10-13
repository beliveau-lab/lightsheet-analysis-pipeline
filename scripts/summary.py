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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


### LOAD DATAFRAME ###

print("Loading dataframe...")

out_dir = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/'

df = pd.read_csv(out_dir + "dataset_fused_features_rawdata.csv")

print("Dataframe loaded. Shape: ", df.shape)


### SUMMARY STATISTICS ###

summary_stats = df.describe()
summary_stats.to_csv(out_dir + 'figures/summary_statistics.csv')


### PLOT SPATIAL DISTRIBUTION OF INTENSITY FEATURES ###

x = df.iloc[:, 0:10].sample(frac=0.01, random_state=1)  # Sample 10% of data for plotting

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
plt.rcParams.update({'font.size': 16})
centroid_dims = ['centroid_z', 'centroid_y', 'centroid_x']
channels = ['mean_intensity_ch0', 'mean_intensity_ch1', 'mean_intensity_ch2']
channel_labels = ['GFP', 'RFP', 'YoPro-1']
colors = ['green', 'magenta', 'cyan']

for i, dim in enumerate(centroid_dims):
    for j, (ch, label, color) in enumerate(zip(channels, channel_labels, colors)):
        ax = axes[i, j]
        ax.scatter(x[dim], x[ch], label=label, alpha=1, s=0.1)
        ax.set_title(f'{dim} vs {label}')
        ax.set_xlabel(dim)
        ax.set_ylabel('Mean Intensity')
        ax.set_ylim(0, np.percentile(x[ch], 99))  # Limit y-axis to 99th percentile for better visualization
plt.tight_layout()
plt.savefig(out_dir + 'figures/spatial_intensity_distribution.svg', format='svg', dpi=300)
