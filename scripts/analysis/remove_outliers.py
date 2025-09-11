"""Module for detecting and removing outliers from cell data using KDE."""
import argparse
import logging
import time
from pathlib import Path


import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.utils import resample

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
sns.set_palette(cc.glasbey_dark)
plt.rcParams['figure.facecolor'] = 'white'

class KDEOutlierDetection:
    def __init__(self, params):
        self.figure_dir = params["figure_dir"] # dir to save final figures
        self.density_thresh = params["density_thresh"] # values below this threshold considered 'low' density (outliers)
        self.csv_path = params['csv_path'] # path to original csv
        self.outlier_path = params['outlier_path']
        self.n_subsample = params['n_subsample']
        self.n_rounds = params['n_rounds']
        self.df = None # df from csv
        self.normalize_factor = 1000
        self.normal_color = cc.glasbey_dark[6]
        self.outlier_color = cc.glasbey_dark[0]
        self.init_data()

    def clean_data(self, str):
        """
        Clean the data by converting the value to a float.
        """
        try:
            return float(str.strip('[]'))
        except ValueError:
            return float(str)

    def init_data(self):
        """ 
        Initialize the KDEOutlierDetection data.
        
        Returns:
        -------
        df: pd.DataFrame
            DataFrame containing the data from the csv file, only reads the columns specified in the metrics list
        """
        # Note: if u want to run this with more metrics please add them to the list 
        metrics = ['area',
                   'label',
                   'major_magnitude', 
                   'intermediate_magnitude', 
                   'minor_magnitude']
        self.df = pd.read_csv(
            self.csv_path, 
            usecols=metrics, 
            index_col='label',
            converters={'label': self.clean_data, 'area':self.clean_data},
            dtype={'major_magnitude': 'float32',
                   'intermediate_magnitude': 'float32',
                   'minor_magnitude': 'float32'})
        logger.info(f"Using metrics: {self.df.columns}. {len(self.df)} objects loaded")
        return 

    def permute_pairs(self):
        """
        Permute the metrics to create all possible pairs.
        
        Returns:
        --------
        pairs: np.array
            Array of all possible pairs of metrics
        """
        metrics = self.df.columns
        feature_names = list(metrics)
        pairs = np.array([[feature_names[f1], feature_names[f2]] for f1 in range(len(feature_names)) 
                        for f2 in range(f1 + 1, len(feature_names))])
        logger.info(f"{len(pairs)} feature pairs")
        return pairs

    def detect_outliers(self, 
                        rounds=5, # not the same as a bootstrap, rounds repeated due to probabilistic nature of density
                        n_subsample=10000):
        """
        Detect outliers using KDE.

        Parameters:
        -----------
        rounds : int
            Number of rounds to run KDE
        normalize_factor : int
            Factor to normalize the data
        n_subsample : int
            Number of samples to use for KDE
            
        Returns:
        --------
        p_outlier : pd.DataFrame
            DataFrame containing the probability of each cell being an outlier for each feature pair
        """
        pairs = self.permute_pairs()
        p_outlier = pd.DataFrame(index=self.df.index)
        logger.info("Starting KDE Estimation")
              
        for pair in pairs:
            metric_x = pair[0]
            metric_y = pair[1]
            logger.info(f"On: {metric_x}_vs_{metric_y}")

            x_values = pd.to_numeric(self.df[metric_x]/self.normalize_factor)   
            y_values = pd.to_numeric(self.df[metric_y]/self.normalize_factor)

            density_estimates = []
            for i in range(rounds):
                xS, yS = resample( # monte carlo sub sampling
                    x_values, 
                    y_values, 
                    replace=False, # NOT bootstrap 
                    n_samples=np.amin([n_subsample, len(x_values)]), 
                    random_state=i
                )
                # representation of a kernel-density estimate using Gaussian kernels.
                kde = gaussian_kde(np.vstack([xS, yS]))
                densities = kde(np.vstack([x_values.values, 
                                           y_values.values]))
                # convert denisty to probability mass function (sums to 1)
                densities = densities / np.sum(densities)
                density_estimates.append(densities)

            median_densities = np.median(density_estimates, axis=0)
            p_outlier[f"{metric_x}_vs_{metric_y}"] = median_densities
        return p_outlier

    def identify_outliers(self, p_outlier):
        """
        Identify outlier cells based on density threshold. saves outliers to csv.
        
        Parameters:
        -----------
        p_outlier : pd.DataFrame
            Output from detect_outliers()
        
        Returns:
        --------
        pd.Series : only object ids where (p_outlier < self.density_threshold)
        """
        # Cell is outlier if ANY feature pair has low density
        outlier_mask = (p_outlier < self.density_thresh).any(axis=1)
        # Save outliers to CSV
        outlier_df = self.df[outlier_mask]
        outlier_df.to_csv(self.outlier_path)
        logger.info(f"Saved {len(outlier_df)} outliers to {self.outlier_path}")
        return outlier_mask

    def outlier_scatterplots(self, 
                            outlier_mask, 
                            plot_name="outliers", 
                            markersize=2, 
                            selected_pairs=None, 
                            save_plots=True):
        """
        Create scatterplots of the data with outliers highlighted.
        
        Parameters:
        -----------
        outlier_mask : pd.Series
            Output from identify_outliers()
        plot_name : str
            Name of the plot
        markersize : int
            Size of the markers
        selected_pairs : list
            List of pairs to plot
        save_plots : bool
            Whether to save the plots
        """
        pairs = self.permute_pairs()
        if selected_pairs is not None:
            pairs = pairs[selected_pairs]
        
        df_normalized = self.df / self.normalize_factor
        df_normalized['outlier'] = outlier_mask
        
        n_pairs = len(pairs)
        ncols = min(4, int(np.ceil(np.sqrt(n_pairs))))
        nrows = int(np.ceil(n_pairs / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_pairs == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, pair in enumerate(pairs): # plot each pair of metrics in a subplot
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            
            metric_x, metric_y = pair[0], pair[1]
            
            # Plot normal points
            normal_mask = ~outlier_mask
            if normal_mask.sum() > 0:
                sns.scatterplot(
                    data=df_normalized[normal_mask],
                    x=metric_x, 
                    y=metric_y,
                    alpha=0.6, 
                    s=markersize*10,
                    color=self.normal_color,
                    ax=ax,
                    label='Normal',
                    legend=False
                )
            
            # Plot outliers on top
            if outlier_mask.sum() > 0:
                sns.scatterplot(
                    data=df_normalized[outlier_mask],
                    x=metric_x, 
                    y=metric_y,
                    alpha=0.8, 
                    s=markersize*20,
                    color=self.outlier_color,
                    ax=ax,
                    label='Outliers',
                    legend=False
                )
            
            # Clean up labels
            clean_x = metric_x.replace('_', ' ').replace('-', ' ').title()
            clean_y = metric_y.replace('_', ' ').replace('-', ' ').title()
            ax.set_xlabel(clean_x)
            ax.set_ylabel(clean_y)
            
            # Add legend
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            
            # Add sample size annotation
            n_total = len(df_normalized)
            n_outliers = outlier_mask.sum()
            ax.text(0.05, 0.95, f'n={n_total}\nout={n_outliers}', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Remove empty subplots
        for idx in range(n_pairs, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.remove()
        
        plt.tight_layout()
        
        # Save or show
        if save_plots and self.figure_dir:
            output_path = Path(self.figure_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / f"{plot_name}.png", format="png", dpi=150, 
                       bbox_inches='tight')
            plt.close()
            logger.info(f"Saved plot: {output_path / f'{plot_name}.png'}")
        else:
            plt.show()

    def outlier_summary(self, p_outlier, outlier_mask, plot_name="outlier_summary", save_plots=True):
        """
        Create summary plots using seaborn styling.
        
        Parameters:
        -----------
        p_outlier : pd.DataFrame
            Output from detect_outliers()
        outlier_mask : pd.Series
            Output from identify_outliers()
        plot_name : str
            Name of the plot
        save_plots : bool
            Whether to save the plots
        """
        # Set up colors from glasbey dark palette
        colors = cc.glasbey_dark
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Histogram of density probabilities using seaborn
        ax = axes[0, 0]
        all_densities = p_outlier.values.flatten()
        log_densities = np.log10(all_densities[all_densities > 0])
        
        sns.histplot(log_densities, 
                     bins=100, 
                     alpha=0.7, 
                     color=colors[2], 
                     ax=ax,
                     label='Density Distribution')
        ax.axvline(np.log10(self.density_thresh), 
                   color=colors[1], 
                   linestyle='--', 
                   linewidth=3, 
                   label=f'Outlier Threshold: {self.density_thresh:.0e}')
        
        ax.set_xlabel('Log10(Probability Density)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Cell Probability Densities')
        ax.legend()
        
        # 2. Number of outliers per feature pair
        ax = axes[0, 1]
        outlier_counts = (p_outlier < self.density_thresh).sum()
        
        # Create readable feature pair names
        feature_pair_names = [col.replace('_vs_', '\nvs\n').replace('-', ' ').replace('_', ' ') 
                            for col in p_outlier.columns]
        
        # Use explicit numeric positions for x-axis
        x_positions = list(range(len(outlier_counts)))
        sns.barplot(x=x_positions, 
                    y=outlier_counts.values, 
                    hue=outlier_counts.values,
                    palette=cc.glasbey_dark,
                    legend=True,
                    ax=ax)
        ax.set_xlabel('Feature Pairs')
        ax.set_ylabel('Number of Outliers')
        ax.set_title('Outliers per Feature Pair')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(feature_pair_names, 
                           rotation=45, 
                           ha='right', 
                           fontsize=8)
        
        # 3. Distribution comparison (normal vs outliers) \
        ax = axes[1, 0]
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col_name = numeric_cols[0]
            data_for_plot = pd.DataFrame({
                'values': self.df[col_name] / self.normalize_factor,
                'type': ['Outlier' if x else 'Normal' for x in outlier_mask]
            })
            
            sns.histplot(data=data_for_plot[data_for_plot['type'] == 'Normal'], 
                        x='values', alpha=0.6, bins=50, ax=ax,
                        color=colors[0], label='Normal')
            sns.histplot(data=data_for_plot[data_for_plot['type'] == 'Outlier'], 
                        x='values', alpha=0.8, bins=20, ax=ax,
                        color=colors[1], label='Outliers')
            
            ax.set_xlabel(f'{col_name.replace("_", " ").title()} (normalized)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution Comparison: Normal vs Outliers')
            ax.legend()
        
        # 4. Overall outlier summary pie chart
        ax = axes[1, 1]
        total_cells = len(outlier_mask)
        total_outliers = outlier_mask.sum()
        clean_cells = total_cells - total_outliers
        
        labels = ['Clean Cells', 'Outliers']
        sizes = [clean_cells, total_outliers]
        pie_colors = [colors[0], colors[1]]
        
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 12})
        ax.set_title(f'Overall Outlier Distribution\n({total_outliers}/{total_cells} outliers)')
        
        plt.tight_layout()
        
        if save_plots and self.figure_dir:
            output_path = Path(self.figure_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / f"{plot_name}.png", format="png", dpi=150, 
                       bbox_inches='tight')
            plt.close()
            logger.info(f"Saved plot: {output_path / f'{plot_name}.png'}")
        else:
            plt.show()

    def comprehensive_pairplot(self, outlier_mask, save_plots=True, max_samples=10000):
        """
        Create a comprehensive pairplot using seaborn with optimizations for speed.
        
        Parameters:
        -----------
        outlier_mask : pd.Series
            Output from identify_outliers()
        save_plots : bool
            Whether to save the plots
        max_samples : int
            Maximum number of samples to use for plotting
        """
        logger.info("Creating comprehensive pairplot...")
        
        # Prepare data
        df_plot = self.df.copy() / self.normalize_factor
        df_plot['outlier_status'] = ['Outlier' if x else 'Normal' for x in outlier_mask]
        
        # Limit to numeric columns only
        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
        # Create pairplot with optimized settings
        g = sns.pairplot(
            df_plot[numeric_cols + ['outlier_status']], 
            hue='outlier_status',
            palette=[self.normal_color, self.outlier_color],
            plot_kws={'alpha': 0.6, 's': 5},  # Smaller points, rasterized for speed
            diag_kind='kde'
        )
        
        g.fig.suptitle('Comprehensive Feature Pairplot', y=1.02, fontsize=16)
        
        if save_plots and self.figure_dir:
            output_path = Path(self.figure_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving comprehensive pairplot...")
            plt.savefig(output_path / "comprehensive_pairplot.png", 
                       format="png", 
                       dpi=100,  # Reduced DPI for faster saving
                       bbox_inches='tight',
                       facecolor='white')
            plt.close()
            logger.info(f"Saved plot: {output_path / 'comprehensive_pairplot.png'}")
        else:
            plt.show()

    def diagnostic_plots(self, p_outlier, outlier_mask):
        """
        Generate all diagnostic plots
        
        Parameters:
        -----------
        p_outlier : pd.DataFrame
            Output from detect_outliers()
        outlier_mask : pd.Series
            Output from identify_outliers()

        """
        logger.info("Generating diagnostic plots...")
        
        # 1. Original data (no outliers highlighted)
        self.outlier_scatterplots(
            np.zeros(len(self.df), dtype=bool),
            plot_name="original_data",
            markersize=1
        )
        
        # 2. Data with outliers highlighted
        self.outlier_scatterplots(
            outlier_mask,
            plot_name="outliers_highlighted", 
            markersize=2
        )
        
        # 3. Clean data only
        clean_mask = ~outlier_mask
        if clean_mask.sum() > 0:
            # Create a plot showing only clean data
            self.outlier_scatterplots(
                np.zeros(len(self.df), dtype=bool),  
                plot_name="clean_data",
                markersize=1
            )
        
        # 4. Summary statistics
        self.outlier_summary(p_outlier, outlier_mask)
        
        # 5. Additional seaborn pairplot for comprehensive view
        self.comprehensive_pairplot(outlier_mask)



def parse_args():
    parser = argparse.ArgumentParser(description="KDE Outlier Detection")
    parser.add_argument('--density_thresh', type=float, required=False, default=1e-15, help="Density threshold at which to consider an outlier")
    parser.add_argument('--figure_dir', type=str, required=True, help="Directory to store figures. Will be created if not exists")
    parser.add_argument('--csv_path', type=str, required=True, help="CSV file with single cell data")
    parser.add_argument('--outlier_path', type=str, required=True, help="Where to store csv file of outliers")
    parser.add_argument('--n_subsample', type=float, required=False, default=10000, help="Size of subsample in Monte Carlo Subsampling")
    parser.add_argument('--n_rounds', type=int, required=False, default=5, help="Num of rounds in KDE")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    params = {
        "figure_dir": args.figure_dir,
        "density_thresh": args.density_thresh,
        "csv_path": args.csv_path,
        "figure_dir": args.figure_dir,
        "outlier_path": args.outlier_path,
        "n_subsample": args.n_subsample,
        "n_rounds": args.n_rounds
    } 
    logger.info("--------------------------------")
    logger.info("Arguments read, starting analysis...")
    logger.info("--------------------------------")
    start_time = time.time()
    detector = KDEOutlierDetection(params)
    p_outlier = detector.detect_outliers()
    outliers = detector.identify_outliers(p_outlier)
    detector.diagnostic_plots(p_outlier, outliers)
    end_time = time.time()
    logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
    return 

if __name__ == "__main__":
    main()
    