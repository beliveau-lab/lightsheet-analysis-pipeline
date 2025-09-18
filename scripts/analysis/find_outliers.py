#!/usr/bin/env python3
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

from dask import delayed
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask

print("Finished imports")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Style and vector outputs
sns.set_style("whitegrid")
sns.set_palette(cc.glasbey_dark)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.format"] = "svg"

def _compute_median_distances(metric_x, metric_y, x_values, y_values, rounds, n_subsample):
    """Picklable helper for Dask: compute median KDE densities for a feature pair."""
    n = len(x_values)
    density_estimates = []
    for i in range(rounds):
        xS, yS = resample(
            x_values, y_values,
            replace=False,
            n_samples=min(n_subsample, n),
            random_state=i
        )
        kde = gaussian_kde(np.vstack([xS, yS]))
        dens = kde(np.vstack([x_values, y_values]))
        dens = dens / dens.sum()
        density_estimates.append(dens)
    median_densities = np.median(density_estimates, axis=0)
    return f"{metric_x}_vs_{metric_y}", median_densities


class KDEOutlierDetection:
    def __init__(self, params):
        self.figure_dir = params["figure_dir"]
        self.density_thresh = params["density_thresh"]
        self.csv_path = params["csv_path"]
        self.outlier_path = params["outlier_path"]
        self.n_subsample = params["n_subsample"]
        self.n_rounds = params["n_rounds"]
        self.df = None
        self.normalize_factor = 1000
        self.normal_color = cc.glasbey_dark[2]
        self.outlier_color = cc.glasbey_dark[0]
        self.init_data()

    def clean_data(self, s):
        try:
            return float(str(s).strip("[]"))
        except ValueError:
            return float(s)

    def init_data(self):
        metrics = [
            "area",
            "label",
            "major_magnitude",
            "intermediate_magnitude",
            "minor_magnitude",
        ]
        self.df = pd.read_csv(
            self.csv_path,
            usecols=metrics,
            index_col="label",
            converters={"label": self.clean_data, "area": self.clean_data},
            dtype={
                "major_magnitude": "float32",
                "intermediate_magnitude": "float32",
                "minor_magnitude": "float32",
            },
        )
        logger.info(f"Using metrics: {list(self.df.columns)}. {len(self.df)} objects loaded")

    def permute_pairs(self):
        feature_names = list(self.df.columns)
        pairs = np.array(
            [[feature_names[i], feature_names[j]]
             for i in range(len(feature_names))
             for j in range(i + 1, len(feature_names))]
        )
        logger.info(f"{len(pairs)} feature pairs")
        return pairs

    # -------- Detection (serial) --------
    def detect_outliers(self, rounds=None, n_subsample=None):
        rounds = self.n_rounds if rounds is None else rounds
        n_subsample = self.n_subsample if n_subsample is None else n_subsample

        pairs = self.permute_pairs()
        p_outlier = pd.DataFrame(index=self.df.index)
        logger.info("Starting KDE Estimation (serial)")

        x_cache = {col: (self.df[col] / self.normalize_factor).to_numpy(dtype=float, copy=False)
                   for col in self.df.columns}

        for metric_x, metric_y in pairs:
            logger.info(f"On: {metric_x}_vs_{metric_y}")
            _, med = _compute_median_distances(
                metric_x, metric_y, x_cache[metric_x], x_cache[metric_y], rounds, n_subsample
            )
            p_outlier[f"{metric_x}_vs_{metric_y}"] = med
        return p_outlier

    # -------- Detection (Dask-parallel) --------
    def detect_outliers_dask(self, client, rounds=None, n_subsample=None):
        rounds = self.n_rounds if rounds is None else rounds
        n_subsample = self.n_subsample if n_subsample is None else n_subsample

        pairs = self.permute_pairs()
        logger.info("Starting KDE Estimation (Dask)")
        # Pre-extract and normalize features once to reduce serialization overhead
        x_cache = {col: (self.df[col] / self.normalize_factor).to_numpy(dtype=float, copy=False)
                   for col in self.df.columns}

        delayed_tasks = []
        for metric_x, metric_y in pairs:
            x_vals = x_cache[metric_x]
            y_vals = x_cache[metric_y]
            task = delayed(_compute_median_distances)(
                metric_x, metric_y, x_vals, y_vals, rounds, n_subsample
            )
            delayed_tasks.append(task)

        results = client.compute(delayed_tasks, sync=True)
        p_outlier = pd.DataFrame(index=self.df.index)
        for col_name, med in results:
            p_outlier[col_name] = med
        return p_outlier

    def identify_outliers(self, p_outlier: pd.DataFrame) -> pd.Series:
        outlier_mask = (p_outlier < self.density_thresh).any(axis=1)
        outlier_df = self.df[outlier_mask]
        outlier_df.to_csv(self.outlier_path)
        logger.info(f"Saved {len(outlier_df)} outliers to {self.outlier_path}")
        return outlier_mask

    # ---------- Plotting helpers (vector-first) ----------
    def _norm_df(self) -> pd.DataFrame:
        df_norm = self.df.copy()
        num_cols = df_norm.select_dtypes(include=[np.number]).columns
        df_norm[num_cols] = df_norm[num_cols] / self.normalize_factor
        return df_norm

    def _make_axes_grid(self, n_plots: int, max_cols: int = 4, figsize=(5, 4)):
        ncols = min(max_cols, int(np.ceil(np.sqrt(n_plots)))) if n_plots > 1 else 1
        nrows = int(np.ceil(n_plots / ncols)) if n_plots > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
        axes = np.atleast_1d(axes).ravel()
        return fig, axes, nrows, ncols

    def _save_vector(self, fig, plot_name: str):
        output_path = Path(self.figure_dir) if self.figure_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path / f"{plot_name}.svg",
                format="svg",
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
            logger.info(f"Saved plot: {output_path / f'{plot_name}.svg'}")
        else:
            plt.show()

    def _clean_label(self, name: str) -> str:
        return name.replace("_", " ").replace("-", " ").title()

    # ---------- Plots ----------
    def outlier_scatterplots(
        self,
        outlier_mask: pd.Series,
        plot_name: str = "outliers",
        markersize: int = 2,
        selected_pairs=None,
        save_plots: bool = True,
        show_legend: bool = True,
    ):
        pairs = self.permute_pairs()
        if selected_pairs is not None:
            pairs = pairs[selected_pairs]

        df_norm = self._norm_df()
        df_norm["__is_outlier__"] = outlier_mask.astype(bool)

        n_pairs = len(pairs)
        fig, axes, _, _ = self._make_axes_grid(n_pairs)

        for idx, (x, y) in enumerate(pairs):
            ax = axes[idx]
            normal_mask = ~df_norm["__is_outlier__"]

            if normal_mask.any():
                sns.scatterplot(
                    data=df_norm.loc[normal_mask],
                    x=x,
                    y=y,
                    alpha=0.6,
                    s=markersize * 10,
                    color=self.normal_color,
                    ax=ax,
                    legend=False,
                )

            if df_norm["__is_outlier__"].any():
                sns.scatterplot(
                    data=df_norm.loc[df_norm["__is_outlier__"]],
                    x=x,
                    y=y,
                    alpha=0.8,
                    s=markersize * 20,
                    color=self.outlier_color,
                    ax=ax,
                    legend=False,
                )

            ax.set_xlabel(self._clean_label(x))
            ax.set_ylabel(self._clean_label(y))

            if show_legend:
                handles = [
                    plt.Line2D([0], [0], marker="o", color="w", label="Normal",
                               markerfacecolor=self.normal_color, markersize=6, alpha=0.8),
                    plt.Line2D([0], [0], marker="o", color="w", label="Outliers",
                               markerfacecolor=self.outlier_color, markersize=6, alpha=0.9),
                ]
                ax.legend(handles=handles, loc="upper right", frameon=True, fancybox=True, shadow=True)

        for j in range(n_pairs, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        if save_plots:
            self._save_vector(fig, plot_name)
        else:
            plt.show()

    def outlier_summary(self, p_outlier, outlier_mask, plot_name="outlier_summary", save_plots=True):
        """
        Concise summary: (1) Outlier counts per feature pair, (2) Overall pie.
        Vector output (svg).
        """
        colors = cc.glasbey_dark

        counts = (p_outlier < self.density_thresh).sum()
        labels = [c.replace("_vs_", " vs ").replace("-", " ").replace("_", " ") for c in p_outlier.columns]
        order = np.argsort(counts.values)[::-1]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        ax.barh(np.array(labels)[order], counts.values[order], color=colors[0])
        ax.invert_yaxis()
        ax.set_xlabel("Number of Outliers")
        ax.set_title("Outliers per Feature Pair")

        ax = axes[1]
        total = int(len(outlier_mask))
        total_out = int(outlier_mask.sum())
        ax.pie(
            [total - total_out, total_out],
            labels=["Clean Cells", "Outliers"],
            colors=[colors[2], colors[1]],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax.set_title(f"Overall Outlier Distribution\n({total_out}/{total} outliers)")

        fig.tight_layout()
        if save_plots:
            self._save_vector(fig, plot_name)
        else:
            plt.show()

    def comprehensive_pairplot(self, outlier_mask, save_plots: bool = True, max_samples: int = 10000):
        logger.info("Creating comprehensive pairplot...")
        df_plot = self._norm_df()
        df_plot["outlier_status"] = pd.Categorical(
            np.where(outlier_mask, "Outlier", "Normal"),
            categories=["Normal", "Outlier"],  # draw Normal first, Outlier second
            ordered=True
        )
        if len(df_plot) > max_samples:
            df_plot = df_plot.sample(n=max_samples, random_state=0)
        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
        cols = numeric_cols + ["outlier_status"]
        g = sns.pairplot(
            df_plot[cols],
            hue="outlier_status",
            hue_order=["Normal", "Outlier"],  
            palette=[self.normal_color, self.outlier_color],
            plot_kws={"alpha": 0.6, "s": 5},
            diag_kind="kde",
        )
        g.fig.suptitle("Comprehensive Feature Pairplot", y=1.02, fontsize=16)

        if save_plots:
            output_path = Path(self.figure_dir) if self.figure_dir else None
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info("Saving comprehensive pairplot...")
                g.fig.savefig(
                    output_path / "comprehensive_pairplot.svg",
                    format="svg",
                    bbox_inches="tight",
                    facecolor="white",
                )
                plt.close(g.fig)
                logger.info(f"Saved plot: {output_path / 'comprehensive_pairplot.svg'}")
            else:
                plt.show()
        else:
            plt.show()

    def diagnostic_plots(self, p_outlier, outlier_mask):
        logger.info("Generating diagnostic plots...")

        self.outlier_scatterplots(
            outlier_mask=np.zeros(len(self.df), dtype=bool),
            plot_name="original_data",
            markersize=1,
        )
        self.outlier_scatterplots(
            outlier_mask=outlier_mask,
            plot_name="outliers_highlighted",
            markersize=2,
        )
        self.outlier_summary(p_outlier, outlier_mask)
        self.comprehensive_pairplot(outlier_mask)


def _setup_dask(params):
    """Setup Dask SGE cluster for distributed processing."""
    try:
        logger.info("Setting up distributed Dask cluster...")
        cluster, client = setup_dask_sge_cluster(
            n_workers=params.get("num_workers", 1),
            cores=params.get("cpu_cores", 4),
            processes=params.get("cpu_processes", 1),
            memory=params.get("cpu_memory", "60G"),
            project=params.get("project", "beliveaulab"),
            queue=params.get("queue", "beliveau-long.q"),
            runtime=params.get("runtime", "7200"),
            resource_spec=params.get("cpu_resource_spec", "mfree=60G"),
            log_directory=params.get("log_dir", None),
            conda_env=params.get("conda_env", "otls-pipeline"),
            dashboard_port=params.get("dashboard_port", None),
        )
        logger.info(f"Dask dashboard link: {client.dashboard_link}")
        return cluster, client
    except Exception as e:
        logger.error(f"Failed to setup distributed cluster: {e}")
        return None, None


def _shutdown_dask(cluster, client):
    if cluster and client:
        try:
            shutdown_dask(cluster, client)
            logger.info("Dask cluster shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down cluster: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="KDE Outlier Detection (Dask-parallel)")
    parser.add_argument("--density_thresh", type=float, default=1e-15,
                        help="Density threshold at which to consider an outlier")
    parser.add_argument("--figure_dir", type=str, required=True,
                        help="Directory to store figures. Will be created if not exists")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV file with single cell data")
    parser.add_argument("--outlier_path", type=str, required=True,
                        help="Where to store csv file of outliers")
    parser.add_argument("--n_subsample", type=int, default=10000,
                        help="Size of subsample in Monte Carlo subsampling")
    parser.add_argument("--n_rounds", type=int, default=5,
                        help="Num of rounds in KDE")
    return parser.parse_args()


def main():
    use_dask = True
    args = parse_args()
    params = {
        "figure_dir": args.figure_dir,
        "density_thresh": args.density_thresh,
        "csv_path": args.csv_path,
        "outlier_path": args.outlier_path,
        "n_subsample": args.n_subsample,
        "n_rounds": args.n_rounds,
    }

    logger.info("--------------------------------")
    logger.info("Arguments read, starting analysis...")
    logger.info("--------------------------------")
    start_time = time.time()

    detector = KDEOutlierDetection(params)

    cluster = client = None
    try:
        if use_dask:
            dask_params = {
                "num_workers": 1,
                "cpu_memory": "64G",
                "cpu_cores": 6,
                "cpu_processes": 1,
                "project": "beliveaulab",
                "queue": "beliveau-long.q",
                "runtime": 7200,
                "cpu_resource_spec": "mfree=60G",
                "log_dir": detector.figure_dir,
                "conda_env": "otls-pipeline",
                "dashboard_port": ":41236",
            }
            cluster, client = _setup_dask(dask_params)
            logger.info(f"Dask dashboard: {client.dashboard_link}")
            p_outlier = detector.detect_outliers_dask(client)
        else:
            p_outlier = detector.detect_outliers()

        outliers = detector.identify_outliers(p_outlier)
        detector.diagnostic_plots(p_outlier, outliers)

    finally:
        _shutdown_dask(cluster, client)

    end_time = time.time()
    logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()