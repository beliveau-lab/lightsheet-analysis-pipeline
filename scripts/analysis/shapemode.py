import os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from aicsshparam.shtools import (get_reconstruction_from_coeffs, 
                                save_polydata, 
                                convert_coeffs_dict_to_matrix)
import pyvista as pv
import seaborn as sns

LMAX = 16
NUM_PCS = 6

class ShapeSpace:
    def __init__(self, full_path, outlier_path, id_name, max_sd=2, generate_plots=True):
        # Initialize empty instance variables
        self.label_index = None # list of label names to be used as index later
        self.shcoeffs_df = None # df of raw shcoeffs indexed by label id
        self.pca_df = None # df of raw PCA coordiantes
        self.pca = None # pca sklearn object 
        self.df_shapespace = None # digitized shape mode data
        self.pc_scales = {} # dict mapping pc -> sd 
        self.sigma_steps = self.get_sigma_range(max_sd)

        # class variables
        self.lmax = LMAX
        self.num_shapemodes = NUM_PCS
        self.generate_plots = generate_plots
        self.removal_pct = 1.0

        # load and process data
        self.init_data(full_path, outlier_path, id_name)

    def workflow(self, path_to_save, str_tag):
        print("=" * 40)
        print("Running PCA...")
        self.run_pca()

        print("Digitizing Shapemodes...")
        mpids_full = self.digitize_all_shapemodes()

        print(f"Removing the {self.removal_pct}% extremes for mesh generation...")
        is_extreme = self.remove_extreme_points()  # returns a Series on the full index
        self.df_shapespace = pd.concat([mpids_full, is_extreme], axis=1)
        self.store_shapemodes(path_to_save, str_tag)

        print("Recreating PC Meshes...\n")
        self.reconstruct_meshes(path_to_save, str_tag)
        if self.generate_plots:
            print("Saving plots...")
            self.plot_variance(path_to_save, str_tag)
            self.mesh_visualization(path_to_save, str_tag)
        print("=" * 40)

    # ----- Main Shapespace Creation -----
    def run_pca(self):
        sh_matrix = self.shcoeffs_df.values
        self.pca = PCA(n_components=self.num_shapemodes, svd_solver='full')
        self.pca.fit(sh_matrix) 
        axes = self.pca.transform(sh_matrix) 
        pc_columns = [f"PC{i+1}" for i in range(self.num_shapemodes)]
        self.pca_df = pd.DataFrame(axes, columns=pc_columns, index=self.label_index)
        print("PCA Complete")
        print(self.pca_df.describe())
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        return self.pca_df
    
    def shapemode_to_shcoeff(self):
        pc_to_coeffs = []
        pc_names = list(self.pca_df.columns)
        for pc_idx, pc_name in enumerate(pc_names):
            scale = self.get_scale(pc_name)
            coords = [s * scale for s in self.sigma_steps]
            matrix = self.get_coordinates_matrix(coords, pc_idx)
            df_inv = self.invert(matrix)
            df_inv["shape_mode"] = pc_name
            df_inv["mpId"] = np.arange(1, 1 + len(self.sigma_steps))
            pc_to_coeffs.append(df_inv)
        return pd.concat(pc_to_coeffs, ignore_index=True)
    
    def remove_extreme_points(self):
        if not self.removal_pct or self.removal_pct <= 0:
            return self.pca_df
        df = self.pca_df.copy()
        mask = np.zeros(len(df), dtype=bool)
        for col in df.columns:
            ql, qh = np.percentile(df[col].values, [self.removal_pct, 100 - self.removal_pct])
            mask |= (df[col] < ql) | (df[col] > qh)
        extreme_mask = pd.Series(mask, index=self.pca_df.index, name="is_extreme")
        print(f"Extreme rows removed (pct per tail={self.removal_pct}%): {mask.sum()}")
        self.pca_df = df.loc[~mask]
        self.shcoeffs_df = self.shcoeffs_df.loc[self.pca_df.index]
        self.label_index = self.pca_df.index
        return extreme_mask

    def digitize_shapemode(self, pc, src):
        values = src[pc].values.astype(np.float32)
        centered = values - values.mean()
        scale = self.get_scale(pc, df=src)
        norm = centered if scale == 0 else centered / scale
        binw = 0.5 * np.diff(self.sigma_steps).mean() if len(self.sigma_steps) > 1 else 1
        bin_edges = np.unique([(b-binw, b+binw) for b in self.sigma_steps])
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        return np.digitize(norm, bin_edges)
    
    def digitize_all_shapemodes(self):
        df_shapespace = self.pca_df.copy()
        for pc in df_shapespace.columns:
            df_shapespace[f"{pc}_mpId"] = self.digitize_shapemode(pc, src=df_shapespace)
        print("Finished digitizing shapemodes.")
        return df_shapespace

    def reconstruct_meshes(self, save_dir, str_tag):
        reconstruction_df = self.shapemode_to_shcoeff()
        os.makedirs(f"{save_dir}/PC_Reconstructions_{str_tag}", exist_ok=True)
        for _, df_sm in reconstruction_df.groupby("shape_mode"):
            shapemode_path = f"{save_dir}/PC_Reconstructions_{str_tag}/{df_sm['shape_mode'].values[0]}"
            os.makedirs(shapemode_path, exist_ok=True)
            for _, row in df_sm.iterrows():
                shcoeffs = row.filter(like="shcoeffs_")
                row_dict = {coeff_name: shcoeffs[coeff_name] for coeff_name in self.shcoeffs_df.columns}
                coeffs = convert_coeffs_dict_to_matrix(row_dict, lmax=self.lmax)
                mesh, _ = get_reconstruction_from_coeffs(coeffs)
                save_polydata(mesh, f"{shapemode_path}/{row['shape_mode']}_{self.sigma_steps[int(row['mpId'])-1]}.vtk")
            print(f"Saved meshes for pc: {df_sm['shape_mode'].values[0]}")
        print(f"All meshes saved to: {save_dir}/PC_Reconstructions_{str_tag}")
        return
    def init_data(self, full_path, outlier_path, id_name='label'):
        print("Loading data and removing outliers if given.")
        df = pd.read_csv(full_path, low_memory=False)
        if outlier_path is not None:
            outlier_ids = pd.read_csv(outlier_path, usecols=[id_name]).values.flatten()
            df = df[~df[id_name].isin(outlier_ids)]
        self.label_index = df[id_name]
        self.shcoeffs_df = df.filter(like='shcoeffs_')
        self.shcoeffs_df = self.shcoeffs_df.set_index(self.label_index)
        return df
    
    def get_sigma_range(self, max_sd):
        step = 0.5
        num_points = int(2 * max_sd / step) + 1
        return np.linspace(-max_sd, max_sd, num_points).tolist()

    def get_scale(self, pc, df=None):
        src = self.pca_df if df is None else df
        values = src[pc].values.astype(np.float32)
        values_centered = values - values.mean()
        return values_centered.std()
    
    def plot_variance(self, path_to_save, str_tag):        
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        sns.set_style("whitegrid")
        ax.plot(100 * self.pca.explained_variance_ratio_[:self.num_shapemodes], 
                "o--",
                color='black')
        ax.set_xlabel("Principal Component", fontsize=18)
        ax.set_ylabel("Explained variance (%)", fontsize=18)
        ax.set_xticks(np.arange(self.num_shapemodes))
        ax.set_xticklabels(np.arange(1, 1 + self.num_shapemodes))
        title = "Variance Explained (Total={0}%)".format(
            int(100 * self.pca.explained_variance_ratio_[:].sum())
        )
        ax.set_title(title, fontsize=18)
        plt.tight_layout()
        plt.savefig(f'{path_to_save}/var_explained_skew_{str_tag}.png')
        return fig
    
    def get_feature_importance(self):
        pc_columns = [f"PC{i+1}" for i in range(self.num_shapemodes)]
        feat_importance = {}
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        for comp, pc_name in enumerate(pc_columns):
            load = loadings[:, comp]
            pc = [v for v in load]  # Raw loadings
            apc = [v for v in np.abs(load)]  # Absolute loadings
            total = np.sum(apc)
            cpc = [100 * v / total for v in apc]  # Percentage contribution
            feat_importance[pc_name] = pc
            feat_importance[pc_name.replace("PC", "aPC")] = apc # absolute loadings
            feat_importance[pc_name.replace("PC", "cPC")] = cpc # percentage contribution
        df_feat_importance = pd.DataFrame(feat_importance, index=self.shcoeffs_df.columns.tolist())
        return df_feat_importance
        
    def store_shapemodes(self, path_to_save, str_tag, filename='shapemodes.csv'):
        if self.df_shapespace is None:
            raise ValueError("Shape space not digitized. Please run digitize_all_shapemodes() first.")
        path_to_save = os.path.join(path_to_save, f'{str_tag}_{filename}')
        try:
            self.df_shapespace.to_csv(path_to_save)
            print(f"Successsfuly saved {filename} to {path_to_save}")
        except:
            print(f"Failed to save {filename} to {path_to_save}")
        return self.df_shapespace

    def mesh_visualization(self, path_to_save, str_tag):
        fixed_position = [75, 75, 75]
        fixed_focal_point = [0, 0, 0]
        output_dir = f'{path_to_save}/PC_Visualizations_{str_tag}'
        os.makedirs(output_dir, exist_ok=True)
        for pc in range(1, NUM_PCS+1):
            plotter = pv.Plotter(shape=(1, len(self.sigma_steps)), 
                                 window_size=(1400, 200), 
                                 off_screen=True)
            for s_index, s in enumerate(self.sigma_steps):
                pc_str = f'{path_to_save}/PC_Reconstructions_{str_tag}/PC{pc}/PC{pc}_{s}.vtk'
                mesh = pv.read(pc_str)
                plotter.subplot(0, s_index)
                plotter.add_mesh(mesh)
                plotter.camera.position = fixed_position
                plotter.camera.focal_point = fixed_focal_point 
            plotter.save_graphic(f"{path_to_save}/PC_Reconstructions_{str_tag}/{pc}_viz.svg")

    def invert(self, pcs):
        """Matrix has shape NxM, where N is the number of
        samples and M is the number of shape modes."""
        # Inverse PCA here: PCA coords -> shcoeffs
        df = pd.DataFrame(self.pca.inverse_transform(pcs))
        df.columns = self.shcoeffs_df.columns.tolist()
        return df
    
    def get_coordinates_matrix(self, coords, pc_indx):
        npts = len(coords)
        matrix = np.zeros((npts, self.pca.n_components), dtype=np.float32)
        matrix[:, pc_indx] = coords
        return matrix
    
def main():
    base_dir = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306'
    outlier_path = f'{base_dir}/outliers_realigned.csv'
    full_path = f'{base_dir}/dataset_fused_features_realigned.csv'
    shape_space = ShapeSpace(
        full_path, outlier_path, id_name = 'label'
        )
    shape_space.workflow(base_dir, str_tag='realigned')

if __name__ == "__main__":
    main()

