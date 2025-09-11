import os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from aicsshparam.shtools import (get_reconstruction_from_coeffs, 
                                save_polydata, 
                                convert_coeffs_dict_to_matrix)
LMAX = 16
NUM_SHAPEMODES = 6

class ShapeSpace:
    def __init__(self, df, num_shapemodes, lmax, id_name):
        self.lmax = lmax
        self.label_index = None
        self.sigma_steps = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        self.num_shapemodes = num_shapemodes
        self.shcoeffs_df = None
        self.pca_df = None # df of raw PCA coordiantes
        self.pca = None # pca object 
        self.df_feat_importance = None 
        self.df_shapespace = None # digitized shape mode data
        self.pc_scales = {}
        self.process_data(df, id_name)

    def process_data(self, df, id_name):
        self.label_index = df[id_name]
        self.shcoeffs_df = df.filter(like='shcoeffs_')
        self.shcoeffs_df = self.shcoeffs_df.set_index(self.label_index)
        return 
        
    def workflow(self, path_to_save):
        print('=' * 40)
        print("Running PCA...")
        self.run_pca()
        print('Recreating PC Meshes...\n')
        self.reconstruct_meshes(path_to_save)
        print('=' * 40)
        # print("Digitizing shapemodes...")
        # self.digitize_all_shapemodes()
        # print('=' * 40)
        # print('\n')

    def run_pca(self):
        sh_matrix = self.shcoeffs_df.values
        self.pca = PCA(n_components=self.num_shapemodes, svd_solver='full')
        self.pca.fit(sh_matrix) 
        axes = self.pca.transform(sh_matrix) 
        
        pc_columns = [f"PC{i+1}" for i in range(self.num_shapemodes)]
        self.pca_df = pd.DataFrame(axes, columns=pc_columns, index=self.label_index)
        
        print(self.pca_df.describe())
        print(f"PCA shape: {axes.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        return self.pca_df
    
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

        self.df_feat_importance = pd.DataFrame(
            feat_importance, index=self.shcoeffs_df.columns.tolist()
            )
        return self.df_feat_importance

    def store_shapemodes(self, path_to_save, filename='shapemodes.csv'):
        """Store shape mode results and optionally save to CSV."""
        if self.df_shapespace is None:
            raise ValueError("Shape space not digitized. Please run digitize_all_shapemodes() first.")
        path_to_save = os.path.join(path_to_save, filename)
        try:
            self.df_shapespace.to_csv(path_to_save)
            print(f"Successsfuly saved {filename} to {path_to_save}")
        except:
            print(f"Failed to save {filename} to {path_to_save}")
        return self.df_shapespace
    
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

    def get_scale(self, pc):
        values = self.pca_df[pc].values.astype(np.float32)
        values_centered = values - values.mean()
        scale = values_centered.std()
        return scale

    def shapemode_to_shcoeff(self):
        dfs = []
        pc_names = list(self.pca_df.columns)
        for pc_idx, pc_name in enumerate(pc_names):
            scale = self.get_scale(pc_name)
            coords = [s * scale for s in self.sigma_steps]
            matrix = self.get_coordinates_matrix(coords, pc_idx)
            print(matrix)
            df_inv = self.invert(matrix)
            df_inv["shape_mode"] = pc_name
            df_inv["mpId"] = np.arange(1, 1 + len(self.sigma_steps))
            dfs.append(df_inv)
        final_df = pd.concat(dfs, ignore_index=True)
        return final_df

    def reconstruct_meshes(self, save_dir, save_meshes=True):
        reconstruction_df = self.shapemode_to_shcoeff()
        meshes = {}
        if save_meshes:
            os.makedirs(f'{save_dir}/PC_Reconstructions', exist_ok=True)

        for sm, df_sm in reconstruction_df.groupby("shape_mode"):
            if save_meshes:
                shapemode_path = f'{save_dir}/PC_Reconstructions/{df_sm['shape_mode'].values[0]}'
                os.makedirs(shapemode_path, exist_ok=True)
            for _, row in df_sm.iterrows():
                shcoeffs = row.filter(like='shcoeffs_')
                row_dict = {coeff_name: shcoeffs[coeff_name] for index, coeff_name in enumerate(self.shcoeffs_df.columns.tolist())}
                coeffs = convert_coeffs_dict_to_matrix(row_dict, lmax=self.lmax)
                mesh, grid = get_reconstruction_from_coeffs(coeffs)
                if save_meshes:
                    save_polydata(mesh, f'{shapemode_path}/{row['shape_mode']}_{self.sigma_steps[int(row['mpId'])-1]}.vtk')
        return

def main():
    base_dir = '/net/beliveau/vol2/instrument/E9.5_317/Zoom_317'
    csv_name = 'dataset_fused_features_sh.csv'
    csv_path = os.path.join(base_dir, csv_name)
    
    print(f"Loading data from {csv_path}...")
    df_dask = pd.read_csv(csv_path, low_memory=False)

    shape_space = ShapeSpace(
        df_dask, num_shapemodes=NUM_SHAPEMODES, lmax=LMAX, id_name='label'
        )
    shape_space.workflow(base_dir)
if __name__ == "__main__":
    main()



# def digitize_all_shapemodes(self):
#     self.df_shapespace = self.pca_df.copy()  # Start with PCA coordinates 
#     for pc in self.pca_df.columns:
#         next_mpId = self.digitize_shapemode(pc) # Bin based on each pc
#         self.df_shapespace[f"{pc}_mpId"] = next_mpId
#     return self.df_shapespace


# def digitize_shapemode(self, pc):
#     values = self.pca_df[pc].values.astype(np.float32)
#     values_centered = values - values.mean()
#     scale = values_centered.std()
#     self.pc_scales[pc] = scale
#     if scale == 0:
#         values_normalized = values_centered  # If no variation, keep centered values
#     else:
#         values_normalized = values_centered / scale
#     binw = 0.5 * np.diff(self.sigma_steps).mean() if len(self.sigma_steps) > 1 else 1
#     bin_edges = np.unique([(b-binw, b+binw) for b in self.sigma_steps])
#     bin_edges[0] = -np.inf
#     bin_edges[-1] = np.inf
#     mpId = np.digitize(values_normalized, bin_edges)
#     return mpId


# def get_labels_by_binned_pc(self, pc, map_point_index=None):
#     if self.df_shapespace is None or pc not in self.pca_df.columns:
#         self.digitize_shapemode(pc)
    
#     df_tmp = self.df_shapespace
#     if map_point_index is not None:
#         if map_point_index < 1 or map_point_index > len(self.sigma_steps):
#             raise ValueError(f"Map point index must be in range [1, {len(self.sigma_steps)}]")
#         df_tmp = df_tmp.loc[df_tmp[f"{pc}_mpId"] == map_point_index]
    
#     return df_tmp.index.values.tolist()
# def plot_variance_explained(self):        
#     fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#     ax.plot(100 * self.pca.explained_variance_ratio_[:self.num_shapemodes], "-o")
#     title = "Variance Explained, Total= {0}%".format(
#         int(100 * self.pca.explained_variance_ratio_[:].sum()),
#     )
#     ax.set_xlabel("Principal Component", fontsize=18)
#     ax.set_ylabel("Explained variance (%)", fontsize=18)
#     ax.set_xticks(np.arange(self.num_shapemodes))
#     ax.set_xticklabels(np.arange(1, 1 + self.num_shapemodes))
#     ax.set_title(title, fontsize=18)
#     plt.tight_layout()
#     plt.show()
#     return fig