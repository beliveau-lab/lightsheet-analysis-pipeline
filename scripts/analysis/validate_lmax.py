import logging
import time
import numpy as np
import pandas as pd
import dask.array as da
import matplotlib.pyplot as plt
from aicsshparam import shtools, shparam
from dask import compute, delayed
from vtk.util import numpy_support as vtknp
import scipy.spatial as spatial
from scipy import ndimage as ndi
import zarr
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask
import sys

DEFAULT_LMAX = 16
MIN_RELIABLE_OBJECTS = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_n5_zarr_array(path, n5_subpath=None, chunks=None):
    """Loads N5 or Zarr, handling potential subpath for N5."""
    logger.info(f"Attempting to load from: {path}" + (f" with N5 subpath: {n5_subpath}" if n5_subpath else ""))
    if path.endswith('.n5'):
        store = zarr.N5Store(path)
        if not n5_subpath:
             raise ValueError(f"N5 path specified ({path}), but n5_subpath is required.")
        arr_handle = zarr.open_array(store=store, path=n5_subpath, mode='r')
        logger.info(f"Loaded N5 array: Shape={arr_handle.shape}, Chunks={arr_handle.chunks}")
        return da.from_zarr(arr_handle, chunks=chunks)
    elif path.endswith('.zarr'):
        arr_handle = zarr.open(path, mode='r')
        logger.info(f"Loaded Zarr array: Shape={arr_handle.shape}")
        return da.from_zarr(path, chunks=chunks)
    else:
        raise ValueError(f"Unsupported array format (expected .n5 or .zarr): {path}")

def get_meshes_distance(mesh1, mesh2):
    """Credit: https://github.com/AllenCell/cvapipe_analysis/tree/main"""
    coords1 = vtknp.vtk_to_numpy(mesh1.GetPoints().GetData())
    coords2 = vtknp.vtk_to_numpy(mesh2.GetPoints().GetData())
    dist = spatial.distance.cdist(coords1, coords2)
    d12 = dist.min(axis=0)
    d21 = dist.min(axis=1)
    return d12, d21

def add_chunk_indices(df_bboxes: pd.DataFrame, chunk_sizes: tuple) -> pd.DataFrame:
    """Add chunk index columns (cz, cy, cx) using bbox centroid and given chunk sizes."""
    cz_size, cy_size, cx_size = (int(chunk_sizes[0][0]), int(chunk_sizes[1][0]), int(chunk_sizes[2][0]))
    cz = ((df_bboxes["z0"] + df_bboxes["z1"]) // 2) // cz_size
    cy = ((df_bboxes["y0"] + df_bboxes["y1"]) // 2) // cy_size
    cx = ((df_bboxes["x0"] + df_bboxes["x1"]) // 2) // cx_size
    df_bboxes = df_bboxes.copy()
    df_bboxes["cz"] = cz.astype(np.int64)
    df_bboxes["cy"] = cy.astype(np.int64)
    df_bboxes["cx"] = cx.astype(np.int64)
    return df_bboxes

def mesh_based_distance(obj_id, mask_slice, min_volume, lmax_range, sampling_rate):
    obj_results = {'id': obj_id, 'distances': {}}
    label_slice = np.where(mask_slice == obj_id, mask_slice, 0)
    label_slice = np.ascontiguousarray(label_slice)
    if np.sum(label_slice > 0) < min_volume:
        logger.debug(f"Object {obj_id} is too small")
        return None
    try:
        for lmax in lmax_range:
            (_, grid_rec), (_, mesh, _, _) = shparam.get_shcoeffs(
                label_slice, lmax=lmax, alignment_2d=False
            )
            mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
            d12, d21 = get_meshes_distance(mesh, mesh_rec)
            d12 = np.median(d12) # distance from mesh to mesh_rec
            d21 = np.median(d21) # distance from mesh_rec to mesh
            error = (d12 + d21) * sampling_rate
            if error is not None and not np.isnan(error):
                obj_results['distances'][lmax] = error 
        return obj_results
    except Exception as e:
        logger.debug(f"Failed to process object {obj_id}: {e}")
        return None

def grid_based_MSE(obj_id, mask_slice, min_volume, lmax_range, sampling_rate):
    label_slice = np.where(mask_slice == obj_id, mask_slice, 0)
    obj_results = {'id': obj_id, 'distances': {}}
    if np.sum(label_slice > 0) < min_volume:
        logger.debug(f"Object {obj_id} is too small")
        return None
    try:
        for lmax in lmax_range:
            (_, grid_rec), _ = shparam.get_shcoeffs(
                label_slice, lmax=lmax, alignment_2d=False
            )
            MSE = shtools.get_reconstruction_error(label_slice, grid_rec) * sampling_rate
            if MSE is not None and not np.isnan(MSE):
                obj_results['distances'][lmax] = MSE 
        return obj_results
    except Exception as e:
        logger.debug(f"Failed to process object {obj_id}: {e}")

def open_mask_roi(mask_path, zstart, zend, ystart, yend, xstart, xend):
    if mask_path.endswith('.n5'):
        store = zarr.N5Store(mask_path)
        mask_arr = zarr.open_array(store=store, path=None, mode='r')
    else:
        mask_arr = zarr.open(mask_path, mode='r')
    return mask_arr[zstart:zend, ystart:yend, xstart:xend]
    
def process_spatial_group(group_df: pd.DataFrame,
                          mask_path: str,
                          lmax_range: list,
                          sampling_rate: float,
                          min_volume: int,
                          group_key: tuple) -> list:
    if group_df.empty:
        return []
    labels = group_df["label"].astype(np.int64).tolist()
    # Calculate bounding box that contains ALL objects in this group
    zstart = int(group_df["z0"].min()); zend = int(group_df["z1"].max())
    ystart = int(group_df["y0"].min()); yend = int(group_df["y1"].max())
    xstart = int(group_df["x0"].min()); xend = int(group_df["x1"].max())
    logger.debug(f"Group {group_key}: Reading ROI [{zstart}:{zend}, {ystart}:{yend}, {xstart}:{xend}] for {len(labels)} objects")
    mask_roi = open_mask_roi(
        mask_path, zstart, zend, ystart, yend, xstart, xend
        )
    # Quick exit if no voxels
    if np.count_nonzero(mask_roi) == 0:
        return []
    results = []
    for obj_id in labels:
        if obj_id not in np.unique(mask_roi):
            continue
        result = grid_based_MSE(
            obj_id, mask_roi, min_volume, lmax_range, sampling_rate
        )
        if result is not None:
            results.append(result)
    logger.debug(f"Group {group_key}: Processed {len(results)} objects successfully")
    return results

class ValidateLMAX:
    def __init__(self, params):
        self.params = params
        self.client = None
        self.cluster = None
        self.lmax_range = list(
            range(self.params['lmax_min'], 
                  self.params['lmax_max'], 
                  8))
        
    def workflow(self, mask_path, bboxes_path, plot_results=False):
        try:
            start_time = time.time()
            if self.params.get('use_dask', True):
                self._setup_dask()
            self.mask_path = mask_path
            mask_array, df_bboxes = self._load_data(mask_path, bboxes_path)
            object_errors = self.run_computation(mask_array, df_bboxes)
            logger.info(f"Computed errors for {len(object_errors)} objects.")
            if plot_results:
                self.plot_reconstruction_error(object_errors, reference_lmax=None)
            end_time = time.time()
            logger.info(f"Time taken: {end_time - start_time} seconds")
            return 
        finally:
            if self.client:
                self._shutdown_dask()

    def _load_data(self, mask_path, bboxes_path):
        """Load objects from mask. Returns DataFrame for efficient grouping."""
        logger.info("Loading data and finding objects...")
        mask_array = load_n5_zarr_array(mask_path)
        df_bboxes = pd.read_csv(bboxes_path)
        return mask_array, df_bboxes
    
    def run_computation(self, mask_array, df_bboxes):
        logger.info("Using optimized spatial loading strategy...")
        
        # Add spatial chunk indices 
        df_bboxes = add_chunk_indices(df_bboxes, mask_array.chunks)
        if self.params.get('min_size', 0) > 0:
            voxel_est = ((df_bboxes["z1"] - df_bboxes["z0"]) * 
                        (df_bboxes["y1"] - df_bboxes["y0"]) * 
                        (df_bboxes["x1"] - df_bboxes["x0"]))
            df_bboxes = df_bboxes.loc[voxel_est >= self.params['min_size'], :]
            logger.info(f"After size filtering: {len(df_bboxes)} objects")

        # Group by spatial chunks 
        grouped = df_bboxes.groupby(["cz", "cy", "cx"], sort=False)
        group_items = list(grouped)
        logger.info(f"Created {len(group_items)} spatial groups (avg {len(df_bboxes)/len(group_items):.1f} objects/group)")
        
        group_sizes = [len(gdf) for _, gdf in group_items]
        logger.info(f"Group sizes - min: {min(group_sizes)}, max: {max(group_sizes)}, mean: {np.mean(group_sizes):.1f}")

        # Process groups in parallel or sequentially
        logger.info("Processing groups with Dask distributed...")
        futures = []
        for (cz, cy, cx), gdf in group_items:
            fut = self.client.submit(
                process_spatial_group,
                gdf,
                self.mask_path,
                self.lmax_range,
                self.params['sampling_rate'],
                self.params['min_size'],
                (int(cz), int(cy), int(cx))
            )
            futures.append(fut)

        results = []
        for i, fut in enumerate(futures):
            try:
                group_results = fut.result()
                results.extend(group_results)
                logger.info(f"Completed {i+1}/{len(futures)} groups")
            except Exception as e:
                logger.error(f"Group processing failed: {e}")
            finally:
                fut.release()
        logger.info(f"Successfully processed {len(results)} objects total")
        return [r for r in results if r is not None]
    
    def structure_plotting_data(self, object_errors):
        plot_data = []
        for obj in object_errors:
            obj_id = obj['id']
            for lmax, error in obj['distances'].items():
                plot_data.append({
                    'object_id': obj_id,
                    'lmax': lmax,
                    'error': error
                })
        df = pd.DataFrame(plot_data)
        df.to_csv(f'{self.params["save_dir"]}/object_reconstruction_errors.csv', index=False)
        return df
    
    def plot_reconstruction_error(self, object_errors, reference_lmax=None):        
        df_plot = self.structure_plotting_data(object_errors)
        logger.info(f"Saved object reconstruction errors to {self.params['save_dir']}/object_reconstruction_errors.csv")
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        # --- Individual Object Curves in Gray ---
        for obj_id in df_plot['object_id'].unique():
            obj_data = df_plot[df_plot['object_id'] == obj_id].sort_values('lmax')
            x_values = (obj_data['lmax'] + 1) ** 2
            ax.plot(x_values, obj_data['error'], '-', color='gray', linewidth=0.2, alpha=0.5)
        # --- Aggregated Statistics ---
        df_agg = (df_plot.groupby('lmax')['error'].agg(['mean', 'std']).reset_index().sort_values('lmax'))
        x_agg = (df_agg['lmax'] + 1) ** 2
        # Mean line
        ax.plot(x_agg, df_agg['mean'], '-', color='k', linewidth=2, label='Mean')
        # Standard deviation band
        ax.fill_between(x_agg, df_agg['mean'] - df_agg['std'], df_agg['mean'] + df_agg['std'], alpha=0.3, color='black', label='±1 STD')
        # --- Axis Formatting ---
        ax.set_yscale('log')
        max_lmax = df_plot['lmax'].max()
        ax.set_xlim(1, (max_lmax + 1) ** 2)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 8, 10])
        ax.set_yticklabels(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0', '10.0'])
        # --- Reference Line and Title ---
        if reference_lmax is not None:
            x_ref = (reference_lmax + 1) ** 2
            ax.axvline(x=x_ref, color='red', linestyle='--', alpha=0.7, label=f'L={reference_lmax}')
        title = f"Reconstruction Error vs L (n={len(df_plot['object_id'].unique())})"
        # --- Final Formatting ---
        ax.set_title(title)
        ax.set_xlabel('L (SHE order)')
        ax.set_ylabel('Mean distance to closest point (μm)')
        plt.tight_layout()
        # --- Save Plot ---
        save_path = f'{self.params["save_dir"]}/reconstruction_error.png'
        plt.savefig(save_path)
        return fig

    def _setup_dask(self):
        """Setup Dask SGE cluster for distributed processing."""
        try:
            logger.info("Setting up distributed Dask cluster...")
            self.cluster, self.client = setup_dask_sge_cluster(
                n_workers=self.params.get('num_workers', 1),
                cores=self.params.get('cpu_cores', 32),
                processes=self.params.get('cpu_processes', 64),
                memory=self.params.get('cpu_memory', '512G'),
                project=self.params.get('project', 'beliveaulab'),
                queue=self.params.get('queue', 'beliveau-long.q'),
                runtime=self.params.get('runtime', '7200'),
                resource_spec=self.params.get('cpu_resource_spec', 'mfree=16G'),
                log_directory=self.params.get('log_dir', None),
                conda_env=self.params.get('conda_env', 'otls-pipeline'),
                dashboard_port=self.params.get('dashboard_port', ':41236')
            )
            logger.info(f"Dask dashboard link: {self.client.dashboard_link}")
            return True
        except Exception as e:
            logger.error(f"Failed to setup distributed cluster: {e}")
            return False

    def _shutdown_dask(self):
        """Shutdown Dask cluster."""
        if self.cluster and self.client:
            try:
                shutdown_dask(self.cluster, self.client)
                logger.info("Dask cluster shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down cluster: {e}")

def main():
    mask_path = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_masks.zarr'
    bboxes_path = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_features_realigned_bboxes.csv'
    params = {
    # -- dask parameters --
    "num_workers": 1,
    "cpu_memory": "512G", 
    "cpu_cores": 16,
    "cpu_processes": 16, # 2 proc per core
    "cpu_resource_spec": "mfree=8G",  # RAM/worker = RAM/core * cores/worker (16G/core * 1 core/2 proc = 8G/proc)
    # -- computation parameters --
    "sampling_rate": 2.752,
    "min_size": 4000,
    "lmax_min": 4,
    "lmax_max": 40,
    "batch_size": 1000,
    "n_iter": 1000,
    # -- tracking parameters --
    'use_dask': True,
    "log_dir" : '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/analysis/choose_lmax_dir/',
    "save_dir": '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/figures/',
    "dashboard_port": ":41263"
    }
    ValidateLMAX(params).workflow(mask_path, bboxes_path, plot_results=True)

if __name__ == "__main__":
    main()
