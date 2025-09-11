# -- standard library --
import logging
import time
# -- third-party --
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sparse
from aicsshparam import shtools, shparam
from dask import compute, delayed
from vtk.util import numpy_support as vtknp
import scipy.spatial as spatial
from matplotlib.ticker import FormatStrFormatter

# -- local --
import align_3d as align
import feature_extraction as fe
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask

# -- constants --
DEFAULT_LMAX = 12
MIN_RELIABLE_OBJECTS = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_meshes_distance(mesh1, mesh2):
    """"Credit: https://github.com/AllenCell/cvapipe_analysis/tree/main"""
    coords1 = vtknp.vtk_to_numpy(mesh1.GetPoints().GetData())
    coords2 = vtknp.vtk_to_numpy(mesh2.GetPoints().GetData())
    dist = spatial.distance.cdist(coords1, coords2)
    d12 = dist.min(axis=0)
    d21 = dist.min(axis=1)
    return d12, d21
    
def compute_mean_distances(obj, mask_slice, min_volume, lmax_range, sampling_rate):
    """Compute the reconstruction error for a single object at each lmax value.

    Params
    ------
    obj: pandas series containing the object id and bounding box
    mask_slice: sparse array containing the object mask
    min_volume: minimum volume of the object
    lmax_range: list of lmax values to compute the reconstruction error for
    sampling_rate: sampling rate of the microscope in um/pixel

    Returns
    -------
        obj_results: dictionary containing the object id -> reconstruction errors for each lmax value
    """
    obj_id = int(obj.name)
    obj_results = {'id': obj_id, 'distances': {}}
    label_slice = np.where(mask_slice == obj_id, mask_slice, 0)
    if np.sum(label_slice > 0) < min_volume:
        logger.debug(f"Object {obj_id} is too small")
        return None
    try:
        for lmax in lmax_range:
            (_, grid_rec), (_, mesh, _, _) = shparam.get_shcoeffs(
                label_slice, lmax=lmax, alignment_2d=False
            )
            # Compute reconstruction error
            mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
            d12, d21 = get_meshes_distance(mesh, mesh_rec)
            d12 = np.median(d12) # distance from mesh to mesh_rec
            d21 = np.median(d21) # distance from mesh_rec to mesh
            error = (d12 + d21) * sampling_rate
            if error is not None and not np.isnan(error):
                obj_results['distances'][lmax] = error 
        return obj_results
    except Exception as e:
        logger.debug(f"Failed to process object {obj.name}: {e}")
        return None
        
class ValidateLMAX:
    """ Driver Class for running parallel computation """
    def __init__(self, params):
        self.params = params
        self.client = None
        self.cluster = None
        self.lmax_range = list(
            range(self.params['lmax_min'], self.params['lmax_max'], 4)
        )
        
    def run_analysis(self, mask_path, plot_results=False):
        try:
            start_time = time.time()
            if self.params.get('use_dask', True):
                self._setup_dask()
            
            self.mask_path = mask_path
            mask_array, df_bboxes = self._load_data(mask_path)
            object_errors = self.run_computation(mask_array, df_bboxes)
            logger.info(f"Computed errors for {len(object_errors)} objects.")
            if plot_results:
                self.plot_reconstruction_error(object_errors, reference_lmax=None)
            end_time = time.time()
            logger.info(f"Time taken: {end_time - start_time} seconds")
            return 
        finally:
            # Always cleanup
            if self.client:
                self._shutdown_dask()

    def _load_data(self, mask_path):
        """Load objects from mask."""
        logger.info("Loading data and finding objects...")
        mask_array = fe.load_n5_zarr_array(mask_path)
        
        # Find objects using sparse representation
        chunk_shape = tuple(c[0] for c in mask_array.chunks)
        meta_block = sparse.COO.from_numpy(np.zeros(chunk_shape, dtype=mask_array.dtype))
        mask_sparse = mask_array.map_blocks(
            fe.to_sparse, 
            dtype=mask_array.dtype,  
            meta=meta_block, 
            chunks=mask_array.chunks
        )
        df_bboxes = fe.find_objects(mask_sparse).compute()
        df_bboxes = pd.DataFrame(df_bboxes)
        logger.info(f"Found {len(df_bboxes)} total objects")
        return mask_array, df_bboxes
    
    def run_computation(self, mask_array, df_bboxes):
        """Sample and create delayed processing tasks for objects."""
        logger.info("Creating delayed tasks for object pre-processing...")
        batch_size = self.params['batch_size']
        n_sample = self.params['sample_size']
        # sample = df_bboxes.sample(n_sample, random_state=42)
        sample = df_bboxes.sample(n_sample)
        logger.info(f"Sampled {len(sample)} objects")
        n_batches = len(sample) // batch_size

        batch, results = [], []
        batch_idx = 0
        for i in range(1, len(sample)):
            obj = sample.iloc[i]
            slice_z, slice_y, slice_x = obj[0], obj[1], obj[2]
            mask_slice = mask_array[slice_z, slice_y, slice_x]
            batch.append(
                delayed(compute_mean_distances)(
                    obj, 
                    mask_slice,
                    self.params['min_size'],
                    self.lmax_range,
                    self.params['sampling_rate']
                )
            )
            if len(batch) >= batch_size:
                batch_idx += 1
                logger.info(f"Processing batch {batch_idx} of {n_batches}")
                results.extend(compute(*batch, sync=True))
                batch = []
        
        if batch:
            results.extend(compute(*batch, sync=True))
        logger.info(f"Found {len(results)} objects")
        return [r for r in results if r is not None]

    def plot_reconstruction_error(self, object_errors, reference_lmax=None):        
        # --- Data Prep ---
        plot_data = []
        for obj in object_errors:
            obj_id = obj['id']
            for lmax, error in obj['distances'].items():
                plot_data.append({
                    'object_id': obj_id,
                    'lmax': lmax,
                    'error': error
                })
        
        df_plot = pd.DataFrame(plot_data)
        df_plot.to_csv(f'{self.params["save_dir"]}/object_reconstruction_errors.csv', index=False)
        logger.info(f"Saved object reconstruction errors to {self.params['save_dir']}/object_reconstruction_errors.csv")

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        
        # --- Individual Object Curves in Gray ---
        for obj_id in df_plot['object_id'].unique():
            obj_data = df_plot[df_plot['object_id'] == obj_id].sort_values('lmax')
            x_values = (obj_data['lmax'] + 1) ** 2
            ax.plot(x_values, 
                    obj_data['error'], 
                    '-', 
                    color='gray', 
                    linewidth=0.2, 
                    alpha=0.5)
        
        # --- Aggregated Statistics ---
        df_agg = (df_plot.groupby('lmax')['error']
                  .agg(['mean', 'std'])
                  .reset_index()
                  .sort_values('lmax'))
        x_agg = (df_agg['lmax'] + 1) ** 2

        # Mean line
        ax.plot(x_agg, 
                df_agg['mean'], 
                '-', 
                color='k', 
                linewidth=2, 
                label='Mean')
        
        # Standard deviation band
        ax.fill_between(x_agg, 
                       df_agg['mean'] - df_agg['std'],
                       df_agg['mean'] + df_agg['std'],
                       alpha=0.3, color='black', label='±1 STD')
        
        # --- Axis Formatting ---
        ax.set_yscale('log')
        # ax.set_ylim(0.1, 10.0)
        max_lmax = df_plot['lmax'].max()
        ax.set_xlim(1, (max_lmax + 1) ** 2)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 8, 10])
        ax.set_yticklabels(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0', '10.0'])
    
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # --- Reference Line and Title ---
        if reference_lmax is not None:
            x_ref = (reference_lmax + 1) ** 2
            ax.axvline(x=x_ref, 
                       color='red', 
                       linestyle='--', 
                       alpha=0.7, 
                      label=f'L={reference_lmax}')

        title = f"Reconstruction Error vs L (n={self.params['sample_size']})"
    
        # --- Final Formatting ---
        ax.set_title(title)
        ax.set_xlabel('L (SHE order)')
        ax.set_ylabel('Mean distance to closest point (μm)')
        plt.tight_layout()
        
        # --- Save Plot ---
        save_path = f'{self.params["save_dir"]}/reconstruction_error.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path)
        return fig

    def _setup_dask(self):
        """Setup Dask SGE cluster for distributed processing."""
        try:
            logger.info("Setting up distributed Dask cluster...")
            self.cluster, self.client = setup_dask_sge_cluster(
                n_workers=self.params.get('num_workers', 1),
                cores=self.params.get('cpu_cores', 4),
                processes=self.params.get('cpu_processes', 1),
                memory=self.params.get('cpu_memory', '60G'),
                project=self.params.get('project', 'beliveaulab'),
                queue=self.params.get('queue', 'beliveau-long.q'),
                runtime=self.params.get('runtime', '7200'),
                resource_spec=self.params.get('cpu_resource_spec', 'mfree=60G'),
                log_directory=self.params.get('log_dir', None),
                conda_env=self.params.get('conda_env', 'otls-pipeline'),
                dashboard_port=self.params.get('dashboard_port', None)
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
    mask_path = '/net/beliveau/vol2/instrument/E9.5_290/Zoom_290_subset_test/dataset_fused_masks_cpsamr5.zarr'
    params = {
    # -- dask parameters --
    "num_workers": 2,
    "cpu_memory": "256G", 
    "cpu_cores": 8,
    "cpu_processes": 16, # 2 proc per core
    "cpu_resource_spec": "mfree=16G",  # RAM/worker = RAM/core * cores/worker (16G/core * 1 core/2 proc = 8G/proc)
    # -- computation parameters --
    "sample_size": 1200,
    "sampling_rate": 2.752,
    "min_size": 4000,
    "lmax_min": 4,
    "lmax_max": 40,
    "batch_size": 100,
    "n_iter": 1000,
    # -- tracking parameters --
    'use_dask': True,
    "log_dir" : '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/choose_lmax/',
    "save_dir": '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/figures/',
    "dashboard_port": ":41263"
    }
    ValidateLMAX(params).run_analysis(mask_path, plot_results=True)

if __name__ == "__main__":
    main()