""" This script will be used to select an optimal lmax value via bootstrapping.

The script will:
1. Load the mask and find objects
2. For each lmax value:
    a. Compute the spherical harmonics coefficients for each object for each lmax value (computationally expensive)
    b. Compute the reconstruction error for each object using the mean distance between the original and reconstructed meshes
        - code taken from Allen
    c. store in a dictionary
3. Run bootstrap test on precomputed errors. 
4. Find elbow point and return final lmax value

Author: Madison Sanchez-Forman
Affiliation: University of Washington - Beliveau Labs
"""
# -- standard library --
import logging

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
        aligned_obj, _ = align.align_object(label_slice, {})
        for lmax in lmax_range:
            (_, grid_rec), (_, mesh, _, _) = shparam.get_shcoeffs(
                aligned_obj, lmax=lmax, alignment_2d=False
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
        
class BootstrapLMAX:
    def __init__(self, params):
        self.params = params
        self.client = None
        self.cluster = None
        self.lmax_range = list(
            range(self.params['lmax_min'], self.params['lmax_max'], 4)
        )
        
    def select_optimal_lmax(self, mask_path):
        try:
            # Setup distributed computing
            if self.params.get('use_dask', True):
                self._setup_dask()
            
            self.mask_path = mask_path

            mask_array, df_bboxes = self._load_data(mask_path)
            object_errors = self.run_computation(mask_array, df_bboxes)
            if len(object_errors) < MIN_RELIABLE_OBJECTS:
                logger.warning(f"Only {len(object_errors)} objects found. Results may be unreliable.")
            logger.info(f"Computed errors for {len(object_errors)} objects. Running bootstrap test...")

            # Run bootstrap test
            results = self.run_bootstrap(object_errors)
            chosen_lmax = self.get_elbow_point(results)
            logger.info(f"Chosen lmax: {chosen_lmax}")
            self.plot_results(results, chosen_lmax)
            return chosen_lmax
        
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
        sample = df_bboxes.sample(n_sample, random_state=42)
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

    def run_bootstrap(self, object_errors):
        """Fixed bootstrap selection."""
        error_dict = {obj['id']: obj['distances'] for obj in object_errors}
        object_ids = list(error_dict.keys())
        if len(object_ids) < MIN_RELIABLE_OBJECTS:
            logger.warning("Too few objects for reliable bootstrap")

        results = {}
        for lmax in self.lmax_range:
            bootstrap_errors = []
            
            # objects that have data for this lmax
            valid_errors = [
                obj_id for obj_id in object_ids 
                if lmax in error_dict[obj_id]
            ]
            
            if len(valid_errors) < 10:
                logger.warning(f"Too few objects for lmax {lmax}")
                continue
                
            sample_errors = []
            for _ in range(self.params['n_iter']):
                bootstrap_sample = np.random.choice(valid_errors, size=len(valid_errors), replace=True)                
                sample_errors = [error_dict[obj_id][lmax] * self.params['sampling_rate'] 
                                for obj_id in bootstrap_sample]
                bootstrap_errors.append(np.mean(sample_errors))

            results[lmax] = {
                'mean_distance': np.mean(bootstrap_errors),
                'std_distance': np.std(bootstrap_errors),
                'bootstrap_errors': bootstrap_errors,  # Store for significance testing
                'n_objects': len(valid_errors),
                'confidence_interval': {
                    'lower': np.percentile(bootstrap_errors, 2.5),
                    'upper': np.percentile(bootstrap_errors, 97.5),
                    'level': 0.95
                }
            }
        return results
    
    def get_elbow_point(self, bootstrap_results, use_confidence_intervals=False):
        lmax_values = np.array(sorted(bootstrap_results.keys()))
        if use_confidence_intervals:
            errors = np.array([bootstrap_results[lmax]['confidence_interval']['lower'] for lmax in lmax_values])
        else:
            errors = np.array([bootstrap_results[lmax]['mean_distance'] for lmax in lmax_values])
        
        if len(lmax_values) < 3:
            logger.warning("Need at least 3 lmax values for second derivative")
            return lmax_values[0]
        
        first_derivative = np.gradient(errors, lmax_values)
        second_derivative = np.gradient(first_derivative, lmax_values)
        
        # Note: We want the most positive because errors are decreasing,
        # so we're looking for where the rate of decrease slows down most
        elbow_idx = np.argmax(second_derivative)
        chosen_lmax = lmax_values[elbow_idx]
        return chosen_lmax
        
    def plot_results(self, results, chosen_lmax):
        """Plot bootstrap results with confidence intervals."""
        plot_data = []
        for lmax, stats in results.items():
            plot_data.append({
                'lmax': lmax,
                'mean_distance': stats['mean_distance'],
                'lower_ci': stats['confidence_interval']['lower'],
                'upper_ci': stats['confidence_interval']['upper']
            })
        
        df = pd.DataFrame(plot_data)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, 
                     x='lmax', 
                     y='mean_distance', 
                     marker='o', 
                     linewidth=2, 
                     markersize=8)
        
        # Add confidence intervals
        plt.fill_between(df['lmax'], 
                        df['lower_ci'], 
                        df['upper_ci'], 
                        alpha=0.3, 
                        label='95% CI')

        # Highlight chosen lmax
        # plt.axvline(chosen_lmax, 
        #             color='red', 
        #             linestyle='--', 
        #             linewidth=2,
        #             label=f'Chosen lmax = {chosen_lmax}')
        plt.xlabel('lmax (Number of Coefficients)', fontsize=12)
        plt.ylabel('Mean Distance (μm)', fontsize=12)
        plt.title('Reconstruction Error', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{self.params["save_dir"]}/bootstrap_results.png')

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
    "num_workers": 1,
    "cpu_memory": "512G", 
    "cpu_cores": 16,
    "cpu_processes": 32,
    "cpu_resource_spec": "mfree=32G",  
    # -- computation parameters --
    "sample_size": 1500,
    "sampling_rate": 2.752,
    "min_size": 4000,
    "lmax_min": 4,
    "lmax_max": 80,
    "batch_size": 100,
    "n_iter": 1000,
    # -- tracking parameters --
    'use_dask': True,
    "log_dir" : '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/choose_lmax/',
    "save_dir": '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/',
    "dashboard_port": ":41263"
    }
    bootstrapper = BootstrapLMAX(params)
    results = bootstrapper.select_optimal_lmax(mask_path)

if __name__ == "__main__":
    main()