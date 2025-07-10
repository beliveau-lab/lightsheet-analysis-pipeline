import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from dask import delayed, compute
import sparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Import your modules
import align_3d as align
import feature_extraction as fe
from aicsshparam import shtools, shparam
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_LMAX = 12
MIN_RELIABLE_OBJECTS = 30

def process_individual(obj, mask_da, min_size):
    """Process a single object in distributed manner."""
    try:
        slice_z, slice_y, slice_x = obj[0], obj[1], obj[2]
        obj_id = int(obj.name)
        
        mask_slice = mask_da[slice_z, slice_y, slice_x].compute()
        label_slice = np.where(mask_slice == obj_id, mask_slice, 0)
        
        if np.sum(label_slice > 0) < min_size:
            logger.debug(f"Object {obj_id} is too small")
            return None
            
        aligned_obj, _ = align.align_object(label_slice, {})
        return {'obj_id': obj_id, 'aligned_obj': aligned_obj}
    except Exception as e:
        logger.debug(f"Failed to process object {obj.name}: {e}")
        return None


def precompute_errors(object_batch, lmax_range):
    """Compute both reconstruction errors and coefficients for a batch of objects."""
    results = {}
    
    for obj_data in object_batch:
        if obj_data is None:
            continue
            
        obj_id = obj_data['obj_id']
        aligned_obj = obj_data['aligned_obj']
        obj_results = {'errors': {}, 'coeffs': {}}
        
        for lmax in lmax_range:
            try:
                # Get coefficients and reconstruction in one call
                (_, grid_rec), (_, _, grid, _) = shparam.get_shcoeffs(
                    aligned_obj, lmax=lmax, alignment_2d=False
                )
                # Compute reconstruction error
                mse = shtools.get_reconstruction_error(grid, grid_rec)
                if mse is not None and not np.isnan(mse):
                    obj_results['errors'][lmax] = mse                    
            except Exception as e:
                logger.debug(f"Failed computation for object {obj_id} lmax {lmax}: {e}")
        
        if obj_results['errors']:  # If we got at least one valid result
            results[obj_id] = obj_results

    return results


class BootstrapLmaxSelection:
    """
    Leave-One-Out Cross-Validation for selecting optimal lmax for spherical harmonics
    decomposition of 3D objects from a mouse brain sample.
    """
    
    def __init__(self, params):
        self.params = params
        self.client = None
        self.cluster = None
        self.lmax_range = list(range(self.params['lmax_min'], 
                                     self.params['lmax_max'], 
                                     4))
        
    def select_optimal_lmax(self, mask_path):
        """
        Main entry point: finds optimal lmax for this mouse sample.
        
        Args:
            mask_path: Path to segmentation mask
            n5_path: Path to n5 image data (optional, not used currently)
        
        Returns:
            optimal_lmax: Best lmax value
            cv_results: Cross-validation results for each lmax
        """
        try:
            # Setup distributed computing
            if self.params.get('use_dask', True):
                self._setup_dask()
            
            # Load and process objects
            mask_da, df_bboxes = self._load_data(mask_path)
            processed_objects = self._process_objects_vanilla(mask_da, df_bboxes)
            
            if len(processed_objects) < MIN_RELIABLE_OBJECTS:
                logger.warning(f"Only {len(processed_objects)} valid objects found. Results may be unreliable.")
            
            # Compute coefficients and errors for all lmax values
            all_data = self._precompute_all(processed_objects)
            print(all_data)

            
        finally:
            # Always cleanup
            if self.client:
                self._shutdown_dask()
    
    def _load_data(self, mask_path):
        """Load objects from mask."""
        logger.info("Loading data and finding objects...")
        mask_da = fe.load_n5_zarr_array(mask_path)
        
        # Find objects using sparse representation
        chunk_shape = tuple(c[0] for c in mask_da.chunks)
        meta_block = sparse.COO.from_numpy(np.zeros(chunk_shape, dtype=mask_da.dtype))
        mask_sparse = mask_da.map_blocks(
            fe.to_sparse, 
            dtype=mask_da.dtype, 
            meta=meta_block, 
            chunks=mask_da.chunks
        )
        df_bboxes = fe.find_objects(mask_sparse).compute()
        df_bboxes = pd.DataFrame(df_bboxes)
        logger.info(f"Found {len(df_bboxes)} total objects")
        return mask_da, df_bboxes
    
    def _process_objects_vanilla(self, mask_da, df_bboxes):
        n_sample = min(self.params['sample_size'], len(df_bboxes))
        sample = df_bboxes.sample(n_sample)
        logger.info(f"Sampled {len(sample)} objects")

        all_objects = []
        for _, obj in sample.iterrows():
            obj_data = process_individual(obj, mask_da, self.params['min_object_size'])
            if obj_data is not None:
                all_objects.append(obj_data)
            else: 
                logger.info("failed to process object")
        if len(all_objects) == 0:
            logger.debug("No valid objects found")
            return None
        
        logger.info(f"Successfully processed {len(all_objects)} valid objects")
        return all_objects
    
    # def _process_objects_batch(self, mask_da, df_bboxes):
    #     """Sample and create delayed processing tasks for objects."""
    #     logger.info("Creating delayed tasks for object pre-processing...")
        
    #     n_sample = min(self.params['sample_size'], len(df_bboxes))
    #     sample = df_bboxes.sample(n_sample, random_state=42)  # Add random_state for reproducibility
    #     logger.info(f"Sampled {len(sample)} objects")
        
    #     # Create delayed tasks for object processing
    #     delayed_tasks = []
    #     for _, obj in sample.iterrows():
    #         delayed_task = delayed(process_individual)(
    #             obj, mask_da, self.params['min_object_size']
    #         )
    #         delayed_tasks.append(delayed_task)
        
    #     logger.info(f"Created {len(delayed_tasks)} delayed object processing tasks.")
    #     return delayed_tasks
    
    def _precompute_all(self, processed_objects):
        """Compute coefficients and errors for all objects and lmax values."""
        lmax_range = range(self.params['lmax_min'], self.params['lmax_max'] + 1)
        logger.info("Computing spherical harmonics data...")
        
        # Create delayed tasks for batches
        batch_size = self.params.get('error_batch_size', 20)
        delayed_tasks = []
        
        for i in range(0, len(processed_objects), batch_size):
            batch = processed_objects[i:i + batch_size]
            task = delayed(precompute_errors)(batch, lmax_range)
            delayed_tasks.append(task)
        
        # Execute all tasks
        logger.info(f"Computing data for {len(delayed_tasks)} batches...")
        batch_results = compute(*delayed_tasks)
        
        # Combine results
        all_data = {}
        for batch_result in batch_results:
            all_data.update(batch_result)
        
        logger.info(f"Computed data for {len(all_data)} objects")
        print(all_data)
        return all_data
   
    # def bootstrap_lmax_selection(self, all_data, n_bootstrap=100):
    #     """Fixed bootstrap selection."""
        
    #     # Get valid object IDs
    #     valid_objects = list(all_data.keys())
        
    #     if len(valid_objects) < 30:
    #         logger.warning("Too few objects for reliable bootstrap")
        
    #     lmax_stats = {}
        
    #     for lmax in self.lmax_range:
    #         bootstrap_errors = []
            
    #         # Get objects that have data for this lmax
    #         objects_with_lmax = [
    #             obj_id for obj_id in valid_objects 
    #             if lmax in all_data[obj_id]['errors']
    #         ]
            
    #         if len(objects_with_lmax) < 10:
    #             logger.warning(f"Too few objects for lmax {lmax}")
    #             continue
                
    #         for _ in range(n_bootstrap):
    #             # Correct bootstrap sampling
    #             bootstrap_sample = np.random.choice(
    #                 objects_with_lmax, 
    #                 size=len(objects_with_lmax), 
    #                 replace=True
    #             )
                
    #             sample_errors = []
    #             for obj_id in bootstrap_sample:
    #                 error = all_data[obj_id]['errors'][lmax]
    #                 sample_errors.append(error)
                
    #             bootstrap_errors.append(np.mean(sample_errors))
            
    #         lmax_stats[lmax] = {
    #             'mean_error': np.mean(bootstrap_errors),
    #             'std_error': np.std(bootstrap_errors),
    #             'n_objects': len(objects_with_lmax)
    #         }
        
    #         return min(lmax_stats.keys(), key=lambda k: lmax_stats[k]['mean_error'])
    

    # def _setup_dask(self):
    #     """Setup Dask SGE cluster for distributed processing."""
    #     try:
    #         logger.info("Setting up distributed Dask cluster...")
    #         self.cluster, self.client = setup_dask_sge_cluster(
    #             n_workers=self.params.get('num_workers', 8),
    #             cores=self.params.get('cores_per_worker', 2),
    #             processes=self.params.get('processes', 1),
    #             memory=self.params.get('mem_per_worker', '30G'),
    #             project=self.params.get('project', 'beliveaulab'),
    #             queue=self.params.get('queue', 'beliveau-long.q'),
    #             runtime=self.params.get('runtime', '7200'),
    #             resource_spec=self.params.get('resource_spec', 'mfree=30G'),
    #             log_directory=self.params.get('log_dir', None),
    #             conda_env=self.params.get('conda_env', 'otls-pipeline')
    #         )
    #         logger.info(f"Dask dashboard link: {self.client.dashboard_link}")
    #         return True
    #     except Exception as e:
    #         logger.error(f"Failed to setup distributed cluster: {e}")
    #         return False

    # def _shutdown_dask(self):
    #     """Shutdown Dask cluster."""
    #     if self.cluster and self.client:
    #         try:
    #             shutdown_dask(self.cluster, self.client)
    #             logger.info("Dask cluster shut down successfully")
    #         except Exception as e:
    #             logger.error(f"Error shutting down cluster: {e}")