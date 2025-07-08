

"""
Optimal lmax selection for spherical harmonics feature extraction using Leave-One-Out
Cross-Validation (LOOCV). Adapted for distributed computing with Dask SGE cluster.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import time
import sys
import os
import datetime
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import sparse
import dask.array as da
from aicsshparam import shtools, shparam
from dask import delayed, compute
from scipy.sparse import csr_matrix

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Ensure local modules can be imported
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask
import align_3d as align
import feature_extraction as fe
logger.info("Imported local modules")


# Add a file handler to write logs to a file
def setup_logging(log_dir):
    """Setup logging to write to both file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"choose_lmax_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")


# --- Constants ---
DEFAULT_LMAX = 12
MIN_RELIABLE_OBJECTS = 30


# --- Helper Functions for Distributed Processing ---

def process_individual(obj, mask_da, min_size):
    """Process a single object: load, filter by size, and align."""
    try:
        slice_z, slice_y, slice_x = obj[0], obj[1], obj[2]
        obj_id = int(obj.name)
        
        mask_slice = mask_da[slice_z, slice_y, slice_x].compute()
        label_slice = np.where(mask_slice == obj_id, mask_slice, 0)
        
        if np.sum(label_slice > 0) < min_size:
            logger.debug(f"Object {obj_id} is too small, skipping.")
            return None
            
        aligned_obj, _ = align.align_object(label_slice, {})
        return {'obj_id': obj_id, 'aligned_obj': aligned_obj}
    except Exception as e:
        logger.warning(f"Failed to process object {obj.name}: {e}")
        return None


def get_coeffs_and_errors(object_batch, lmax_range):
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
                (coeffs_dict, grid_rec), (_, _, grid, _) = shparam.get_shcoeffs(
                    aligned_obj, lmax=lmax, alignment_2d=False
                )
                
                mse = shtools.get_reconstruction_error(grid, grid_rec)
                
                if mse is not None and not np.isnan(mse):
                    obj_results['errors'][lmax] = mse
                    obj_results['coeffs'][lmax] = coeffs_dict
                    
            except Exception as e:
                logger.debug(f"Failed computation for object {obj_id} lmax {lmax}: {e}")
        
        if obj_results['errors']:
            results[obj_id] = obj_results

    return results


# --- Main LOOCV Class ---

class SphericalHarmonicsLOOCV:
    """
    Selects optimal lmax for spherical harmonics using Leave-One-Out Cross-Validation.
    """
    
    def __init__(self, params):
        self.params = params
        self.client = None
        self.cluster = None
        self.lmax_range = list(range(
            self.params['lmax_min'], 
            self.params['lmax_max'] + 1, 
            self.params.get('lmax_step', 4)
        ))
        logger.info(f"Testing lmax range: {self.lmax_range}")
        
    def select_optimal_lmax(self, mask_path: str):
        """
        Main entry point: finds optimal lmax for the sample.
        
        Args:
            mask_path: Path to segmentation mask.
        
        Returns:
            A dictionary containing the optimal lmax and the full CV results.
        """
        start_time = time.time()
        try:
            mask_da, df_bboxes = self._load_data(mask_path)
            processed_objects = self._process_objects_distributed(mask_da, df_bboxes)
            
            if not processed_objects:
                logger.error("No valid objects were processed. Aborting.")
                return self._create_result_dict(DEFAULT_LMAX, {}, 0)

            if len(processed_objects) < MIN_RELIABLE_OBJECTS:
                logger.warning(f"Only {len(processed_objects)} valid objects found. Results may be unreliable.")
            
            all_data = self._precompute_all(processed_objects)
            
            if not all_data:
                logger.error("Could not compute SH coefficients for any objects. Aborting.")
                return self._create_result_dict(DEFAULT_LMAX, {}, len(processed_objects))

            cv_results = self._evaluate_lmax_loocv(all_data)
            optimal_lmax = self._find_optimal_lmax(cv_results)
            
            result = self._create_result_dict(optimal_lmax, cv_results, len(all_data))
            
            elapsed_time = time.time() - start_time
            logger.info(f"Optimization complete in {elapsed_time:.2f}s. Optimal lmax: {optimal_lmax}")
            return result
            
        finally:
            pass  # Dask cluster lifecycle handled entirely in main()

    def _load_data(self, mask_path: str):
        """Load object bounding boxes from mask."""
        logger.info(f"Loading data from: {mask_path}")
        mask_da = fe.load_n5_zarr_array(mask_path)
        
        chunk_shape = tuple(c[0] for c in mask_da.chunks)
        meta_block = sparse.COO.from_numpy(np.zeros(chunk_shape, dtype=mask_da.dtype))
        mask_sparse = mask_da.map_blocks(
            fe.to_sparse, dtype=mask_da.dtype, meta=meta_block, chunks=mask_da.chunks
        )
        df_bboxes = fe.find_objects(mask_sparse).compute()
        df_bboxes = pd.DataFrame(df_bboxes)
        logger.info(f"Found {len(df_bboxes)} total objects")
        return mask_da, df_bboxes
    
    def _process_objects_distributed(self, mask_da, df_bboxes):
        """Sample, load, and align objects using Dask."""
        n_sample = min(self.params['sample_size'], len(df_bboxes))
        if n_sample == 0:
            logger.warning("No objects to sample.")
            return []
        sample = df_bboxes.sample(n_sample, random_state=42)
        logger.info(f"Sampled {len(sample)} objects for processing.")

        delayed_tasks = [
            delayed(process_individual)(obj, mask_da, self.params['min_object_size'])
            for _, obj in sample.iterrows()
        ]
        
        batch_size = self.params.get('batch_size', 100)
        logger.info(f"Processing {len(delayed_tasks)} objects in batches of {batch_size}...")
        all_results = []
        for i in range(0, len(delayed_tasks), batch_size):
            batch = delayed_tasks[i:i + batch_size]
            batch_results = compute(*batch)
            all_results.extend([r for r in batch_results if r is not None])
            logger.info(f"  ... processed batch {i//batch_size + 1}, found {len(all_results)} valid objects so far.")
        
        logger.info(f"Successfully processed {len(all_results)} valid objects.")
        return all_results
    
    def _precompute_all(self, processed_objects):
        """Compute coefficients and errors for all objects and lmax values."""
        logger.info("Pre-computing spherical harmonics coefficients and errors...")
        
        batch_size = self.params.get('error_batch_size', 5)
        delayed_tasks = [
            delayed(get_coeffs_and_errors)(batch, self.lmax_range)
            for i in range(0, len(processed_objects), batch_size)
            if (batch := processed_objects[i:i + batch_size])
        ]
        
        if not delayed_tasks:
            logger.warning("No batches to process for SH pre-computation.")
            return {}

        logger.info(f"Computing data for {len(delayed_tasks)} batches...")
        batch_results = compute(*delayed_tasks)
        
        all_data = {}
        for batch_result in batch_results:
            all_data.update(batch_result)
        
        logger.info(f"Computed data for {len(all_data)} objects.")
        return all_data
    
    def _evaluate_lmax_loocv(self, all_data):
        """Evaluate each lmax using closed-form LOOCV."""
        cv_results = {}
        object_ids = list(all_data.keys())
        
        for lmax in self.lmax_range:
            logger.info(f"Evaluating lmax={lmax}...")
            
            # Check if we have enough samples for this lmax
            n_features = (lmax + 1)**2
            if len(object_ids) <= n_features:
                logger.warning(
                    f"  Skipping lmax={lmax}: Not enough samples ({len(object_ids)}) "
                    f"for the number of features ({n_features})."
                )
                cv_results[lmax] = {'cv_error': float('inf'), 'n_objects': 0, 'reason': 'n <= p'}
                continue

            residuals, coeffs_dicts = [], []
            valid_obj_ids = []
            for obj_id in object_ids:
                obj_data = all_data[obj_id]
                if lmax in obj_data['errors'] and lmax in obj_data['coeffs']:
                    error = obj_data['errors'][lmax] * self.params.get('sampling_rate', 1.0)
                    residuals.append(error)
                    coeffs_dicts.append(obj_data['coeffs'][lmax])
                    valid_obj_ids.append(obj_id)

            if len(valid_obj_ids) <= n_features:
                logger.warning(
                    f"  Skipping lmax={lmax}: Not enough valid objects ({len(valid_obj_ids)}) "
                    f"for the number of features ({n_features})."
                )
                cv_results[lmax] = {'cv_error': float('inf'), 'n_objects': len(valid_obj_ids), 'reason': 'n <= p for valid objects'}
                continue

            X = self._build_design_matrix(coeffs_dicts, lmax)
            cv_error = self._compute_loocv_error(np.array(residuals), X)
            
            cv_results[lmax] = {
                'cv_error': cv_error,
                'n_objects': len(residuals),
                'mean_residual': np.mean(residuals),
                'std_residual': np.std(residuals)
            }
            logger.info(f"  lmax={lmax}: CV error = {cv_error:.6f}, Mean residual = {np.mean(residuals):.6f}")
        return cv_results
    
    def _get_coeff_mapping(self, lmax):
        """Create mapping from coefficient names to column indices."""
        col_map, col_idx = {}, 0
        for L in range(lmax + 1):
            for M in range(L + 1):
                if M == 0:
                    col_map[f"shcoeffs_L{L}M{M}C"] = col_idx
                    col_idx += 1
                else:
                    col_map[f"shcoeffs_L{L}M{M}C"] = col_idx
                    col_map[f"shcoeffs_L{L}M{M}S"] = col_idx + 1
                    col_idx += 2
        return col_map
    
    def _build_design_matrix(self, coeffs_dicts, lmax):
        """Build design matrix from coefficient dictionaries."""
        n_samples = len(coeffs_dicts)
        n_coeffs = (lmax + 1) ** 2
        col_map = self._get_coeff_mapping(lmax)
        rows, cols, data = [], [], []
        for i, coeffs_dict in enumerate(coeffs_dicts):
            for key, value in coeffs_dict.items():
                if key in col_map and abs(value) > 1e-10:
                    rows.append(i)
                    cols.append(col_map[key])
                    data.append(value)
        return csr_matrix((data, (rows, cols)), shape=(n_samples, n_coeffs))
    
    def _compute_hat_diagonal(self, X_sparse, XtX_inv):
        """Computes the diagonal of the hat matrix H = X(X'X)^-1X'."""
        hat_diagonal = np.zeros(X_sparse.shape[0])
        for i in range(X_sparse.shape[0]):
            x_i = X_sparse.getrow(i)
            hat_diagonal[i] = (x_i @ XtX_inv @ x_i.T).toarray()[0, 0]
        return hat_diagonal
    
    def _compute_loocv_error(self, residuals, X_sparse):
        """Compute LOOCV error using closed-form solution with pseudo-inverse."""
        try:
            XtX = X_sparse.T @ X_sparse
            XtX_dense = XtX.toarray()
            XtX_inv = np.linalg.pinv(XtX_dense)
            
            H_diag = self._compute_hat_diagonal(X_sparse, XtX_inv)
            
            if np.any(H_diag >= 1.0 - 1e-9):
                logger.warning("Numerical instability detected in hat matrix (H_diag >= 1).")
                return float('inf')
            
            loocv_residuals = residuals / (1 - H_diag)
            return np.mean(loocv_residuals ** 2)
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error during LOOCV computation: {e}")
            return float('inf')
    
    def _find_optimal_lmax(self, cv_results):
        """Find optimal lmax from CV results."""
        valid_results = {
            lmax: res['cv_error'] 
            for lmax, res in cv_results.items() 
            if res.get('cv_error') is not None and np.isfinite(res['cv_error'])
        }
        if not valid_results:
            logger.warning(f"No valid CV results found. Falling back to default lmax: {DEFAULT_LMAX}")
            return DEFAULT_LMAX
        
        optimal_lmax = min(valid_results, key=valid_results.get)
        logger.info(f"Optimal lmax selected: {optimal_lmax} with CV error: {valid_results[optimal_lmax]:.6f}")
        return int(optimal_lmax)
    
    def _create_result_dict(self, optimal_lmax, cv_results, n_objects):
        """Create a comprehensive, JSON-serializable result dictionary."""
        # Convert numpy types to native Python types for JSON serialization
        serializable_cv_results = {}
        for lmax, data in cv_results.items():
            serializable_cv_results[str(lmax)] = {k: (float(v) if isinstance(v, (np.number, np.bool_)) else v) for k, v in data.items()}

        return {
            'optimal_lmax': int(optimal_lmax),
            'loocv_results': serializable_cv_results,
            'n_objects_used': int(n_objects),
            'parameters': {
                'sample_size': self.params['sample_size'],
                'lmax_range': [self.params['lmax_min'], self.params['lmax_max']],
                'min_object_size': self.params['min_object_size'],
                'sampling_rate': self.params.get('sampling_rate', 1.0)
            }
        }


# --- Execution Control ---

def save_results(result, cache_file):
    """Save the final result dictionary to a JSON file."""
    try:
        path = Path(cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(f"Results successfully saved to {cache_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {cache_file}: {e}")


def create_default_result(cache_file):
    """Create and save a default result if optimization is disabled or fails."""
    logger.warning(f"Lmax optimization disabled or failed. Using default value: {DEFAULT_LMAX}")
    result = {
        'optimal_lmax': DEFAULT_LMAX, 
        'loocv_results': {}, 
        'method': 'default_fallback',
        'n_objects_used': 0
    }
    save_results(result, cache_file)
    return result


def parse_arguments():
    """Parse command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Optimal LMAX selection using LOOCV.")
    parser.add_argument("--mask_path", required=True, help="Path to segmentation mask Zarr array.")
    parser.add_argument("--cache_file", required=True, help="Path to output JSON cache file.")
    
    # Processing parameters
    parser.add_argument("--sample_size", type=int, default=300, help="Number of objects to sample for optimization.")
    parser.add_argument("--lmax_min", type=int, default=4, help="Minimum lmax value to test.")
    parser.add_argument("--lmax_max", type=int, default=16, help="Maximum lmax value to test.")
    parser.add_argument("--lmax_step", type=int, default=4, help="Step size for lmax range.")
    parser.add_argument("--min_object_size", type=int, default=100, help="Minimum size of objects to consider.")
    parser.add_argument("--sampling_rate", type=float, default=1.0, help="Sampling rate used for reconstruction error.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing individual objects.")
    parser.add_argument("--error_batch_size", type=int, default=5, help="Batch size for computing SH errors and coefficients.")

    # Dask SGE Cluster Arguments
    parser.add_argument("--use_dask", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable Dask for distributed computing.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Dask workers (SGE jobs).")
    parser.add_argument("--cores_per_worker", type=int, default=1, help="Number of cores per worker.")
    parser.add_argument("--mem_per_worker", default="10G", help="Memory per worker (e.g., '10G').")
    parser.add_argument("--project", help="SGE project for Dask.", default="beliveaulab")
    parser.add_argument("--queue", help="SGE queue for Dask.", default="beliveau-long.q")
    parser.add_argument("--log_dir", default=None, help="Directory for Dask worker logs.")
    parser.add_argument("--conda_env", default="otls-pipeline", help="Conda environment to activate on workers.")
    parser.add_argument("--processes", type=int, default=2, help="Number of Python processes per worker.")
    parser.add_argument("--runtime", default="4:00:00", help="Job runtime (SGE format or seconds).")
    parser.add_argument("--resource_spec", default=None, help="SGE resource specification (e.g., 'mfree=32G')")
    
    return parser.parse_args()


def main():
    """Main entry point for Snakemake or standalone execution."""
    
    # Check if running under Snakemake
    if 'snakemake' in globals():
        if not snakemake.params.get('enabled', True):
            create_default_result(snakemake.output.lmax_cache)
            return
            
        params = {
            'mask_path': snakemake.input.zarr,
            'cache_file': snakemake.output.lmax_cache,
            'sample_size': snakemake.params.sample_size,
            'lmax_min': snakemake.params.lmax_min,
            'lmax_max': snakemake.params.lmax_max,
            'lmax_step': snakemake.params.lmax_step,
            'min_object_size': snakemake.params.min_object_size,
            'dask_config': {
                'n_workers': snakemake.resources.num_workers,
                'cores': snakemake.resources.get('cores_per_worker', 1),
                'memory': snakemake.resources.get('mem_per_worker', '10G'),
                'project': snakemake.resources.get('project', 'beliveaulab'),
                'queue': snakemake.resources.get('queue', 'beliveau-long.q'),
                'processes': snakemake.resources.get('processes', 1),
                'runtime': snakemake.resources.get('runtime', '4:00:00'),
                'resource_spec': snakemake.resources.get('resource_spec', None),
                'conda_env': snakemake.resources.get('conda_env', 'otls-pipeline'),
                'log_directory': snakemake.params.log_dir
            }
        }
    else:
        # Standalone mode
        args = parse_arguments()
        params = vars(args)
        params['dask_config'] = {
            'n_workers': args.num_workers,
            'cores': args.cores_per_worker,
            'memory': args.mem_per_worker,
            'project': args.project,
            'queue': args.queue,
        }

    # Setup Dask cluster
    cluster, client = setup_dask_sge_cluster(**params['dask_config'])
    
    try:
        # Run optimization
        optimizer = SphericalHarmonicsLOOCV(params)
        result = optimizer.select_optimal_lmax(params['mask_path'])
        save_results(result, params['cache_file'])
    finally:
        shutdown_dask(cluster, client)


if __name__ == "__main__":
    main()
# def main():
#     """Main entry point for Snakemake or standalone execution."""
#     try:
#         if 'snakemake' in globals():
#             logger.info("Running under Snakemake, using snakemake object for parameters.")
            
#             if not snakemake.params.get('enabled', True):
#                 create_default_result(snakemake.output.lmax_cache)
#                 return

#             # Setup logging
#             setup_logging(Path("logs") / "choose_lmax")

#             params = {
#                 'mask_path': snakemake.input.zarr,
#                 'cache_file': snakemake.output.lmax_cache,
#                 'sample_size': snakemake.params.sample_size,
#                 'lmax_min': snakemake.params.lmax_min,
#                 'lmax_max': snakemake.params.lmax_max,
#                 'lmax_step': snakemake.params.lmax_step,
#                 'min_object_size': snakemake.params.min_object_size,
#                 'sampling_rate': snakemake.params.get('sampling_rate', 1.0),
#                 'batch_size': snakemake.params.get('batch_size', 100),
#                 'error_batch_size': snakemake.params.get('error_batch_size', 5),
#                 'use_dask': True,
#                 'dask_config': {
#                     'n_workers': snakemake.resources.num_workers,
#                     'cores': snakemake.resources.get('cores_per_worker', 1),
#                     'memory': snakemake.resources.get('mem_per_worker', '10G'),
#                     'project': snakemake.resources.get('project', 'beliveaulab'),
#                     'queue': snakemake.resources.get('queue', 'beliveau-long.q'),
#                     'log_directory': str(Path("logs") / "dask" / f"choose_lmax_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
#                     'processes': snakemake.resources.get('processes', 2),
#                     'runtime': snakemake.resources.get('runtime', '4:00:00'),
#                     'resource_spec': snakemake.resources.get('resource_spec', None),
#                     'conda_env': snakemake.conda_env_name if hasattr(snakemake, 'conda_env_name') else "otls-pipeline"
#                 }
#             }
        
#         else:
#             logger.info("Running in standalone mode.")
#             args = parse_arguments()
            
#             # Setup logging
#             setup_logging(Path(args.log_dir) / "choose_lmax" if args.log_dir else Path("logs") / "choose_lmax")
            
#             params = vars(args)
#             params['dask_config'] = {
#                 'n_workers': args.num_workers,
#                 'cores': args.cores_per_worker,
#                 'memory': args.mem_per_worker,
#                 'project': args.project,
#                 'queue': args.queue,
#                 'log_directory': str(Path(args.log_dir) / "dask" / f"choose_lmax_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") if args.log_dir else str(Path("logs") / "dask" / f"choose_lmax_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
#                 'conda_env': args.conda_env,
#                 "processes": args.processes,
#                 "runtime": args.runtime,
#                 "resource_spec": args.resource_spec,
#             }

#         # --- Dask Cluster Setup ---
#         cluster = None
#         client = None
#         if params.get('use_dask', True):
#             try:
#                 logger.info("Setting up Dask cluster...")
#                 dc = params['dask_config']
#                 cluster, client = setup_dask_sge_cluster(
#                     n_workers=dc['n_workers'],
#                     cores=dc['cores'],
#                     processes=dc.get('processes', 1),
#                     memory=dc['memory'],
#                     project=dc['project'],
#                     queue=dc['queue'],
#                     runtime=dc['runtime'],
#                     resource_spec=dc.get('resource_spec'),
#                     log_directory=dc.get('log_directory'),
#                     conda_env=dc.get('conda_env', 'otls-pipeline')
#                 )
#                 logger.info(f"Dask dashboard link: {client.dashboard_link}")
#             except Exception as e:
#                 logger.error(f"Failed to set up Dask cluster: {e}", exc_info=True)
#                 sys.exit(1)

#         # --- Run Optimization ---
#         try:
#             logger.info(f"Starting lmax optimization with parameters: {params}")
#             optimizer = SphericalHarmonicsLOOCV(params)
#             result = optimizer.select_optimal_lmax(params['mask_path'])
#             save_results(result, params['cache_file'])
#         finally:
#             if cluster and client:
#                 shutdown_dask(cluster, client)

#     except Exception as e:
#         logger.critical(f"A critical error occurred in the main execution block: {e}", exc_info=True)
        
#         cache_file = None
#         if 'snakemake' in globals() and hasattr(snakemake, 'output') and hasattr(snakemake.output, 'lmax_cache'):
#             cache_file = snakemake.output.lmax_cache
#         elif 'params' in locals() and 'cache_file' in params:
#             cache_file = params['cache_file']
            
#         if cache_file:
#             create_default_result(cache_file)
#         else:
#             logger.error("Could not determine cache file path to save default result.")
            
#         sys.exit(1)


if __name__ == "__main__":
    main()
