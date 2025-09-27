# This script should take as input the cell x feature .csv and a spline_points.csv (made by the user) and return the .csv with transformed coordinates
# 1. spin up a CPU dask cluster and run the function in batches? 
# 

# Prevent BLAS/OpenMP oversubscription (must be set before heavy imports)

import os


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from scipy.optimize import minimize_scalar
import argparse
import multiprocessing
from functools import partial

# parse command-line arguments
parser = argparse.ArgumentParser(description="Transform coordinates using spline points")
parser.add_argument('--csv', required=True, help='Path to the cell features CSV file')
parser.add_argument('--spline', required=True, help='Path to the spline points CSV file')
parser.add_argument('--ds', required=True, help='Downsample level the spline points were drawn on')

args = parser.parse_args()

# LOAD SPLINE POINTS AND CSV
base_dir = os.path.dirname(args.csv)
print(f"Loading spline points from {args.spline}")
spline_points = pd.read_csv(args.spline)

print(f"Loading csv from {args.csv}")
cell_df = pd.read_csv(args.csv)

cell_df_subset = cell_df.sample(10000)

ds_level = int(args.ds)

# GET SPLINE POINT COORDS
z = spline_points['axis-0'].values
y = spline_points['axis-1'].values
x = spline_points['axis-2'].values

# SCALE SPLINE POINT COORDS TO s0 RESOLUTION

x = x*ds_level
y = y*ds_level
z = z*ds_level

cell_centroids_x = cell_df['centroid_x'].values
cell_centroids_y = cell_df['centroid_y'].values
cell_centroids_z = cell_df['centroid_z'].values


cell_centroids_x_subset = cell_df_subset['centroid_x'].values
cell_centroids_y_subset = cell_df_subset['centroid_y'].values
cell_centroids_z_subset = cell_df_subset['centroid_z'].values

# Fit a 3D spline
tck, u_knots = interpolate.splprep([x, y, z], s=2)

# Evaluate spline points for plotting
u_fine = np.linspace(0, 1, 200) # Increased points for smoother spline visualization
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

fig_spline = plt.figure(figsize=(10, 8))
ax_spline = fig_spline.add_subplot(111, projection='3d')
ax_spline.plot(x, y, z, 'o', label='Original points', alpha=0.6)
ax_spline.plot(x_fine, y_fine, z_fine, '-', label='Fitted spline', linewidth=2)
ax_spline.set_xlabel('X')
ax_spline.set_ylabel('Y')
ax_spline.set_zlabel('Z')
ax_spline.legend()
plt.title('3D Spline Fit to Points')
plt.savefig(os.path.join(base_dir, 'spline_fit.png'), dpi=300)

# --- Helper function to calculate Frenet frames ---
def calculate_frenet_frames(u_values, tck_spline):
    u_values = np.asarray(u_values)
    is_scalar_input = (u_values.ndim == 0)
    if is_scalar_input:
        u_values_arr = u_values[np.newaxis]
    else:
        u_values_arr = u_values

    s_prime = np.array(interpolate.splev(u_values_arr, tck_spline, der=1)).T
    s_double_prime = np.array(interpolate.splev(u_values_arr, tck_spline, der=2)).T
    
    if s_prime.ndim == 1: s_prime = s_prime[np.newaxis,:]
    if s_double_prime.ndim == 1: s_double_prime = s_double_prime[np.newaxis,:]

    norms_s_prime = np.linalg.norm(s_prime, axis=1)
    tangents_norm = np.zeros_like(s_prime)
    valid_tangents_mask = norms_s_prime > 1e-9
    tangents_norm[valid_tangents_mask] = s_prime[valid_tangents_mask] / norms_s_prime[valid_tangents_mask, np.newaxis]

    binormal_candidate = np.cross(tangents_norm, s_double_prime, axis=1)
    norms_binormal_candidate = np.linalg.norm(binormal_candidate, axis=1)
    binormals_norm = np.zeros_like(binormal_candidate)
    valid_binormals_mask = norms_binormal_candidate > 1e-9
    binormals_norm[valid_binormals_mask] = binormal_candidate[valid_binormals_mask] / norms_binormal_candidate[valid_binormals_mask, np.newaxis]

    normals_norm_candidate = np.cross(binormals_norm, tangents_norm, axis=1)
    norms_normals_candidate = np.linalg.norm(normals_norm_candidate, axis=1)
    normals_norm = np.zeros_like(normals_norm_candidate)
    valid_normals_mask = norms_normals_candidate > 1e-9
    normals_norm[valid_normals_mask] = normals_norm_candidate[valid_normals_mask] / norms_normals_candidate[valid_normals_mask, np.newaxis]

    if is_scalar_input:
        return (tangents_norm.squeeze(), normals_norm.squeeze(), binormals_norm.squeeze(),
                valid_tangents_mask.squeeze(), valid_normals_mask.squeeze(), valid_binormals_mask.squeeze())
    else:
        return (tangents_norm, normals_norm, binormals_norm,
                valid_tangents_mask, valid_normals_mask, valid_binormals_mask)

# --- Visualize the Frenet-Serret frame on the spline (using u_fine) ---
(tangents_fine, normals_fine, binormals_fine, 
 valid_t_fine, valid_n_fine, valid_b_fine) = calculate_frenet_frames(u_fine, tck)

fig_frames = plt.figure(figsize=(12, 10))
ax_frames = fig_frames.add_subplot(111, projection='3d')
ax_frames.plot(x_fine, y_fine, z_fine, '-', label='Fitted spline', alpha=0.4, linewidth=2)
ax_frames.scatter(x, y, z, c='gray', marker='o', label='Original points', alpha=0.3)

num_frames_to_plot = 100
indices_to_plot = np.linspace(0, len(u_fine)-1, num_frames_to_plot, dtype=int)
data_ranges = np.array([np.ptp(arr) for arr in [x_fine, y_fine, z_fine]])
arrow_length = 0.05 * np.mean(data_ranges[data_ranges > 1e-6]) if np.any(data_ranges > 1e-6) else 0.1
if not np.isfinite(arrow_length) or arrow_length <= 1e-6: arrow_length = 0.1

for idx, i in enumerate(indices_to_plot):
    origin = np.array([x_fine[i], y_fine[i], z_fine[i]])
    if valid_t_fine[i]:
        ax_frames.quiver(origin[0], origin[1], origin[2], tangents_fine[i,0], tangents_fine[i,1], tangents_fine[i,2],
                         length=arrow_length, normalize=True, color='r', label='Tangent' if idx == 0 else "")
    if valid_n_fine[i]:
        ax_frames.quiver(origin[0], origin[1], origin[2], normals_fine[i,0], normals_fine[i,1], normals_fine[i,2],
                         length=arrow_length, normalize=True, color='g', label='Normal' if idx == 0 else "")
    if valid_b_fine[i]:
        ax_frames.quiver(origin[0], origin[1], origin[2], binormals_fine[i,0], binormals_fine[i,1], binormals_fine[i,2],
                         length=arrow_length, normalize=True, color='b', label='Binormal' if idx == 0 else "")

ax_frames.set_xlabel('X'); ax_frames.set_ylabel('Y'); ax_frames.set_zlabel('Z')
handles, labels = ax_frames.get_legend_handles_labels()
by_label = dict(zip(labels, handles)); ax_frames.legend(by_label.values(), by_label.keys())
max_plot_range = np.max(data_ranges[data_ranges > 1e-6] if np.any(data_ranges > 1e-6) else [1.0])
ax_frames.set_xlim(x_fine.mean() - max_plot_range/1.5, x_fine.mean() + max_plot_range/1.5) # Adjusted factor for better view
ax_frames.set_ylim(y_fine.mean() - max_plot_range/1.5, y_fine.mean() + max_plot_range/1.5)
if data_ranges[2] > 1e-6:
    ax_frames.set_zlim(z_fine.mean() - max_plot_range/1.5, z_fine.mean() + max_plot_range/1.5)
else:
    z_center = z_fine.mean()
    ax_frames.set_zlim(z_center - 0.5 * max_plot_range, z_center + 0.5 * max_plot_range)
plt.title('Frenet-Serret Frames on 3D Spline')
plt.savefig(os.path.join(base_dir, 'frenet_serret_frames.png'))

# --- 1. Generate 100 random points near the spline ---
num_random_points = 10000
# Use a fraction of the spline's characteristic size for noise
noise_std_dev = arrow_length * 2 # e.g., 2 times the arrow_length used for Frenet frames

# Generate random points along the spline to perturb
random_u_on_spline = np.random.rand(num_random_points)
points_on_spline = np.array(interpolate.splev(random_u_on_spline, tck)).T

# Add Gaussian noise
noise = np.random.normal(scale=noise_std_dev, size=(num_random_points, 3))
random_points_near_spline = np.stack([cell_centroids_x_subset, cell_centroids_y_subset, cell_centroids_z_subset], axis=1)
# --- Robust projection setup: dense sampling + KD-tree seeds ---
from scipy.spatial import cKDTree
u_dense_proj = np.linspace(0.0, 1.0, 5001)
curve_dense_proj = np.array(interpolate.splev(u_dense_proj, tck)).T  # (N,3)
kdt_curve = cKDTree(curve_dense_proj)

# --- 2. Project points onto the spline (robust) ---
projected_points_coords = np.zeros((num_random_points, 3))
projected_points_u_params = np.zeros(num_random_points)

def distance_sq_to_point(u_param, P_rand_xyz, tck_spline):
    S_u = np.array(interpolate.splev(u_param, tck_spline))
    return np.sum((S_u - P_rand_xyz)**2)

def robust_project_point(P_xyz, tck_spline, u_dense, curve_dense, kdt, seeds_k=5, bracket_delta=0.07):
    # Seed candidate u's from nearest sampled spline points
    dists, idxs = kdt.query(P_xyz, k=seeds_k)
    idxs = np.atleast_1d(idxs)
    best_u = None
    best_val = np.inf
    for idx in idxs:
        u_seed = float(u_dense[int(idx)])
        a = max(0.0, u_seed - bracket_delta)
        b = min(1.0, u_seed + bracket_delta)
        if b - a < 1e-6:
            a = max(0.0, u_seed - 0.02)
            b = min(1.0, u_seed + 0.02)
        res_local = minimize_scalar(distance_sq_to_point,
                                    bounds=(a, b),
                                    args=(P_xyz, tck_spline),
                                    method='bounded',
                                    options={'xatol': 1e-8, 'maxiter': 200})
        if res_local.success and res_local.fun < best_val:
            best_val = res_local.fun
            best_u = float(res_local.x)
    # Fallback: if optimizer fails, use nearest sampled u
    if best_u is None:
        best_u = float(u_dense[int(idxs[0])])
        best_val = distance_sq_to_point(best_u, P_xyz, tck_spline)
    return best_u, best_val

# show the random points and spline
fig_random_points = plt.figure(figsize=(10, 8))
ax_random_points = fig_random_points.add_subplot(111, projection='3d')
ax_random_points.scatter(random_points_near_spline[:, 0], random_points_near_spline[:, 1], random_points_near_spline[:, 2], c='gray', marker='o', label='Random points', alpha=0.5, s=1)
ax_random_points.plot(x_fine, y_fine, z_fine, '-', label='Fitted spline', linewidth=2.5, color='red')
ax_random_points.set_xlabel('X'); ax_random_points.set_ylabel('Y'); ax_random_points.set_zlabel('Z')
plt.savefig(os.path.join(base_dir, 'frenet_serret_points.png'))

# FULL RUN ON ALL POINTS

projected_points_coords = np.zeros((len(cell_centroids_x), 3))
projected_points_u_params = np.zeros(len(cell_centroids_x))

def _init_worker():
    # ensure each worker sets single-threaded BLAS (redundant-safe)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def _project_point(idx):
    try:
        P_rand = np.array([cell_centroids_x[idx], cell_centroids_y[idx], cell_centroids_z[idx]])
        # use the robust projector (KD-tree seeded local minimization)
        u_opt, _ = robust_project_point(P_rand, tck, u_dense_proj, curve_dense_proj, kdt_curve,
                                        seeds_k=5, bracket_delta=0.07)
        proj_coord = np.array(interpolate.splev(u_opt, tck))
        V = P_rand - proj_coord
        T_proj, N_proj, B_proj, valid_t_proj, valid_n_proj, valid_b_proj = calculate_frenet_frames(u_opt, tck)
        tangent_coord = float(np.dot(V, T_proj)) if valid_t_proj else 0.0
        normal_coord = float(np.dot(V, N_proj)) if valid_n_proj else 0.0
        binormal_coord = float(np.dot(V, B_proj)) if valid_b_proj else 0.0
        return (idx, float(u_opt), proj_coord.astype(float), tangent_coord, normal_coord, binormal_coord)
    except Exception as e:
        # return index with NaNs on failure so caller can handle/log
        return (idx, np.nan, np.array([np.nan, np.nan, np.nan]), np.nan, np.nan, np.nan)

# Use most CPUs minus one (leave one free), fall back to 1
n_procs = max(1, (int(os.environ['NSLOTS']) or 1) - 1)
print(f"Projecting all points (using {n_procs} cores)...")
# Use multiprocessing.Pool (Linux: fork semantics keep large objects in memory)
with multiprocessing.Pool(processes=n_procs, initializer=_init_worker) as pool:
    # imap_unordered yields results as they finish; chunksize speeds up dispatch
    chunksize = 32
    it = pool.imap_unordered(_project_point, range(len(cell_centroids_x)), chunksize)
    processed = 0
    for res in it:
        idx, u_val, coord, t_val, n_val, b_val = res
        # skip failed entries
        if not np.isfinite(u_val):
            # optionally log failures
            if processed % 10000 == 0:
                print(f"Warning: projection failed for index {idx}")
            processed += 1
            continue
        projected_points_u_params[idx] = u_val
        projected_points_coords[idx, :] = coord
        cell_df.loc[cell_df.index[idx], 'tangent_coord'] = t_val
        cell_df.loc[cell_df.index[idx], 'normal_coord'] = n_val
        cell_df.loc[cell_df.index[idx], 'binormal_coord'] = b_val
        cell_df.loc[cell_df.index[idx], 'projected_u_param'] = u_val
        processed += 1
        if processed % 10000 == 0:
            print(f"Projected {processed} / {len(cell_centroids_x)} points")

cell_df.to_csv(os.path.join(base_dir, 'cell_features_bioAxes.csv'), index=False)