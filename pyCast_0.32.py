# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:39:50 2024

@author: giorgio
"""

# -*- coding: utf-8 -*-
"""
Batch Pocket Analysis - Sep 2024

@author: giorgio
"""
import os
import glob
import numpy as np
from biopandas.pdb import PandasPdb
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist
import pyvista as pv  # For visualization (currently commented out)

# Atomic radii for common elements in proteins.
ATOMIC_RADII = {
    'H': 1.20,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'S': 1.80  # Add other elements as needed.
}

#--------1---------2---------3---------4---------5---------6---------7---------8
# Utility Functions
#--------1---------2---------3---------4---------5---------6---------7---------8
def safe_normalize(v):
    """
    Safely normalize a vector, avoiding division by zero.
    
    Args:
        v (np.ndarray): Input vector.
    
    Returns:
        np.ndarray: Normalized vector, or the original vector if its norm is very small.
    """
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        return v / norm
    return v
#--------1---------2---------3---------4---------5---------6---------7---------8
# Delaunay Triangulation and Alpha Complex Function
#--------1---------2---------3---------4---------5---------6---------7---------8
def get_atomic_radii(ppdb):
    """
    Extract the atomic radii for each atom in the protein from the given PandasPdb object.
    
    Args:
        ppdb (PandasPdb): Parsed PDB data.
        
    Returns:
        np.ndarray: Array of atomic radii.
    """
    atom_elements = ppdb.df['ATOM']['element_symbol'].values
    return np.array([ATOMIC_RADII.get(elem, 1.5) for elem in atom_elements])  # Default radius if unknown

def load_and_separate_pdb(pdb_file):
    """
    Load a PDB file and separate protein and ligand coordinates.
    
    Args:
        pdb_file (str): Path to the PDB file.
    
    Returns:
        tuple: (protein_coords, ligand_coords, ppdb) where each is extracted from the file.
    """
    ppdb = PandasPdb().read_pdb(pdb_file)
    protein_coords = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    ligand_coords = ppdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']].values
    return protein_coords, ligand_coords, ppdb

def compute_weighted_delaunay(points, radii):
    """
    Compute a weighted Delaunay triangulation.
    Each point is "adjusted" by adding its atomic radius.
    
    Args:
        points (np.ndarray): Array of point coordinates (N x 3).
        radii (np.ndarray): Array of atomic radii for each point.
    
    Returns:
        Delaunay: A Delaunay triangulation object computed on weighted points.
    """
    weighted_points = points + radii[:, np.newaxis]
    return Delaunay(weighted_points)

def compute_alpha_complex(tetrapos, alpha):
    """
    Compute the alpha complex from the tetrahedra positions obtained from Delaunay triangulation.
    
    This function was previously returning None, causing a "cannot unpack non-iterable NoneType object" error.
    Now it returns a tuple: (alpha_complex, tetrapos).
    
    Args:
        tetrapos (np.ndarray): Array of shape (n_tetra, 4, 3) representing the positions of the vertices of each tetrahedron.
        alpha (float): Alpha threshold value.
        
    Returns:
        tuple: (alpha_complex, tetrapos) where alpha_complex is a boolean array indicating
               which tetrahedra satisfy the alpha criterion.
    """
    # Compute the squared norm for each vertex in each tetrahedron.
    normsq = np.sum(tetrapos ** 2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    
    # Compute determinants needed for the circumradius calculation.
    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
    
    # Compute the circumradius for each tetrahedron.
    r = np.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c) / (2 * np.abs(a))
    alpha_complex = (r < alpha)
    return alpha_complex, tetrapos
#--------1---------2---------3---------4---------5---------6---------7---------8
# Surface Normal, Flow Field, and Flow Tracing Functions
#--------1---------2---------3---------4---------5---------6---------7---------8
def compute_surface_normals(points, tetra, alpha_complex):
    """
    Compute surface normals for the alpha shape.
    
    For each tetrahedron that is part of the alpha complex, calculate the normals of its faces,
    and accumulate them for each vertex. Finally, normalize the accumulated normals.
    
    Args:
        points (np.ndarray): Array of point coordinates.
        tetra (Delaunay): Delaunay triangulation object.
        alpha_complex (np.ndarray): Boolean array indicating tetrahedra in the alpha complex.
    
    Returns:
        np.ndarray: Array of normalized surface normals for each point.
    """
    normals = np.zeros_like(points)
    counts = np.zeros(len(points))
    
    for i, simplex in enumerate(tetra.simplices):
        if alpha_complex[i]:
            faces = [
                [simplex[0], simplex[1], simplex[2]],
                [simplex[0], simplex[1], simplex[3]],
                [simplex[0], simplex[2], simplex[3]],
                [simplex[1], simplex[2], simplex[3]],
            ]
            for face in faces:
                v1 = points[face[1]] - points[face[0]]
                v2 = points[face[2]] - points[face[0]]
                normal = np.cross(v1, v2)
                normal = safe_normalize(normal)
                for j in face:
                    normals[j] += normal
                    counts[j] += 1
    
    # Normalize the accumulated normals.
    mask = counts > 0
    normals[mask] = np.array([safe_normalize(n) for n in normals[mask]])
    return normals

def compute_flow_field(points, normals, sigma=2.0):
    """
    Compute the flow field based on the local surface normals.
    
    For each point, look for neighboring points within radius sigma and combine
    their normals (weighted by a Gaussian) to determine the flow vector.
    
    Args:
        points (np.ndarray): Array of point coordinates.
        normals (np.ndarray): Array of surface normals for each point.
        sigma (float): Radius used for neighbor search.
    
    Returns:
        np.ndarray: Flow field (vector at each point).
    """
    tree = KDTree(points)
    flow_field = np.zeros_like(points)
    
    for i, point in enumerate(points):
        indices = tree.query_ball_point(point, r=sigma)
        nearby_points = points[indices]
        nearby_normals = normals[indices]
        
        distances = np.linalg.norm(nearby_points - point, axis=1)
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        flow_vector = np.sum(weights[:, np.newaxis] * nearby_normals, axis=0)
        flow_field[i] = safe_normalize(flow_vector)
    
    return flow_field

def trace_flow(point, flow_field, points, max_steps=100, step_size=0.1):
    """
    Trace a flow trajectory starting from a given point using the flow field.
    
    At each step, the nearest point in the dataset is identified and the flow
    vector at that point is used to update the trajectory.
    
    Args:
        point (np.ndarray): Starting point.
        flow_field (np.ndarray): Flow field vectors for each point.
        points (np.ndarray): Array of all point coordinates.
        max_steps (int): Maximum number of steps to trace.
        step_size (float): Step size for each iteration.
    
    Returns:
        np.ndarray: Array of points representing the trajectory.
    """
    trajectory = [point]
    current_point = point
    
    for _ in range(max_steps):
        distances = np.linalg.norm(points - current_point, axis=1)
        nearest_index = np.argmin(distances)
        
        flow_vector = flow_field[nearest_index]
        new_point = current_point + step_size * flow_vector
        
        # Stop if new point is invalid or the movement is too small.
        if np.any(np.isnan(new_point)) or np.any(np.isinf(new_point)):
            break
        if np.linalg.norm(new_point - current_point) < 1e-5:
            break
        
        trajectory.append(new_point)
        current_point = new_point
    
    return np.array(trajectory)

def compute_geometric_center(points):
    """
    Compute the geometric center (centroid) of a set of points.
    
    Args:
        points (np.ndarray): Array of point coordinates.
    
    Returns:
        np.ndarray: The centroid of the points.
    """
    return np.mean(points, axis=0)

#--------1---------2---------3---------4---------5---------6---------7---------8
# Pocket Identification and Discrete Flow Functions
#--------1---------2---------3---------4---------5---------6---------7---------8

def identify_pockets(points, tetra, alpha_complex, sigma=2.0, max_steps=100, step_size=0.1, size_threshold=10,
                     hydrophobic_density=None, apolar_flags=None):
    """
    Identify pockets using the discrete flow method and classify them based on geometric
    and physicochemical metrics.
    
    If hydrophobic_density and apolar_flags are not provided, they are set to zeros (placeholder).
    
    Args:
        points (np.ndarray): Array of point coordinates (e.g., alpha spheres).
        tetra (Delaunay): Delaunay triangulation object.
        alpha_complex (np.ndarray): Boolean array indicating tetrahedra in the alpha complex.
        sigma (float): Neighbor search radius.
        max_steps (int): Maximum steps for flow tracing.
        step_size (float): Step size for flow tracing.
        size_threshold (int): Minimum number of points for a valid pocket.
        hydrophobic_density (np.ndarray, optional): Array of hydrophobic density values.
        apolar_flags (np.ndarray, optional): Binary array (1 if apolar, 0 otherwise).
    
    Returns:
        tuple: (ranked_pockets, trajectories, ranked_scores)
            - ranked_pockets: List of pockets (each is a list of point indices) sorted by score.
            - trajectories: List of trajectories from surface points.
            - ranked_scores: Array of combined scores corresponding to each pocket.
    """
    print("Calculating surface normals...")
    normals = compute_surface_normals(points, tetra, alpha_complex)
    
    print("Calculating flow field...")
    flow_field = compute_flow_field(points, normals, sigma)
    
    # Extract surface points from the alpha complex.
    surface_points = set()
    for i, is_alpha in enumerate(alpha_complex):
        if is_alpha:
            simplex = tetra.simplices[i]
            faces = [
                (simplex[0], simplex[1], simplex[2]),
                (simplex[0], simplex[1], simplex[3]),
                (simplex[0], simplex[2], simplex[3]),
                (simplex[1], simplex[2], simplex[3]),
            ]
            for face in faces:
                surface_points.update(tuple(sorted(face)))
    surface_points = list(surface_points)
    
    print("Tracing flow from surface points...")
    trajectories = [trace_flow(points[i], flow_field, points, max_steps, step_size)
                    for i in surface_points]
    
    print("Clustering endpoints to identify pockets...")
    end_points = np.array([traj[-1] for traj in trajectories])
    
    # Remove invalid endpoints (NaN or Inf)
    valid_end_points = end_points[~np.any(np.isnan(end_points) | np.isinf(end_points), axis=1)]
    if len(valid_end_points) == 0:
        print("No valid endpoints found. Unable to identify pockets.")
        return [], [], []
    
    tree = KDTree(valid_end_points)
    pockets = []
    visited = set()
    for i, pt in enumerate(valid_end_points):
        if i not in visited:
            indices = tree.query_ball_point(pt, r=sigma)
            if len(indices) >= size_threshold:
                pockets.append(indices)
                visited.update(indices)
    
    # Map pocket indices from valid_end_points to surface_points, then to indices in points.
    pocket_points = [[surface_points[i] for i in pocket] for pocket in pockets]
    
    # Set placeholders for hydrophobic_density and apolar_flags if not provided.
    if hydrophobic_density is None:
        hydrophobic_density = np.zeros(points.shape[0])
    if apolar_flags is None:
        apolar_flags = np.zeros(points.shape[0])
    
    # Compute metrics for each pocket.
    geom_metrics_list = []
    chem_metrics_list = []
    for pocket in pocket_points:
        pocket_indices = np.array(pocket)
        geom_metrics = compute_geometric_metrics(points, pocket_indices)
        chem_metrics = compute_chemical_metrics(hydrophobic_density, apolar_flags, pocket_indices)
        geom_metrics_list.append(geom_metrics)
        chem_metrics_list.append(chem_metrics)
    
    # Normalize the metrics separately.
    norm_geom = normalize_metrics(geom_metrics_list)
    norm_chem = normalize_metrics(chem_metrics_list)
    
    # Compute the combined ranking score (separate weights for geometric and chemical metrics).
    scores = compute_ranking_score(norm_geom, norm_chem)
    
    # Sort pockets by score (highest score first).
    sorted_indices = np.argsort(scores)[::-1]
    ranked_pockets = [pocket_points[i] for i in sorted_indices]
    ranked_scores = scores[sorted_indices]
    
    print("Ranking of pockets:")
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"Rank {rank}: Pocket {idx} with score {scores[idx]:.4f}")
    
    return ranked_pockets, trajectories, ranked_scores

def discrete_flow(points, tetra, alpha_complex, sigma=2.0, max_steps=200, step_size=0.1, size_threshold=10):
    """
    Implement the discrete-flow method for identifying pockets with a size threshold.
    
    Returns:
        tuple: (all_pocket_points, pockets, trajectories)
    """
    pockets, trajectories = identify_pockets(points, tetra, alpha_complex, sigma, max_steps, step_size, size_threshold)
    
    # Sort pockets by their size (number of points).
    pockets.sort(key=len, reverse=True)
    
    # Flatten the list of pocket points.
    all_pocket_points = [point for pocket in pockets for point in pocket]
    
    return all_pocket_points, pockets, trajectories

#--------1---------2---------3---------4---------5---------6---------7---------8
# Metric Computation Functions (Geometric & Chemical)
#--------1---------2---------3---------4---------5---------6---------7---------8

def compute_geometric_metrics(alpha_positions, pocket_indices):
    """
    Compute geometric metrics for a given pocket.
    
    Args:
        alpha_positions (np.ndarray): Array of point coordinates.
        pocket_indices (array-like): Indices of the points that form the pocket.
    
    Returns:
        dict: Contains:
            - 'num_alpha': Number of alpha spheres in the pocket.
            - 'mean_pairwise_distance': Mean pairwise distance among the points (inf if only one point).
    """
    num_alpha = len(pocket_indices)
    if num_alpha == 0:
        return {"num_alpha": 0, "mean_pairwise_distance": float('inf')}
    
    pocket_positions = alpha_positions[pocket_indices]
    if num_alpha > 1:
        pairwise_dists = pdist(pocket_positions)
        mean_pairwise_distance = np.mean(pairwise_dists)
    else:
        mean_pairwise_distance = float('inf')
    
    return {"num_alpha": num_alpha, "mean_pairwise_distance": mean_pairwise_distance}

def compute_chemical_metrics(hydrophobic_density, apolar_flags, pocket_indices):
    """
    Compute physicochemical metrics for a given pocket.
    
    Args:
        hydrophobic_density (np.ndarray): Array of hydrophobic density values.
        apolar_flags (np.ndarray): Binary array (1 if apolar, 0 otherwise).
        pocket_indices (array-like): Indices of the points that form the pocket.
    
    Returns:
        dict: Contains:
            - 'mean_hydro': Mean hydrophobic density in the pocket.
            - 'proportion_apolar': Proportion of apolar alpha spheres in the pocket.
    """
    if len(pocket_indices) == 0:
        return {"mean_hydro": 0, "proportion_apolar": 0}
    
    mean_hydro = np.mean(hydrophobic_density[pocket_indices])
    proportion_apolar = np.sum(apolar_flags[pocket_indices]) / len(pocket_indices)
    
    return {"mean_hydro": mean_hydro, "proportion_apolar": proportion_apolar}

def normalize_metrics(metrics_list):
    """
    Apply min-max normalization to a list of metric dictionaries.
    
    Args:
        metrics_list (list of dict): List of dictionaries with metrics.
    
    Returns:
        dict: Dictionary containing normalized arrays for each metric key.
    """
    keys = metrics_list[0].keys()
    norm_metrics = {}
    
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=float)
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val - min_val > 0:
            norm_metrics[key] = (values - min_val) / (max_val - min_val)
        else:
            norm_metrics[key] = values  # If all values are equal.
    return norm_metrics

def compute_ranking_score(norm_geom, norm_chem, geom_weights=None, chem_weights=None):
    """
    Combine normalized geometric and chemical metrics into a final ranking score for each pocket.
    
    For the metric 'mean_pairwise_distance', lower values (denser pockets) are better,
    so we invert that metric (1 - value).
    
    Args:
        norm_geom (dict): Normalized geometric metrics.
        norm_chem (dict): Normalized chemical metrics.
        geom_weights (dict, optional): Weights for geometric metrics (default: {'num_alpha': 1.0, 'mean_pairwise_distance': 1.0}).
        chem_weights (dict, optional): Weights for chemical metrics (default: {'mean_hydro': 1.0, 'proportion_apolar': 1.0}).
    
    Returns:
        np.ndarray: Array of combined scores for each pocket.
    """
    num_pockets = len(next(iter(norm_geom.values())))
    
    if geom_weights is None:
        geom_weights = {'num_alpha': 1.0, 'mean_pairwise_distance': 1.0}
    if chem_weights is None:
        chem_weights = {'mean_hydro': 1.0, 'proportion_apolar': 1.0}
    
    scores = np.zeros(num_pockets)
    
    # Add geometric metrics.
    for key, values in norm_geom.items():
        if key == 'mean_pairwise_distance':
            scores += geom_weights.get(key, 1.0) * (1 - values)
        else:
            scores += geom_weights.get(key, 1.0) * values
    
    # Add chemical metrics.
    for key, values in norm_chem.items():
        scores += chem_weights.get(key, 1.0) * values
    
    return scores

#--------1---------2---------3---------4---------5---------6---------7---------8
# Additional Helper Functions
#--------1---------2---------3---------4---------5---------6---------7---------8
def check_ligand_proximity(pocket, ligand_coords, threshold=4.0):
    """
    Check if any ligand atom is within the specified threshold (in Å) of the pocket's geometric center.
    
    Args:
        pocket (np.ndarray): Array of point coordinates for the pocket.
        ligand_coords (np.ndarray): Array of ligand atom coordinates.
        threshold (float): Distance threshold.
    
    Returns:
        bool: True if any ligand atom is within the threshold, False otherwise.
    """
    center = compute_geometric_center(pocket)
    ligand_tree = KDTree(ligand_coords)
    distances, _ = ligand_tree.query(center, distance_upper_bound=threshold)
    return distances < threshold

#--------1---------2---------3---------4---------5---------6---------7---------8
# Batch Analysis Function
#--------1---------2---------3---------4---------5---------6---------7---------8

def batch_pocket_analysis(pdb_dir, alpha_range, min_pocket_size=10, top_n=5, output_dir=None, proximity_threshold=4.0):
    """
    Perform batch analysis on PDB files in the specified directory.
    For each PDB file, identify and rank pockets using the discrete flow method
    and the computed geometric and chemical metrics.
    
    If hydrophobic_density and apolar_flags are not available, placeholders (zeros) are used.
    
    Args:
        pdb_dir (str): Directory containing PDB files.
        alpha_range (list): [min, max] range for the alpha parameter.
        min_pocket_size (int): Minimum number of points required to consider a pocket valid.
        top_n (int): Number of top pockets to consider for ligand proximity testing.
        output_dir (str, optional): Directory to save visualizations (if implemented).
        proximity_threshold (float): Distance threshold (in Å) for ligand proximity.
    """
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    success_count = 0
    total_count = len(pdb_files)
    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        print(f"\nProcessing {pdb_file}...")
        
        try:
            protein_coords, ligand_coords, ppdb = load_and_separate_pdb(pdb_path)
        except Exception as e:
            print(f"Error loading {pdb_file}: {str(e)}")
            continue
        
        if len(ligand_coords) == 0:
            print(f"No ligand found in {pdb_file}. Skipping.")
            continue
        
        radii = get_atomic_radii(ppdb)
        protein_size = np.ptp(protein_coords, axis=0).max()
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], num=5) * (protein_size / 100)
        
        best_alpha = None
        best_ranked_pockets = []
        best_trajectories = None
        
        for alpha in alpha_values:
            print(f"Trying alpha = {alpha:.4f}")
            try:
                # Compute weighted Delaunay triangulation.
                tetra = compute_weighted_delaunay(protein_coords, radii)
                tetrapos = np.take(protein_coords, tetra.simplices, axis=0)
                # Compute alpha complex.
                alpha_complex, _ = compute_alpha_complex(tetrapos, alpha)
                
                # Identify and rank pockets using the modular approach.
                ranked_pockets, trajectories, ranked_scores = identify_pockets(
                    protein_coords, tetra, alpha_complex,
                    sigma=protein_size / 50,
                    max_steps=200,
                    step_size=protein_size / 1000,
                    size_threshold=min_pocket_size,
                    hydrophobic_density=None,  # Placeholder for future implementation.
                    apolar_flags=None          # Placeholder for future implementation.
                )
                
                if len(ranked_pockets) > 0:
                    best_alpha = alpha
                    best_ranked_pockets = ranked_pockets
                    best_trajectories = trajectories
                    break
            except Exception as e:
                print(f"Error with alpha = {alpha:.4f} for {pdb_file}: {str(e)}")
                continue
        
        if best_ranked_pockets:
            # Test ligand proximity for the top n pockets.
            passed_tests = sum(1 for pocket in best_ranked_pockets[:top_n]
                               if check_ligand_proximity(protein_coords[pocket], ligand_coords, proximity_threshold))
            
            if passed_tests > 0:
                success_count += 1
                print(f"{pdb_file}: Success! {passed_tests}/{top_n} pockets passed proximity test with alpha = {best_alpha:.4f}.")
            else:
                print(f"{pdb_file}: No pockets passed the proximity test.")
            
            if output_dir:
                output_file = os.path.join(output_dir, f"{pdb_file}_pockets.png")
                # Visualization function (to be implemented later)
                # visualize_alpha_shape(protein_coords, ligand_coords, alpha_complex, tetra, output_file=output_file)
        else:
            print(f"{pdb_file}: No valid pockets found for any alpha value.")
    
    print(f"\nBatch analysis completed: {success_count}/{total_count} PDB files passed the proximity test.")

#--------1---------2---------3---------4---------5---------6---------7---------8
# Here we go!
#--------1---------2---------3---------4---------5---------6---------7---------8

# Define parameters and directories.
pdb_directory = r'./Astex'         # Directory containing the PDB dataset.
min_pocket_size = 0.1              # Minimum number of points to consider a valid pocket.
top_n_pockets = 3                  # Number of top pockets to consider.
output_directory = r'./output'     # Directory to save visualizations.
alpha_range = [1.5, 3.5]           # Range for the alpha parameter.
proximity_threshold = 4.0          # Distance threshold (in Å) for ligand proximity.

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Run the batch analysis.
batch_pocket_analysis(
    pdb_dir=pdb_directory,
    alpha_range=alpha_range,
    min_pocket_size=min_pocket_size,
    top_n=top_n_pockets,
    output_dir=output_directory,
    proximity_threshold=proximity_threshold
)



