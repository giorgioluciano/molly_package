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
import numpy as np
from Bio import PDB
from biopandas.pdb import PandasPdb
from scipy.spatial import Delaunay, KDTree
import pyvista as pv
import glob

# Atomic radii for common elements in proteins
ATOMIC_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80  # Add other elements as needed
}

def get_atomic_radii(ppdb):
    """
    Extract the atomic radii for each atom in the protein.
    """
    atom_elements = ppdb.df['ATOM']['element_symbol'].values
    return np.array([ATOMIC_RADII.get(elem, 1.5) for elem in atom_elements])  # Default radius if unknown

def compute_weighted_delaunay(points, radii):
    """
    Compute a Delaunay triangulation with weighted points based on atomic radii.
    Each point is adjusted by its corresponding atomic radius.
    """
    # Adjust points based on radii
    weighted_points = points + radii[:, np.newaxis]
    return Delaunay(weighted_points)

def load_and_separate_pdb(pdb_file):
    """
    Load a PDB file and separate protein and ligand coordinates.
    """
    ppdb = PandasPdb().read_pdb(pdb_file)
    protein_coords = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    ligand_coords = ppdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']].values
    return protein_coords, ligand_coords, ppdb


def compute_alpha_complex(tetrapos, alpha):
    """Compute the alpha complex from the Delaunay triangulation."""
    normsq = np.sum(tetrapos ** 2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))

    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))

    r = np.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c) / (2 * np.abs(a))
    alpha_complex = (r < alpha)
    return alpha_complex, tetrapos

def safe_normalize(v):
    """Normalizza un vettore in modo sicuro, gestendo i casi di vettori nulli."""
    norm = np.linalg.norm(v)
    if norm > 1e-10:  # Soglia per evitare divisioni per numeri molto piccoli
        return v / norm
    return v

def compute_surface_normals(points, tetra, alpha_complex):
    """Calcola le normali di superficie per l'alpha shape in modo più robusto."""
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
                        
    # Normalizza in modo sicuro
    mask = counts > 0
    normals[mask] = np.array([safe_normalize(n) for n in normals[mask]])
    return normals

def compute_flow_field(points, normals, sigma=2.0):
    """Calcola il campo di flusso per ogni punto in modo più robusto."""
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
    """Traccia il flusso da un dato punto in modo più robusto."""
    trajectory = [point]
    current_point = point
    
    for _ in range(max_steps):
        distances = np.linalg.norm(points - current_point, axis=1)
        nearest_index = np.argmin(distances)
        
        flow_vector = flow_field[nearest_index]
        new_point = current_point + step_size * flow_vector
        
        if np.any(np.isnan(new_point)) or np.any(np.isinf(new_point)):
            break  # Interrompi se si generano valori non finiti
        
        if np.linalg.norm(new_point - current_point) < 1e-5:
            break
        
        trajectory.append(new_point)
        current_point = new_point
        
    return np.array(trajectory)
    
def identify_pockets(points, tetra, alpha_complex, sigma=2.0, max_steps=100, step_size=0.1, size_threshold=10):
    """Identifica le tasche usando il metodo del flusso discreto con una soglia di dimensione."""
    print("Calculating surface normals...")
    normals = compute_surface_normals(points, tetra, alpha_complex)
    
    print("Calculating flow field...")
    flow_field = compute_flow_field(points, normals, sigma)
    
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
                face = tuple(sorted(face))
                surface_points.update(face)
    surface_points = list(surface_points)
    
    print("Tracing flow from surface points...")
    trajectories = [trace_flow(points[i], flow_field, points, max_steps, step_size) 
                    for i in surface_points]
    
    print("Clustering endpoints to identify pockets...")
    end_points = np.array([traj[-1] for traj in trajectories])
    
    # Rimuovi eventuali punti finali non finiti
    valid_end_points = end_points[~np.any(np.isnan(end_points) | np.isinf(end_points), axis=1)]
    
    if len(valid_end_points) == 0:
        print("No valid end points found. Unable to identify pockets.")
        return [], []
    
    pocket_tree = KDTree(valid_end_points)
    
    pockets = []
    visited = set()
    for i, end_point in enumerate(valid_end_points):
        if i not in visited:
            indices = pocket_tree.query_ball_point(end_point, r=sigma)
            if len(indices) >= size_threshold:
                pockets.append(indices)
                visited.update(indices)
    
    pocket_points = [[surface_points[i] for i in pocket] for pocket in pockets]
    
    return pocket_points, trajectories


def discrete_flow(points, tetra, alpha_complex, sigma=2.0, max_steps=200, step_size=0.1, size_threshold=10):
    """Implement the discrete-flow method for identifying pockets with size threshold."""
    pockets, trajectories = identify_pockets(points, tetra, alpha_complex, sigma, max_steps, step_size, size_threshold)
    
    # Sort pockets by size
    pockets.sort(key=len, reverse=True)
    
    # Flatten the list of pocket points
    all_pocket_points = [point for pocket in pockets for point in pocket]
    
    return all_pocket_points, pockets, trajectories


def compute_geometric_center(points):
    """Compute the geometric center of a set of points."""
    return np.mean(points, axis=0)

def check_ligand_proximity(pocket, ligand_coords, threshold=4.0):
    """Check if any ligand atom is within 4 Å of the pocket's geometric center."""
    center = compute_geometric_center(pocket)
    ligand_tree = KDTree(ligand_coords)
    distances, _ = ligand_tree.query(center, distance_upper_bound=threshold)
    return distances < threshold

#%%
def visualize_top_n_pockets(all_coords, pockets, n=5, trajectories=None, output_file=None):
    """
    Visualize the top n pockets using PyVista.
    """
    # Create a PyVista point cloud for all atoms
    cloud = pv.PolyData(all_coords)
    
    # Create a plotter
    plotter = pv.Plotter()
    
    # Add all atoms as small spheres
    spheres = cloud.glyph(scale=0.5, geom=pv.Sphere())
    plotter.add_mesh(spheres, color='lightgray', opacity=0.5)
    
    # Add the top n pockets with different colors
    colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'pink']
    for i, pocket in enumerate(pockets[:n]):  # Show only top n pockets
        pocket_coords = all_coords[pocket]
        pocket_cloud = pv.PolyData(pocket_coords)
        pocket_spheres = pocket_cloud.glyph(scale=0.7, geom=pv.Sphere())
        plotter.add_mesh(pocket_spheres, color=colors[i % len(colors)], opacity=0.7)
    
    # Add trajectories (flow paths) if provided
    if trajectories:
        for traj in trajectories:
            line = pv.Spline(traj, 100)
            plotter.add_mesh(line, color='black')
    
    # Set up the camera
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.5)
    
    # Either save to file or show interactively
    if output_file:
        plotter.show(screenshot=output_file)
    else:
        plotter.show()
        


def improved_batch_pocket_analysis(pdb_dir, alpha_range, min_pocket_size=10, top_n=5, output_dir=None, proximity_threshold=4.0):
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    success_count = 0
    total_count = len(pdb_files)

    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        print(f"Processing {pdb_file}...")

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
        best_pockets = []
        best_trajectories = None

        for alpha in alpha_values:
            print(f"Trying alpha = {alpha:.4f}")
            try:
                tetra = compute_weighted_delaunay(protein_coords, radii)
                tetrapos = np.take(protein_coords, tetra.simplices, axis=0)
                alpha_complex, _ = compute_alpha_complex(tetrapos, alpha)

                all_pocket_points, individual_pockets, trajectories = discrete_flow(
                    protein_coords, tetra, alpha_complex, 
                    sigma=protein_size/50,
                    max_steps=200, 
                    step_size=protein_size/1000,
                    size_threshold=min_pocket_size)

                if len(individual_pockets) > 0:
                    best_alpha = alpha
                    best_pockets = individual_pockets
                    best_trajectories = trajectories
                    break
            except Exception as e:
                print(f"Error with alpha = {alpha:.4f} for {pdb_file}: {str(e)}")
                continue

        if best_pockets:
            passed_tests = sum(1 for pocket in best_pockets[:top_n] 
                               if check_ligand_proximity(protein_coords[pocket], ligand_coords, proximity_threshold))

            if passed_tests > 0:
                success_count += 1
                print(f"{pdb_file}: Success! {passed_tests}/{top_n} pockets passed proximity test with alpha = {best_alpha:.4f}.")
            else:
                print(f"{pdb_file}: No pockets passed the proximity test.")
            
            if output_dir:
                output_file = os.path.join(output_dir, f"{pdb_file}_pockets.png")
                visualize_top_n_pockets(protein_coords, best_pockets, n=top_n, trajectories=best_trajectories, output_file=output_file)
        else:
            print(f"{pdb_file}: No valid pockets found for any alpha value.")
    
    print(f"\nBatch analysis completed: {success_count}/{total_count} PDB files passed the proximity test.")
    
#%%
pdb_directory = r'C:/temp/test/setAlex'
min_pocket_size = 7
top_n_pockets = 5
output_directory = r'C:/temp/test/output'
alpha_range = [1.5, 3.5]
proximity_threshold = 4.0

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

improved_batch_pocket_analysis(
    pdb_dir=pdb_directory,
    alpha_range=alpha_range,
    min_pocket_size=min_pocket_size,
    top_n=top_n_pockets,
    output_dir=output_directory,
    proximity_threshold=proximity_threshold
)
#%%%% Only Alfas
import os
import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from biopandas.pdb import PandasPdb

def load_and_separate_pdb(pdb_file):
    """
    Load a PDB file and separate protein and ligand coordinates.
    """
    ppdb = PandasPdb().read_pdb(pdb_file)
    protein_coords = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    ligand_coords = ppdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']].values
    return protein_coords, ligand_coords, ppdb

def compute_alpha_complex(points, alpha):
    """Compute the alpha complex from the Delaunay triangulation."""
    tetra = Delaunay(points)
    tetrapos = tetra.points[tetra.simplices]
    normsq = np.sum(tetrapos ** 2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))

    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))

    r = np.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c) / (2 * np.abs(a))
    alpha_complex = r < alpha
    return alpha_complex, tetra

def compute_surface_normals(points, tetra, alpha_complex):
    """Compute surface normals for the alpha shape."""
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
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normal /= norm_length
                    for j in face:
                        normals[j] += normal
                        counts[j] += 1
                        
    mask = counts > 0
    normals[mask] /= counts[mask, np.newaxis]
    return normals

def visualize_alpha_shape(protein_coords, ligand_coords, alpha_complex, tetra, output_file=None):
    """Visualize the alpha shape of the protein and ligand coordinates."""
    plotter = pv.Plotter(off_screen=output_file is not None)
    
    try:
        # Select tetrahedra that are part of the alpha complex
        if not np.any(alpha_complex):
            raise ValueError("No valid alpha complex tetrahedra found.")

        valid_tetra = tetra.points[tetra.simplices[alpha_complex]]
        
        # Create a list of unique points and a mapping for their indices
        unique_points, inverse = np.unique(valid_tetra.reshape(-1, 3), axis=0, return_inverse=True)
        
        # Create tetrahedra connectivity
        num_cells = valid_tetra.shape[0]
        cells = np.column_stack((np.full(num_cells, 4), inverse.reshape(-1, 4)))
        
        # Create the unstructured mesh
        mesh = pv.UnstructuredGrid(cells, np.array([pv.CellType.TETRA] * num_cells), unique_points)
        
        # Extract the outer surface
        surface = mesh.extract_surface()-
        
        # Add the protein surface to the plotter
        plotter.add_mesh(surface, color="lightblue", opacity=1, show_edges=True)
        
        # Add ligand points
        plotter.add_points(ligand_coords, color="red", point_size=5.0, render_points_as_spheres=True)
        
        # Set isometric view
        plotter.view_isometric()
        
        # Save or show the plot
        if output_file:
            plotter.screenshot(output_file)
            print(f"Alpha surface visualization saved to {output_file}.")
        else:
            plotter.show()
    
    except Exception as e:
        print(f"Error in creating surface mesh: {str(e)}")
        # Print additional debug information
        print(f"Number of valid tetrahedra: {valid_tetra.shape[0] if 'valid_tetra' in locals() else 'N/A'}")
        print(f"Number of unique points: {unique_points.shape[0] if 'unique_points' in locals() else 'N/A'}")
        print(f"Shape of connectivity: {cells.shape if 'cells' in locals() else 'N/A'}")

def process_directory(directory, alpha=2.0):
    """Batch process all molecules in the directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".pdb"):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            try:
                # Load molecule data
                protein_coords, ligand_coords, _ = load_and_separate_pdb(filepath)
                
                # Compute alpha shape for the protein
                alpha_complex, tetra = compute_alpha_complex(protein_coords, alpha)
                
                # Compute surface normals
                normals = compute_surface_normals(protein_coords, tetra, alpha_complex)

                # Visualize results
                visualize_alpha_shape(protein_coords, ligand_coords, alpha_complex, tetra, output_file=f"{filename}_alpha_surface.png")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

# Run the batch processing
directory = "c:/temp/test/setAlex"
process_directory(directory)


#%%

def improved_batch_pocket_analysis(pdb_dir, alpha_range, min_pocket_size=10, top_n=5, output_dir=None, proximity_threshold=4.0):
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    success_count = 0
    total_count = len(pdb_files)

    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        print(f"Processing {pdb_file}...")

        try:
            protein_coords, ligand_coords, ppdb = load_and_separate_pdb(pdb_path)
        except Exception as e:
            print(f"Error loading {pdb_file}: {str(e)}")
            continue

        if len(ligand_coords) == 0:
            print(f"No ligand found in {pdb_file}. Skipping.")
            continue

        radii = get_atomic_radii(ppdb)

        # Calcola un range di alpha basato sulle dimensioni della proteina
        protein_size = np.ptp(protein_coords, axis=0).max()
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], num=5) * (protein_size / 100)

        best_alpha = None
        best_pockets = []
        best_trajectories = None

        for alpha in alpha_values:
            try:
                tetra = compute_weighted_delaunay(protein_coords, radii)
                tetrapos = np.take(protein_coords, tetra.simplices, axis=0)
                alpha_complex, _ = compute_alpha_complex(tetrapos, alpha)

                all_pocket_points, individual_pockets, trajectories = discrete_flow(
                    protein_coords, tetra, alpha_complex, 
                    sigma=protein_size/50,  # Adatta sigma alla dimensione della proteina
                    max_steps=300, 
                    step_size=protein_size/1000,  # Adatta step_size alla dimensione della proteina
                    size_threshold=min_pocket_size)

                if len(individual_pockets) > 0:
                    best_alpha = alpha
                    best_pockets = individual_pockets
                    best_trajectories = trajectories
                    break
            except Exception as e:
                print(f"Error with alpha = {alpha} for {pdb_file}: {str(e)}")
                continue

        if best_pockets:
            passed_tests = sum(1 for pocket in best_pockets[:top_n] 
                               if check_ligand_proximity(protein_coords[pocket], ligand_coords, proximity_threshold))

            if passed_tests > 0:
                success_count += 1
                print(f"{pdb_file}: Success! {passed_tests}/{top_n} pockets passed proximity test with alpha = {best_alpha}.")
            else:
                print(f"{pdb_file}: No pockets passed the proximity test.")
            
            if output_dir:
                output_file = os.path.join(output_dir, f"{pdb_file}_pockets.png")
                visualize_top_n_pockets(protein_coords, best_pockets, n=top_n, trajectories=best_trajectories, output_file=output_file)
        else:
            print(f"{pdb_file}: No valid pockets found for any alpha value.")
    
    print(f"\nBatch analysis completed: {success_count}/{total_count} PDB files passed the proximity test.")



#%%
# Uso:
pdb_directory = r'C:/temp/test/setAlex'
min_pocket_size = 7  # Ridotto da 10
top_n_pockets = 5
output_directory = r'C:/temp/test/output'
alpha_range = [1.5, 2.5]  # Range di alpha da testare
proximity_threshold = 4.0

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

improved_batch_pocket_analysis(
    pdb_dir=pdb_directory,
    alpha_range=alpha_range,
    min_pocket_size=min_pocket_size,
    top_n=top_n_pockets,
    output_dir=output_directory,
    proximity_threshold=proximity_threshold
)

#%%


pdb_directory = r'C:/temp/test/setAlex/2nd'
min_pocket_size = 5  # Ridotto da 10
top_n_pockets = 8
output_directory = r'C:/temp/test/output'
alpha_range = [1.1, 3]  # Range di alpha da testare
proximity_threshold = 4.0

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

improved_batch_pocket_analysis(
    pdb_dir=pdb_directory,
    alpha_range=alpha_range,
    min_pocket_size=min_pocket_size,
    top_n=top_n_pockets,
    output_dir=output_directory,
    proximity_threshold=proximity_threshold
)
