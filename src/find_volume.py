# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:59:58 2024

@author: giorg
"""


#%% MoleculeMonteCarloCalculator Class
import random
from tqdm import tqdm

class MoleculeMonteCarloCalculator:
    def __init__(self, pdb_file):
        self.structure = self.load_pdb(pdb_file)
        self.atoms = list(self.structure.get_atoms())
        if not self.atoms:
            raise ValueError("No atoms found in the PDB file.")
        self.bounding_box = self.calculate_bounding_box()

    def load_pdb(self, pdb_file):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure('molecule', pdb_file)

    def calculate_bounding_box(self):
        coords = np.array([atom.coord for atom in self.atoms])
        return np.min(coords, axis=0), np.max(coords, axis=0)

    def is_point_inside_any_atom(self, point):
        for atom in self.atoms:
            element = atom.element
            radius = VDW_RADII.get(element, 1.5)
            distance = np.linalg.norm(point - atom.coord)
            if distance <= radius:
                return True
        return False

    def generate_random_point_in_box(self):
        return np.random.uniform(self.bounding_box[0], self.bounding_box[1])

    def generate_random_point_on_sphere(self, center, radius):
        u = np.random.uniform()
        v = np.random.uniform()
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return center + np.array([x, y, z])

    def calculate_volume(self, num_points):
        points_inside = 0
        for _ in tqdm(range(num_points), desc="Calculating volume"):
            point = self.generate_random_point_in_box()
            if self.is_point_inside_any_atom(point):
                points_inside += 1
        total_volume = np.prod(self.bounding_box[1] - self.bounding_box[0])
        molecule_volume = (points_inside / num_points) * total_volume
        return molecule_volume

    def calculate_surface(self, num_points):
        total_surface = sum(4 * np.pi * VDW_RADII[atom.element]**2 for atom in self.atoms)
        points_on_surface = 0
        for _ in tqdm(range(num_points), desc="Calculating surface area"):
            atom = random.choices(self.atoms, weights=[VDW_RADII[atom.element]**2 for atom in self.atoms])[0]
            point = self.generate_random_point_on_sphere(atom.coord, VDW_RADII[atom.element])
            if not self.is_point_inside_any_atom(point):
                points_on_surface += 1
        molecule_surface = (points_on_surface / num_points) * total_surface
        return molecule_surface

#%% Results Handling
import csv
import os

def save_volume_surface_results(file_path, pdb_id, avg_volume, avg_surface):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['PDB_ID', 'Average Volume (Å³)', 'Average Surface Area (Å²)'])
        writer.writerow([pdb_id, avg_volume, avg_surface])
    print(f"Results saved for {pdb_id}")

def load_volume_surface_results(file_path):
    results = {}
    if os.path.isfile(file_path):
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                pdb_id = row['PDB_ID']
                # Find the volume and surface area columns
                volume_key = next((key for key in row.keys() if 'volume' in key.lower()), None)
                surface_key = next((key for key in row.keys() if 'surface' in key.lower()), None)
                
                if volume_key and surface_key:
                    avg_volume = float(row[volume_key])
                    avg_surface = float(row[surface_key])
                    results[pdb_id] = {'volume': avg_volume, 'surface': avg_surface}
                else:
                    print(f"Warning: Could not find volume or surface data for {pdb_id}")
    return results

#%% Volume Calculation Runner
import statistics

def run_vol_calc(pdb_file, num_runs=5, num_points=2000, results_file='volume_surface_results.csv'):
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    
    existing_results = load_volume_surface_results(results_file)
    
    if pdb_id in existing_results:
        print(f"Results already available for {pdb_id}. Volume = {existing_results[pdb_id]['volume']:.2f} Å³, Surface Area = {existing_results[pdb_id]['surface']:.2f} Å²")
        return existing_results[pdb_id]['volume'], existing_results[pdb_id]['surface']
    
    volumes = []
    surfaces = []
    for run in range(num_runs):
        print(f"\nRunning simulation {run + 1} of {num_runs}...")
        calculator = MoleculeMonteCarloCalculator(pdb_file)
        volume = calculator.calculate_volume(num_points=num_points)
        surface = calculator.calculate_surface(num_points=num_points)
        volumes.append(volume)
        surfaces.append(surface)
        print(f"Run {run + 1}: Volume = {volume:.2f} Å³, Surface Area = {surface:.2f} Å²")
    
    avg_volume = statistics.mean(volumes)
    avg_surface = statistics.mean(surfaces)
    std_volume = statistics.stdev(volumes)
    std_surface = statistics.stdev(surfaces)

    print(f"\nFinal results after {num_runs} runs:")
    print(f"Average Volume: {avg_volume:.2f} Å³ ± {std_volume:.2f} Å³")
    print(f"Average Surface Area: {avg_surface:.2f} Å² ± {std_surface:.2f} Å²")
    
    save_volume_surface_results(results_file, pdb_id, avg_volume, avg_surface)
    
    return avg_volume, avg_surface