import numpy as np
from biopandas.pdb import PandasPdb
import pyvista as pv

# Atomic radii for common elements in proteins
ATOMIC_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80  # Add other elements as needed
}

# Color mapping for different elements
ELEMENT_COLORS = {
    'H': 'white',  # Hydrogen
    'C': 'gray',   # Carbon
    'N': 'blue',   # Nitrogen
    'O': 'red',    # Oxygen
    'S': 'yellow'  # Sulfur
}

def get_atomic_radii(ppdb):
    """
    Extract the atomic radii for each atom in the protein.
    """
    atom_elements = ppdb.df['ATOM']['element_symbol'].values
    return np.array([ATOMIC_RADII.get(elem, 1.5) for elem in atom_elements])  # Default radius if unknown

def get_atom_colors(ppdb):
    """
    Extract colors for each atom based on the element type.
    """
    atom_elements = ppdb.df['ATOM']['element_symbol'].values
    return [ELEMENT_COLORS.get(elem, 'white') for elem in atom_elements]  # Default color is white

def load_and_separate_pdb(pdb_file):
    """
    Load a PDB file and separate protein coordinates.
    """
    ppdb = PandasPdb().read_pdb(pdb_file)
    protein_coords = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    return protein_coords, ppdb

def visualize_protein_with_colored_spheres_interactive(pdb_file):
    """
    Visualize the protein structure from a PDB file using spheres
    with radii proportional to atomic radii and colors based on element type.
    Interactive controls are added for opacity, color, and saving screenshots.
    """
    # Load the PDB file and get coordinates, radii, and colors
    protein_coords, ppdb = load_and_separate_pdb(pdb_file)
    radii = get_atomic_radii(ppdb)
    colors = get_atom_colors(ppdb)
    
    # Initialize Pyvista plotter
    plotter = pv.Plotter()
    
    # Add spheres for each atom, using atomic radii and element-based colors
    spheres = []
    for coord, radius, color in zip(protein_coords, radii, colors):
        sphere = pv.Sphere(radius=radius, center=coord)
        spheres.append(plotter.add_mesh(sphere, color=color, opacity=0.8))
    
    # Add a slider to control the opacity of spheres
    def change_opacity(value):
        for mesh in spheres:
            mesh.GetProperty().SetOpacity(value)
        plotter.update()

    plotter.add_slider_widget(change_opacity, [0.1, 1.0], title="Opacity", value=0.8)

    # Add a button to change the color of the spheres (random color)
    def change_color():
        color = np.random.random(3)
        for mesh in spheres:
            mesh.GetProperty().SetColor(color)
        plotter.update()

    plotter.add_key_event("c", change_color)  # Press 'c' to change color

    # Add a button to save the current view as an image
    def save_screenshot():
        plotter.screenshot("protein_spheres_view.png")
        print("Screenshot saved as 'protein_spheres_view.png'")

    plotter.add_key_event("s", save_screenshot)  # Press 's' to save screenshot

    # Add interactivity
    plotter.add_axes()
    plotter.show_grid()

    # Display the plot with interactive controls
    plotter.show()

# Visualize the PDB file with element-based colored spheres and interactive controls
visualize_protein_with_colored_spheres_interactive("c:/temp/test/1MBN.pdb")
