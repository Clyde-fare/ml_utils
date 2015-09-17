import numpy as np
from ase import Atoms
from ase.units import Bohr

def extended_xyz_parse(xyz_d):
    """Extracts information contained in the extended xyz format
    Used in the paper: Quantum chemistry structures and properties of 134 kilo molecules
    doi:10.1038/sdata.2014.22"""
    
    s_properties =  ['rot_A', 
                     'rot_B', 
                     'rot_C', 
                     'dipole', 
                     'polarizability', 
                     'homo', 
                     'lumo', 
                     'band_gap', 
                     'ese', 
                     'zpe', 
                     'u_0K', 
                     'u_298.15K', 
                     'h_298.15K', 
                     'f_298.15K', 
                     'cp_298.15K']

    mol_properties = {}


    lines = xyz_d.replace('*^','e').splitlines()
    
    r_no_atoms = lines[0]
    no_atoms = int(r_no_atoms)

    r_scalars = lines[1]
    mol_id = r_scalars.split()[:2]
    scalar_properties = np.array(r_scalars.split()[2:], np.float32)

    r_mcoords = lines[2:2+no_atoms]
    symbols = [m.split()[0] for m in r_mcoords]
    coords = np.array([m.split()[1:4] for m in r_mcoords], dtype=np.float32)
    
    charges = np.array([m.split()[4] for m in r_mcoords], dtype=np.float32)

    r_vibfreqs = lines[2+ no_atoms]
    vib_freqs = np.array([float(freq) for freq in r_vibfreqs.split()], dtype=np.float32)

    smiles = lines[3+no_atoms].split()
    inchi = lines[4+no_atoms].split()

    mol_properties['no_atoms'] = no_atoms
    mol_properties['mol_id'] = mol_id
    
    for i, p in enumerate(s_properties):
        mol_properties[p] = scalar_properties[i]

    mol_properties['symbols'] = symbols
    mol_properties['coords'] = coords
    mol_properties['charges'] = charges
    mol_properties['vib_freqs'] = vib_freqs
    mol_properties['smiles'] = smiles
    mol_properties['inchi'] = inchi
    
    return mol_properties

def ase_mol_parse(m):
    """Extracts information contained in an ASE molecule following a Gaussian calculation to a data dictionary
    matching the format used above """
    
    coords = np.array(m.calc.max_data['Positions'])
    no_atoms = len(m)

    ##these try except blocks are because the machine readable part of log files 
    ##for the calculations on the single atoms in explicably lack the fields (though
    ##in the log file they are liste
    try:
        charges = np.array(m.calc.max_data['atomcharges']['mulliken'])
    except KeyError:
        charges = [0 for i in range(no_atoms)]
    try:
        zpe = m.calc.max_data['Zeropoint']
    except KeyError:
        zpe = 0
    try:
        vib_freqs = m.calc.max_data['vibfreqs']
    except KeyError:
        vib_freqs = []
    try:
        rot_A, rot_B, rot_C = m.calc.max_data['rotconstants']
    except KeyError:
        rot_A, rot_B, rot_C = 0,0,0
    
    h_298k = m.calc.max_data['enthalpy']
    f_298k = m.calc.max_data['freeenergy']
    u_0k = m.calc.max_data['Hf'] + zpe
    u_298k = m.calc.max_data['Hf'] + m.calc.max_data['Thermal']
    
    dipole = np.linalg.norm(m.calc.max_data['Dipole'])
    polarisability = m.calc.max_data['Polar'][0]
    homo_ind = m.calc.max_data['homos'][0]
    homo = m.calc.max_data['moenergies'][0][homo_ind]
    lumo = m.calc.max_data['moenergies'][0][homo_ind+1]
    band_gap = lumo-homo
  
    symbols = m.get_chemical_symbols()

    # the biotools environment on this machine and the ml environment on cx1
    # contain a hacked cclib that extract heatcapacity/rotconstants and ese from log files
    cp_298k = m.calc.max_data['heatcapacity']
    
    ese = m.calc.max_data['ese']
    
    data_dict = {'mol_id': m.calc.label,
                 'coords': coords,
                 'charges': charges,
                 'rot_A': rot_A,
                 'rot_B': rot_B,
                 'rot_C': rot_C,
                 'u_0K': u_0k,
                 'u_298.15K': u_298k,
                 'h_298.15K': h_298k,
                 'f_298.15K': f_298k,
                 'cp_298.15K': cp_298k,
                 'zpe': zpe,
                 
                 'symbols': symbols,
                 'no_atoms': no_atoms,
                 'homo': homo,
                 'lumo': lumo,
                 'band_gap': band_gap,
                 
                 'dipole': dipole,
                 'polarizability': polarisability,
                 'ese': ese,
                 'vib_freqs': vib_freqs,
            
                 'smiles': '',     
                 'inchi': '',
                 }
    
    return data_dict

def to_coulomb_m(mol, max_size=23):
    """Generates a coulomb matrix representing the ASE atoms object"""
    mol = to_ase(mol)
    atomic_nos = mol.get_atomic_numbers()
    positions = mol.get_positions()
    bohr_positions = positions/Bohr
    no_atoms = len(mol)
    coulomb_m = np.zeros([no_atoms, no_atoms])

    for i in range(no_atoms):
        for j in range(no_atoms):
            if i==j:
                coulomb_m[i,j] = 0.5*atomic_nos[i]**2.4
            else:
                coulomb_m[i,j] = atomic_nos[i]*atomic_nos[j]/np.linalg.norm(bohr_positions[i]-bohr_positions[j], ord=2)
    
    #get sorted indices of rows, where we sort by the L2 norm of the row 
    sorted_indices = sorted(range(len(coulomb_m)), key= lambda i: np.linalg.norm(coulomb_m[i]), reverse=True)
    
    #sort rows and columns of the coulomb matrix according the row norm
    coulomb_m = coulomb_m[sorted_indices][:,sorted_indices]
    padding = max_size - no_atoms
    
    return np.pad(coulomb_m, pad_width= ((0,padding),(0,padding)), mode='constant', constant_values=0)

def to_coulomb_vec(mol, addition=None):
    """Generates a coulomb matrix representing the molecule passed in mol
    
    if an additional array is passed this is added on to the end of the vector """

    cm = to_coulomb_m(mol)
    cm_vec = cm[np.tril_indices(cm.shape[0])]
    
    if addition:
        cm_vec = np.hstack([cm_vec, addition])
    
    return cm_vec

to_ase = lambda mol_p: Atoms(symbols = mol_p['symbols'], positions=mol_p['coords'])
