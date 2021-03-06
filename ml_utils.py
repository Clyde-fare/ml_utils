import numpy as np
from ase import Atoms
from ase.units import Bohr
from ase.io import read
from pybel import readfile
import mdtraj as md
import simtk.unit as u
import hack_parser
from hack_parser.utils import convertor
from gausspy import Gaussian
import tarfile
import os
import dill

import warnings


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
    
    try:
        coords = read(m.calc.log).positions
    except AttributeError:
        try:
            coords = np.array(m.calc.max_data['Positions'])
        except ValueError:
            pass

    symbols = m.get_chemical_symbols()
    no_atoms = len(m)

    ##these try except blocks are because the machine readable part of log files 
    ##for the calculations on the single atoms in explicably lack the fields (though
    ##in the log file they are liste
    try:
        charges = np.array(m.calc.max_data['atomcharges']['mulliken'])
    except KeyError:
        charges = [None for i in range(no_atoms)]
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
        rot_A, rot_B, rot_C = None,None,None
    
    try:
        h_298k = m.calc.max_data['enthalpy']
    except KeyError:
        h_298k = None

    try:
        f_298k = m.calc.max_data['freeenergy']
    except KeyError:
        f_298k = None

    try:
        u_0k = m.calc.max_data['Hf'] + zpe
    except KeyError:
        u_0k = None
    
    try:
        u_298k = m.calc.max_data['Hf'] + m.calc.max_data['Thermal']
    except KeyError:
        u_298k = None

    try:
        dipole = np.linalg.norm(m.calc.max_data['Dipole'])
    except KeyError:
        dipole = None

    try:
        polarisability = m.calc.max_data['Polar'][0]
    except KeyError:
        polarisability = None

    try:   
        homo_ind = m.calc.max_data['homos'][0]
    except KeyError:
        homo_ind = None

    try:
        homo = m.calc.max_data['moenergies'][0][homo_ind]
    except KeyError:
        homo = None

    try:
        lumo = m.calc.max_data['moenergies'][0][homo_ind+1]
    except KeyError:
        lumo = None

    try:
        band_gap = lumo-homo
    except TypeError:
        band_gap = None

    # the biotools environment on this machine and the ml environment on cx1
    # contain a hacked cclib that extract heatcapacity/rotconstants and ese from log files
    try:
        cp_298k = m.calc.max_data['heatcapacity']
    except KeyError:
        cp_298k = None

    try:
        ese = m.calc.max_data['ese']
    except KeyError:
        ese = None

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


def to_coulomb_vec(mol, max_size=23, addition=None):
    """Generates a coulomb matrix representing the molecule passed in mol
    
    if an additional array is passed this is added on to the end of the vector """

    cm = to_coulomb_m(mol, max_size=max_size)
    cm_vec = cm[np.tril_indices(cm.shape[0])]
    
    if addition:
        cm_vec = np.hstack([cm_vec, addition])
    
    return cm_vec


def to_ase(mol_p):
    return Atoms(symbols = mol_p['symbols'], positions=mol_p['coords'])


def solvate(ase_mol, mol_id, ase_solvent_mol, solvent_id, n_solvent=100):
    """Build solvent sphere around molecule"""
    
    from openmoltools import packmol
    # needs to be here rather than global because we sometimes pickle this function and send it to the server to run
    # because on importing packmol sets various paths (e.g. where temporary files are kept)
    # so we don't want to import those from the local session to the cluster
    
    ##File paths
    mol_xyz_path =  mol_id + '.xyz'
    mol_pdb_path = mol_id + '.pdb'

    solvent_xyz_path = solvent_id + '.xyz'
    solvent_pdb_path = solvent_id + '.pdb'

    ##File construction of xyz and pdb files
    ase_mol.write(mol_xyz_path)
    pybel_mol = readfile("xyz", mol_xyz_path).next()
    pybel_mol.write('pdb', mol_pdb_path, overwrite=True)

    ##File construction of solvent pdb files

    if not os.path.isfile(solvent_xyz_path):
        ase_solvent_mol.write(solvent_xyz_path)
    
    if not os.path.isfile(solvent_pdb_path):
        pybel_mol = readfile("xyz", solvent_xyz_path).next()
        pybel_mol.write('pdb', solvent_pdb_path)
    
    solute_traj = md.load(mol_pdb_path)
    solvent_traj = md.load(solvent_pdb_path)
    
    #currently only works for solvent composed of Hydrogens and Carbons!
    #size returns length of box as measured in Angstrom
    size = packmol.approximate_volume([mol_pdb_path, solvent_pdb_path], [1,n_solvent])

    if ase_mol != ase_solvent_mol:
        solvated_traj = packmol.pack_box([solute_traj,solvent_traj],[1,n_solvent],
                                         fix=True, shape='sphere', size=size*0.75,
                                         seed= np.random.randint(2**16), )
    else:
        solvated_traj = packmol.pack_box([solvent_traj],[n_solvent+1],
                                         fix=True, shape='sphere', size=size*0.75,
                                         seed= np.random.randint(2**16), )
        
    return to_ase_frames(solvated_traj)[0]


def to_ase_frames(traj):
    """Converts an mdtraj trajectory object corresponding to snapshots from an openMM simulation
    to a list of ASE objects each corresponding to one frame"""
    
    unit_conversion = u.nanometer.conversion_factor_to(u.angstrom)
    
    symbols = ([a.element.symbol for a in traj.topology.atoms])
    sim_positions = [frame_coords*unit_conversion for frame_coords in traj.xyz]
    
    frames = []
    for frame_positions in sim_positions:
        frames.append(Atoms(symbols=symbols,positions=frame_positions))
            
    return frames

    
def generate_solvated_ensemble(orig_mol, mol_id, solvent_mol, solvent_id, n_solvent, ensemble_size):
    """Generated ensemble of solvated molecules"""
    
    #joblib fails, multiprocessing + pool.map fails
    # maybe need to try using partial + pool.map?
    # e.g. http://stackoverflow.com/questions/16542261/python-multiprocessing-pool-with-map-async

    #from joblib import Parallel, delayed
    #if we are in a pbs job parallelise loop
    #ncpus = os.environ.get('NCPUS', 1)
    #ensemble = Parallel(n_jobs=ncpus)( \
    #                delayed(solvate)(orig_mol, mol_id, solvent_mol, solvent_id, n_solvent) for i in range(ensemble_size) \
    #                                 )
    
    #p_solvate = lambda e: solvate(orig_mol, mol_id, solvent_mol, solvent_id, n_solvent)
    #ncpus = os.environ.get('NCPUS', 1)
    #pool = Pool(processes=ncpus)
    #ensemble = pool.map(p_solvate, range(ensemble_size))    
    #ensemble_dfs = [delayed(solvate)(orig_mol, mol_id, solvent_mol, solvent_id, n_solvent) for i in range(ensemble_size)]
    #ensemble = Parallel(n_jobs=ncpus)(ensemble_dfs)
    #ensemble = [df[0](*df[1]) for df in ensemble_dfs]           
    
    ensemble = (solvate(orig_mol, mol_id, solvent_mol, solvent_id, n_solvent) for i in range(ensemble_size))
    
    return ensemble


#energy extraction for oniom calcs, returns 0 if calculation has no oniom energy
def get_o_energies(mol):
    """Parse using hacked old cclib version that parses oniom optimisation jobs"""
    try:
        ev_to_hartree = 1./convertor(1,'hartree','eV')
        g=hack_parser.Gaussian(mol.calc.log, loglevel=50)
        d=g.parse()
        #lm, hm, lr
        o_component_es = np.array(d.oniomenergies)
    except AttributeError:
        return 0

    return (ev_to_hartree * o_component_es * [-1,1,1]).sum(axis=1)


def extract_bz2(tar_bz_f, calc_type):
    """
    Searches through a .tar.bz2 file extracting data from all .log files that
    match string argument 'calc_type'. 
    Then pickles the total list of extracted data
    """
    
    # we give the gaussian calculator a label that includes the directory the
    # log file is contained in, this does not affect the parsing process but
    # generate a warning as it is not an expected form for the label. To avoid
    # the profusion of warning messages in stdout we suppress warnings here.
    warnings.filterwarnings("ignore")

    extended_xyzs = []

    with tarfile.open(tar_bz_f,'r:bz2') as data_tar:
        calc_logs = (log_tar for log_tar in data_tar 
                     if calc_type in log_tar.name and '.log' in log_tar.name)
        
        for calc_log in calc_logs:
            try:
                # extract log file (and relevant directory)
                data_tar.extract(calc_log)

                # convert to ASE/Gaussian format
                mol = read(calc_log.name)
                mol.set_calculator(
                    Gaussian(
                        label=calc_log.name.replace('.log','')
                    )
                )

                # parse into ML readable format
                parsed_mol = ase_mol_parse(mol)
                extended_xyzs.append(parsed_mol)
            except AttributeError:
                # if the calculation didn't complete we get an empty dictionary 
                # so we we keep track of these failed calculations with a blank 
                # entry other than mol_id
                blank_entry = {'mol_id':calc_log.name.replace('.log','')}
                extended_xyzs.append(blank_entry)
            finally:
                # delete extracted log file
                os.remove(calc_log.name)

            # solves memory problem
            # http://blogs.it.ox.ac.uk/inapickle/2011/06/20/high-memory-usage-when-using-pythons-tarfile-module/ # noqa
            data_tar.members=[]
            
    dir_name = tar_bz_f.replace('.tar.bz2','')

    # delete extracted directory
    os.rmdir(dir_name)

    # search key function
    # assumes files are named xxx_frame_{frame}_mol_{no}
    def mol_no_frame(mol):
        mol_no = int(mol['mol_id'].split('_')[-1].replace('.log',''))
        mol_frame = int(mol['mol_id'].split('_')[-3])
        return mol_no,mol_frame

    extended_xyzs = sorted(extended_xyzs,key=mol_no_frame)

    with open(dir_name + '_' + calc_type + '_data.pkl', 'w') as data_f:
        dill.dump(extended_xyzs, data_f)

def fd_no_bins(y):
    """
    Determines the number of bins a histogram of y should use
    Via the Freedman-Diaconis rule
    """
    q75, q25 = np.percentile(y, [75 ,25])
    iqr = q75 - q25
    h=2*iqr*len(y)**(-1./3) 
    return int((max(y)-min(y))/h)

#Hacky
def clean_labels(labels):
    """
    Cleans up an assigned set of bins such that no bin has less than 2
    members
    """

    llabels, slabels = list(labels), set(labels)
    
    for l in slabels:
        if llabels.count(l) <2 and l != max(slabels):
            llabels[llabels.index(l)] = l+1
            return clean_labels(llabels)
        elif llabels.count(l) <2 and l == max(slabels):
            llabels[llabels.index(l)] = l-1
            return clean_labels(llabels)
    else:
        return np.array(llabels)
