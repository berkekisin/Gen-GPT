import os
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from torch_geometric.utils import to_dense_adj
import pickle

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

def compute_validity_and_uniqueness(sequence, adj_matrix):
    mols = []
    smiles_list = []
    valid_count = 0
    largest_mol = 0
        
    for i in range(sequence.shape[0]):
        num_nodes = (sequence[0,:,-1] == 0).nonzero()[0].item()

        valid, smiles, mol = is_valid_molecule(sequence[i,:num_nodes].to(torch.int),
                                    adj_matrix[i,:num_nodes, :num_nodes],
                                    verbose=False)
        if valid:
            valid_count+= 1
            mols.append(mol)
            smiles_list.append(smiles)
                
                    
    unique_mols = list(set(smiles_list))
    
        
def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(
    atom_types, edge_types, atom_decoder=["H", "C", "N", "O", "F"], verbose=False
):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.tril(edge_types)
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
    return mol


def build_sequence_to_mol(build_sequence, adj_matrix, hydrogens=True):
    atom_types = build_sequence
    edge_types = adj_matrix

    if hydrogens:
        atom_decoder = ["H", "C", "N", "O", "F"]
    else:
        atom_decoder = ["C", "N", "O", "F"]
    return build_molecule(atom_types, edge_types, atom_decoder)


def is_valid_molecule(build_sequence, adj_matrix, verbose=True, hydrogens=True):
    if not verbose:
        RDLogger.DisableLog("rdApp.*")
    mol = build_sequence_to_mol(build_sequence, adj_matrix, hydrogens)
    smiles = mol2smiles(mol)
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        num_components = len(mol_frags)
    except:
        pass
    if smiles is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
            valid = True
        except Chem.rdchem.AtomValenceException:
            if verbose:
                print("Valence error in GetmolFrags")
            valid = False
        except Chem.rdchem.KekulizeException:
            if verbose:
                print("Can't kekulize molecule")
            valid = False
    else:
        if verbose:
            print("Smiles is None")
        valid = False

    return valid, smiles, mol

def pyg_to_sequence(graph):
    assert graph.num_nodes == graph.x.shape[0]
    adj_matrix = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze(dim=0).to(torch.int)
    build_seq = torch.argmax(graph.x[:,:5], dim=-1)
    
    return build_seq, adj_matrix

def save_mol_pickle(sequence, adj, sub_dir, main_dir):
    data_dict = {
        "sequence": sequence,
        "adj": adj
    }
    full_path = os.path.join(main_dir,  sub_dir)
    os.makedirs(full_path, exist_ok=True)
    num_files = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path,f))])
    file_name = os.path.join(full_path, f"{num_files}.pkl")
    with open(file_name, "wb") as f:
        pickle.dump(data_dict, f)

def draw_mol(mol, sub_dir, main_dir ="molecules"):
    # Draw the molecule as an image
    full_path = os.path.join(main_dir,  sub_dir)
    os.makedirs(full_path, exist_ok=True)
    num_files = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path,f))])
    img = Draw.MolToImage(mol.GetMol())
    img.save(os.path.join(full_path, f"{num_files}.png"))

if __name__ == "__main__":
    #["H", "C", "N", "O", "F"]
    build_sequence = [[1,1],[1,1],[2,1],[3,1],[0,1]]
    edges = [[0,0,0,0],[2,0,0,0],[0,1,0,0],[1,0,1,0]]
    
    build_sequence = torch.tensor(build_sequence)
    edges = torch.tensor(edges)
    
    res = is_valid_molecule(build_sequence, edges, verbose=True )
    print(res)
    # Draw the molecule as an image
    img = Draw.MolToImage(res[-1].GetMol())
    # Modify the alpha channel to make parts of the image transparent


    # Save the image with transparency
    img.save('mol_image/water_molecule.png',)

    # Display the image (if using a Jupyter notebook or similar)
    #img.save('mol_image/water_molecule.png')
 
    