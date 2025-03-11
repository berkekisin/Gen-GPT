import pickle
from g3.molecules import draw_mol, build_molecule
import os

if __name__ == "__main__":
    main_dir = 'molecules'
    exp_name  = 'deney2'
    data_name = '4.pkl'
    full_path = os.path.join(main_dir, exp_name,'invalid_pickle', data_name)
    
    with open(full_path, "rb") as f:
        data_dict = pickle.load(f)

    # Access the tensors
    sequence = data_dict["sequence"][:,0]
    adj = data_dict["adj"]
    
    for i in range(sequence.shape[0]):
        mol = build_molecule(sequence[:i], adj[:i,:i])
        sub_dir = os.path.join(exp_name, 'vis2')
        draw_mol(mol, sub_dir=sub_dir, main_dir='berke')
        
   
    
    
    #
    #draw_mol(mol, sub_dir=sub_dir, main_dir=cfg.mol_dir)