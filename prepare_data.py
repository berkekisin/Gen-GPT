import os
import hydra
import sys

from g3.dataset.qm9 import prepare_qm9
from g3.dataset.grids import prepare_grids

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

@hydra.main(version_base=None, config_path="./config/", config_name="preprocessing")
def main(cfg):
    dataset_name = f'{cfg.dataset.name}-r{cfg.dataset.repeats}-{cfg.dataset.build_algorithm}'
    
    if cfg.dataset.name == 'qm9':
        prepare_qm9(
            name=dataset_name,
            root=cfg.root,
            vocab=cfg.dataset.vocab,
            repeats=cfg.dataset.repeats,
            batch_size=cfg.dataset.batch_size,
            num_tokens=cfg.dataset.num_tokens,
            num_train= cfg.dataset.num_train,
            build_alg=cfg.dataset.build_algorithm,
            filter=cfg.dataset.filter
        )
    elif cfg.dataset.name == "grids":
        prepare_grids(cfg)

if __name__ == "__main__":
    main()
