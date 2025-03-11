from setuptools import setup

setup(
    name="g3",
    packages=["g3"],
    install_requires=[
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torch_geometric==2.4.0",
        "ogb==1.3.6",
        "hydra-core==1.3.2",
        "wandb==0.15.12",
        "performer-pytorch==1.1.4",
        "loguru==0.7.2",
        "tqdm==4.66.1",
        "prettytable==3.9.0",
        "rdkit==2023.9.1",
        "pandas==2.1.2",
    ],
    version="0.0.1",
)
