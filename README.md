# Generalist Generative Graph Models (G3)

## Install

```bash
conda create --name gen-gpt python=3.10
pip install -e .
```
which will install `g3` and all dependencies

## Run

Prepare Data
```bash
python prepare_data repeats=32 
```

Train Model
```bash
python train.py 
```

Generate 

```bash
python generate.py checkpoint=qm9-adjbias steps=steps
```

## Design choices
- there is no start token its assumed to be carbon