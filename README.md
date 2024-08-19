## Introduction

This repository contains the source code for the publications [From Coupled Oscillators to Graph Neural Networks: Reducing Over-smoothing via a Kuramoto Model-based Approach].

The source code is modified from the code of Graph Neural Diffusion (GRAND) (https://github.com/twitter-research/graph-neural-pde). Please see the GRAND's repo in case of environment setting and troubleshooting.

## Running the experiments
A general command to run the code:
```
python3 run_GNN.py --method dopri5 --time $time --no_early --coupling_strength $coup --dataset $data
```

For example, for the Citeseer dataset:
```
python3 run_GNN.py --method dopri5 --time 16 --no_early --coupling_strength 0.9 --dataset Citeseer
```

### Requirements
Dependencies (with python >= 3.7):
Main dependencies are
torch==1.8.1
torch-cluster==1.5.9
torch-geometric==1.7.0
torch-scatter==2.0.6
torch-sparse==0.6.9
torch-spline-conv==1.2.1
torchdiffeq==0.2.1
Commands to install all the dependencies in a new conda environment
```
conda create --name grand python=3.7
conda activate grand

pip install ogb pykeops
pip install torch==1.8.1
pip install torchdiffeq -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-geometric
```
