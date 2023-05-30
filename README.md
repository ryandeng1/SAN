# SAN

Implementation of Spectral Attention Networks, a powerful GNN that leverages key principles from spectral graph theory to enable full graph attention.

![full_method](https://user-images.githubusercontent.com/47570400/119883871-046aa280-befe-11eb-9063-108f4fe1a123.png)

# Overview

* ```nets``` contains the Node, Edge and no LPE architectures implemented with PyTorch.
* ```layers``` contains the multi-headed attention employed by the Main Graph Transformer implemented in DGL.
* ```train``` contains methods to train the models.
* ```data``` contains dataset classes and various methods used in precomputation.
* ```configs``` contains the various parameters used in the ablation and SOTA comparison studies.
* ```misc``` contains scripts from https://github.com/graphdeeplearning/graphtransformer to download datasets and setup environments.
* ```scripts``` contains scripts to reproduce ablation and SOTA comparison results. See ```scripts/reproduce.md``` for details.


# Installation guide
1. Create a conda environment: `conda create -n SAN python=3.8`
2. Install pytorch (currently tested using Cuda 11.6) `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia`
3. Install dgl 1.1 to use their built in cora dataset `pip install --pre dgl -f https://data.dgl.ai/wheels-test/cu116/repo.html`
4. Install remaining needed libraries `pip install tensorboardX tqdm networkx`, although this may not be comprehensive
5. Script used to run on cora: `bash scripts/CORA/optimized`
