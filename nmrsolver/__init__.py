"""
NMR Solver – Inverse ¹³C NMR → Molecular Structure Identification.

Modules
-------
config      – global paths, hyper-parameters, constants
data        – SQLite loading, tokenisation, dataset/dataloader
retrieval   – fast nearest-neighbour baseline (FAISS)
models      – forward and inverse Transformer models
scoring     – candidate ranking via forward-model rescoring
train       – training loops for forward / inverse models
predict     – end-to-end inference pipeline & CLI
"""

__version__ = "0.1.0"
