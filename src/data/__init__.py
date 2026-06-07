"""Graph dataset construction and PyG data loading.

Converts FEM mesh/solution snapshots into PyTorch Geometric ``Data`` graphs:
node/edge feature engineering, coordinate transformations (e.g. relative edge
displacements), normalization statistics, and dataset/dataloader utilities.
"""
