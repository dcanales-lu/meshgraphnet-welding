"""MeshGraphNet architecture.

Encoder (node/edge MLP embeddings) -> Processor (stacked message-passing
GraphNet blocks with residual updates) -> Decoder (per-node output MLP)
implementing the MeshGraphNets formulation for the welding thermal surrogate.
"""

from models.meshgraphnet import (
    Decoder,
    Encoder,
    GraphNetBlock,
    MeshGraphNet,
    MeshGraphNetConfig,
    build_mlp,
)

__all__ = [
    "MeshGraphNet",
    "MeshGraphNetConfig",
    "Encoder",
    "GraphNetBlock",
    "Decoder",
    "build_mlp",
]
