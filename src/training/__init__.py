"""Training, noise injection, and RunPod orchestration.

Holds the training/evaluation loops, the training-noise injection mechanism
(for stable autoregressive rollouts), checkpointing, and helpers for launching
and managing GPU training jobs on RunPod.
"""
