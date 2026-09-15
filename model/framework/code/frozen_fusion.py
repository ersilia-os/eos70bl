"""Ersilia fix for CapMolPred's non-reproducible cross-attention fusion layer.

The upstream `crossfusion.py::create_molecular_model()` builds a brand-new
`MolecularCrossAttention` Keras model with randomly-initialized weights on
every call, and it is never trained (only `.predict()` is used on it) nor
persisted. That means the 1536-dim feature vector handed to the capsule
classifier differs across separate Python process runs, so predictions for
the *same* molecule are not reproducible once the fusion step is re-run in
a fresh process (as happens for every new molecule submitted to the
authors' own web server, and would happen for every `ersilia run` call).

This module seeds every relevant RNG, builds `create_molecular_model()`
exactly once, and persists those (still-untrained-by-design, but now fixed)
weights to disk. Both the one-off training-data feature extraction and the
Ersilia model's runtime inference must load this same saved file so they
see an identical projection.

Usage:
    # one-off, during training data preparation:
    python frozen_fusion.py build /path/to/fusion_weights.weights.h5

    # from other scripts, at train or inference time:
    from frozen_fusion import load_frozen_fusion
    fusion_model = load_frozen_fusion("/path/to/fusion_weights.weights.h5")
"""
import os
import sys

import numpy as np
import tensorflow as tf

from crossfusion import create_molecular_model

FUSION_SEED = 42


def _seed_everything(seed=FUSION_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_frozen_fusion(weights_path):
    """Create create_molecular_model() once with a fixed seed and persist
    its (untrained-by-design) weights to `weights_path`. Run this exactly
    once; every later training or inference call should use
    load_frozen_fusion() against the same file instead of calling this
    again, or a different random draw would be produced."""
    _seed_everything()
    model = create_molecular_model()
    os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
    model.save_weights(weights_path)
    print(f"Saved frozen fusion weights to {weights_path}")
    return model


def load_frozen_fusion(weights_path):
    """Rebuild the identical architecture and load the frozen weights
    saved by build_frozen_fusion(). Raises if the file is missing, rather
    than silently falling back to a fresh random instantiation."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Frozen fusion weights not found at {weights_path}. "
            "Run `python frozen_fusion.py build <weights_path>` once first, "
            "and reuse the resulting file for every later train/predict call."
        )
    _seed_everything()
    model = create_molecular_model()
    model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "build":
        print("Usage: python frozen_fusion.py build <weights_path>")
        sys.exit(1)
    build_frozen_fusion(sys.argv[2])
