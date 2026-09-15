import os

import numpy as np
from rdkit import Chem
from tensorflow import keras

from Capsule_MPNN import Capsule, Length, TransformerEncoderReadout, squash
from feature_extraction import FUSION_WEIGHTS_PATH, cross_fusion
from frozen_fusion import load_frozen_fusion

_CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")

_CUSTOM_OBJECTS = {
    "Capsule": Capsule,
    "Length": Length,
    "TransformerEncoderReadout": TransformerEncoderReadout,
    "squash": squash,
}

# Column order returned by predict_batch(); must match run_columns.csv.
ORGANISMS = ("ecoli", "abaumannii", "saureus")
_CLASSIFIER_FILENAMES = {
    "ecoli": "ecoli_classifier.h5",
    "abaumannii": "baumannii_classifier.h5",
    "saureus": "saureus_classifier.h5",
}

_fusion_model = None
_classifiers = None


def _load_models():
    global _fusion_model, _classifiers
    if _fusion_model is None:
        _fusion_model = load_frozen_fusion(FUSION_WEIGHTS_PATH)
    if _classifiers is None:
        _classifiers = {
            organism: keras.models.load_model(
                os.path.join(_CHECKPOINTS_DIR, filename),
                compile=False,
                custom_objects=_CUSTOM_OBJECTS,
            )
            for organism, filename in _CLASSIFIER_FILENAMES.items()
        }
    return _fusion_model, _classifiers


def _is_valid(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def predict_batch(smiles_list):
    """Per-organism activity probability for each SMILES.

    Returns an (n, len(ORGANISMS)) float array, columns ordered as ORGANISMS.
    Rows for unparseable SMILES are NaN rather than crashing the whole batch.
    """
    fusion_model, classifiers = _load_models()

    valid_idx = [i for i, s in enumerate(smiles_list) if _is_valid(s)]
    valid_smiles = [smiles_list[i] for i in valid_idx]

    results = np.full((len(smiles_list), len(ORGANISMS)), np.nan, dtype=np.float32)

    if valid_smiles:
        fusion_input = cross_fusion(valid_smiles)
        features = fusion_model.predict(fusion_input)
        for col, organism in enumerate(ORGANISMS):
            proba = classifiers[organism].predict([features])
            active_proba = proba[:, 1]
            for row_pos, orig_idx in enumerate(valid_idx):
                results[orig_idx, col] = active_proba[row_pos]

    return results
