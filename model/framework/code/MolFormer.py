import torch
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
import pandas as pd


def molformer(smiles_list: List[str]) -> pd.DataFrame:

    # ERSILIA FIX: the upstream code loads a raw PyTorch-Lightning checkpoint
    # (`./model/N-Step-Checkpoint_3_30000.ckpt`) that is not shipped in the
    # CapMolPred repo, and IBM's original download link
    # (https://ibm.box.com/v/MoLFormer-data) is dead (404). IBM's official
    # HuggingFace port `ibm-research/MoLFormer-XL-both-10pct` is the same
    # underlying pretrained model - verified by comparing its config.json
    # (hidden_size=768, num_attention_heads=12, num_hidden_layers=12,
    # max_position_embeddings=202, num_random_features=32) against the exact
    # values in this repo's `model/hparams.yaml` (n_embd/n_head/n_layer/
    # max_len/num_feats, model_arch: BERT_16GPU_Both_10percent_rotate_no_masking).
    # Loaded here via `transformers.AutoModel` instead of the dead-checkpoint
    # LightningModule path.
    #
    # This file also drops the upstream module's large amount of unused
    # legacy code (a custom Encoder, LightningModule and fast_transformers-
    # based rotary-attention classes used only by the old, dead-checkpoint
    # loading path) - none of it is reachable from molformer() any more, and
    # keeping it would require pytorch_lightning and pytorch-fast-transformers
    # (a fragile C-extension package) as runtime dependencies for no benefit.
    # The full original file, with only the loader swapped, is kept as the
    # historical record in model/framework/fit/src/MolFormer.py.
    _MOLFORMER_HF_REPO = "ibm-research/MoLFormer-XL-both-10pct"

    def _load_model():
        """load pretrained model and tokenizer from the HF-hosted checkpoint"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(_MOLFORMER_HF_REPO, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                _MOLFORMER_HF_REPO, trust_remote_code=True, deterministic_eval=True
            )
            model.eval()
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Fail to load model: {str(e)}")

    def _batch_split(data: List[str], batch_size: int = 64):
        """split data into batches"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def _canonicalize(smiles: str) -> Optional[str]:
        """canonical SMILES str"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol else None
        except Exception:
            return None

    def _embed(model, tokenizer, smiles_batch: List[str]) -> torch.Tensor:
        """generate embedding for one batch"""
        batch_enc = tokenizer(
            smiles_batch,
            padding=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            token_embeddings = model(**batch_enc).last_hidden_state

        mask = batch_enc['attention_mask'].unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu()

    try:
        valid_smiles = []
        for s in smiles_list:
            canon = _canonicalize(s)
            if canon:
                valid_smiles.append(canon)

        if not valid_smiles:
            raise ValueError("No valid SMILES input detected.")

        model, tokenizer = _load_model()

        # generate embeddings
        embeddings = []
        for batch in _batch_split(valid_smiles):
            embeddings.append(_embed(model, tokenizer, batch))
        embeddings_array = torch.cat(embeddings).numpy()

        columns = [f"molformer_{i}" for i in range(embeddings_array.shape[1])]
        df = pd.DataFrame(embeddings_array, columns=columns)
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
