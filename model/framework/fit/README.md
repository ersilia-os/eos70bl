# Fit folder — CapMolPred (eos70bl)

`Source Type: Replicated`. The original authors' repository
(https://github.com/Roxie2816/capsule-antimicrobial) ships open training code and
data, but **no trained classifier checkpoints** for any of the three organisms — only
the two upstream pretrained encoders (`Mole-BERT.pth`, `gin_model.pth`, already copied
into `model/checkpoints/`) are included. This was verified against the GitHub repo, the
author's fork, the archived Zenodo record, and the web portal's own download page — none
of them expose the final capsule-network classifier weights.

**Status: training has been run.** The three classifiers were retrained end-to-end (see
below for exactly what was done and how). This file is kept as the record of what was
changed and why, in case retraining needs to be repeated (e.g. a future architecture
change, or a dependency stops resolving).

## Fixes applied to the copied source before retraining

The files in `src/` are copies of the upstream repository's training code. Several
corrections were required — the upstream code did not run as released:

1. **`frozen_fusion.py` (new file).** The upstream `crossfusion.py::create_molecular_model()`
   — the cross-attention module that fuses the five pretrained-LM embeddings into the
   1536-dim vector the capsule classifier trains on — is built fresh with randomly
   initialized weights on *every call*, and is never trained or saved (only `.predict()`
   is used on it). That means the same molecule gets a different feature vector every
   time the Python process restarts, so predictions are not reproducible as released.
   `frozen_fusion.py` seeds every RNG, builds `create_molecular_model()` exactly once,
   and persists its weights so training and inference always load the identical
   projection. This was built **once** (`fusion_weights.weights.h5`) and reused for all
   three organisms — never rebuilt per organism, or each would get a different frozen
   projection and the three classifiers would be mutually inconsistent.
2. **`feature_extraction.py`**: the single call site (`get_feature()`, `drug_descripter
   == 'fusion'` branch) now calls `load_frozen_fusion(FUSION_WEIGHTS_PATH)` instead of
   `create_molecular_model()` directly. Also removed a line setting `HF_HOME` to a URL
   (`https://hf-mirror.com`) — `HF_HOME` must be a local cache directory, not a URL, and
   as written it would have broken `transformers`' cache resolution.
3. **`MolFormer.py`**: the upstream `molformer()` loads a raw PyTorch-Lightning checkpoint
   (`./model/N-Step-Checkpoint_3_30000.ckpt`) that is **not shipped in the CapMolPred
   repo**, and IBM's original download link for it (https://ibm.box.com/v/MoLFormer-data)
   is dead (404, confirmed). IBM's official HuggingFace port
   `ibm-research/MoLFormer-XL-both-10pct` is the same underlying pretrained model —
   verified by comparing its `config.json` (hidden_size=768, num_attention_heads=12,
   num_hidden_layers=12, max_position_embeddings=202, num_random_features=32) against the
   exact values in `model/hparams.yaml` (n_embd/n_head/n_layer/max_len/num_feats,
   `model_arch: BERT_16GPU_Both_10percent_rotate_no_masking`). `molformer()` now loads
   this via `transformers.AutoModel`/`AutoTokenizer` instead; `model/hparams.yaml` and
   `model/bert_vocab.txt` are no longer read by this encoder (kept for reference). The
   now-unused `fast_transformers`-based legacy classes lower in the same file were left
   in place (harmless, still importable) rather than deleted.
4. **`bert_capsule_model.py`**: two upstream bugs that would crash on import/execution
   regardless of any of the above — `Union`/`Callable` used in a type hint but never
   imported from `typing`; and a hardcoded `focal_plus(...)` loss in
   `model_bert_chemmolefusion_capsule()`'s initial `.compile()` call, where `focal_plus`
   is never defined anywhere in the repo (grepped the full source tree). This compile
   call's loss is always immediately overwritten by the caller
   (`drugclassification.py`'s `fitting()` recompiles with `ASL(...)` before any `.fit()`
   happens), so it was replaced with a harmless placeholder (`'binary_crossentropy'`)
   rather than trying to guess the authors' intended loss.
5. **`data/ecoli.csv`**: one row (a sucralfate-like aluminum-sulfate complex, several
   `[AlH3]` centers) fails RDKit sanitization (`Explicit valence for atom # 16 Al, 6, is
   greater than permitted`) and crashed the full training run with no Python traceback
   (a hard crash, not a catchable exception) after feature extraction had already run on
   all other 2,334 compounds. Dropped that single row before retraining; `baumannii.csv`
   and `saureus_external_val.csv` were checked the same way and had zero unsanitizable
   rows, so were left untouched.

## `data/` folder

Copied from the source repo with the one exception noted above (fix 5); no other
processing applied:
- `ecoli.csv`, `baumannii.csv`, `saureus_external_val.csv` — full datasets (SMILES,
  Activity), 2,335 / 7,684 / 2,984 compounds respectively.
- `5FoldScaffoldSplit/{ecoli,baumannii,saureus}_no_overlap/` — the authors' own 5-fold
  scaffold-split train/test CSVs per organism (used internally by `drugclassification.py`
  for its own train/val split, not something you need to invoke directly).

## `src/` folder

Copied from the source repo with the fixes above applied in place (not separate patch
files, to keep `drugclassification.py`'s imports working unmodified):
`drugclassification.py` (entry point), `feature_extraction.py`, `crossfusion.py`,
`bert_capsule_model.py`, `Capsule_MPNN.py`, `Mole_Bert.py`, `MolFormer.py`,
`ginet_molclr.py`, plus the new `frozen_fusion.py`.

## Environment actually used

The upstream README's stated pins (TensorFlow 2.20.0, PyTorch 1.11.0) are not jointly
resolvable/compatible in practice — TensorFlow ≥2.16 forces Keras 3, which this code's
mixed `tensorflow.keras`/bare `keras` imports are not safe under, and PyTorch 1.11.0 is
no longer available from the standard index. What was actually used, in a dedicated
`conda` env (`capmolpred-train`, Python 3.10), which does **not** need to match
`install.yml` (the CPU-only packaged-model runtime) since this step never runs inside the
shipped Ersilia model:

- `tensorflow-cpu==2.15.1` (bundles Keras 2.15.0 natively — avoids the Keras-3 mixed-import
  hazard entirely; no `tf_keras`/`TF_USE_LEGACY_KERAS` shim needed)
- `torch==2.6.0+cpu` (from `https://download.pytorch.org/whl/cpu`) — needed `>=2.6` because
  `transformers`' safe-loading guard (CVE-2025-32434) refuses `torch.load` of non-safetensors
  checkpoints (used internally by `BertModel.from_pretrained` for the ChemBERTa-2/Smole-BERT
  encoders) below that version
- `torch_geometric` + `torch_scatter==2.1.2+pt26cpu` (from `https://data.pyg.org/whl/torch-2.6.0+cpu.html`)
- `transformers` (latest; needed for the MolFormer HF port's remote code, which requires
  `transformers.masking_utils` — not present in the originally-planned `4.39.3`)
- `pytorch-lightning==2.0.3`, `pytorch-fast-transformers==0.4.0` (builds fine against
  whichever torch is installed; ends up unused after fix 3 above, but kept installed
  since `MolFormer.py`'s legacy classes still import it at module level)
- `rdkit`, `scikit-learn`, `pandas==2.3.1`, `scipy` (>=1.13, newer than the 1.10.1 originally
  planned — `tensorflow-cpu==2.20.0` pulled numpy≥2 which conflicted with 1.10.1; resolved
  by switching to `tensorflow-cpu==2.15.1` instead, which is numpy<2-compatible, and using a
  newer scipy for headroom), `matplotlib==3.9.2`, `tqdm==4.67.1`, `setuptools<81` (needed for
  `pkg_resources`, required by `pytorch_lightning` but removed by default in newer setuptools)

## Steps actually run

From `model/framework/fit/src/`:

```bash
mkdir -p model
cp ../../checkpoints/Mole-BERT.pth ../../checkpoints/gin_model.pth \
   ../../checkpoints/bert_vocab.txt ../../checkpoints/hparams.yaml model/
mkdir -p output   # required - drugclassification.py does not create this itself

python frozen_fusion.py build fusion_weights.weights.h5
```

Then, **for each organism**, in its own working directory (each needs its own `model/`
checkpoints copy, `output/`, and the frozen fusion weights, since `drugclassification.py`
and `feature_extraction.py` use relative `./model/...`/`./output` paths):

```bash
python drugclassification.py --input-path data.csv \
  --model-name <organism>_bert_chemmolefusion_capsule --drug-descripter fusion \
  --train --batch-size 64 -e 100 -dp data
```

**The `--model-name` value must contain the literal substring `bert_chemmolefusion_capsule`.**
`drugclassification.py`'s `fitting()` gates its entire training branch on
`if "bert_chemmolefusion_capsule" in model_type:` — anything else silently skips training
and later crashes with `UnboundLocalError: local variable 'history' referenced before
assignment`. This is not documented anywhere in the repo or its README; discovered by
smoke-testing on a 30-row balanced sample before committing to full-scale runs. Organism
names were disambiguated as `ecoli_bert_chemmolefusion_capsule`,
`baumannii_bert_chemmolefusion_capsule`, `saureus_bert_chemmolefusion_capsule`.

Each run performs the authors' own hyperparameter search over `drug_dense ∈ {200, 400}`,
`kernel_size ∈ {5, 10}`, `routings ∈ {3, 6}` (`num_capsule` fixed at 2) with 5-fold CV
(40 fits total), then refits the best configuration on the full training split. The paper
does not state which configuration was finally deployed, so this reproduces the authors'
own selection procedure rather than guessing one config.

**Outputs**, per organism, under
`<organism>_bert_chemmolefusion_capsule/<organism>_bert_chemmolefusion_capsule/`:
- `checkpoints/best_model.h5` — the trained classifier
- sibling `performance.txt` and `output/5fold-performance.csv` — held-out metrics, to be
  compared against the paper's reported values (Tables 1–2, Fig. 2G) before accepting

Final artifacts to persist in `model/checkpoints/`: `fusion_weights.weights.h5` (built
once, shared across all three) plus `ecoli_classifier.h5`, `baumannii_classifier.h5`,
`saureus_classifier.h5` (each organism's `best_model.h5`, renamed).

**Note on file size:** `fusion_weights.weights.h5` is ~118 MB (four `CrossAttentionLayer`s
at hidden_dim=768/12 heads/intermediate_dim=3072, per the untrained-fusion architecture in
fix 1 above) — over the 100 MB threshold where Ersilia usually asks whether to store a
checkpoint via Git LFS or `eosvc`. Since `model/checkpoints/` is gitignored in this
template regardless (see the repo's `CLAUDE.md`: persist via `eosvc`, never `git add`),
this doesn't change how it's stored, but flagging the size here for visibility.

## `results/` folder

Leave the per-organism `performance.txt` / `5fold-performance.csv` outputs (or copies of
them) here alongside a short note comparing them to the paper's reported metrics, per
the template convention.
