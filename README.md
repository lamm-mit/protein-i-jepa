# Protein-I-JEPA

This repository contains a small, readable implementation of an I-JEPA-style
self-supervised model for protein sequences.

The goal is not to compete with ESM or AlphaFold-scale systems. The goal is to
teach and test the JEPA idea on a real scientific data type: amino-acid
sequences.

## What JEPA Means

JEPA stands for Joint Embedding Predictive Architecture. The important idea is
that the model predicts **representations** of hidden information, not the raw
hidden tokens.

A masked language model does this:

```text
visible protein sequence -> predict missing amino-acid identities
```

This Protein-I-JEPA does this:

```text
visible protein sequence -> predict the target encoder's latent vectors
```

The training loop has three learned components:

- Context encoder: sees a protein sequence where contiguous residue spans have
  been replaced by a mask token.
- Target encoder: sees the unmasked sequence and produces target latent vectors.
  It is an exponential-moving-average copy of the context encoder.
- Predictor: maps the context encoder output at masked positions to the target
  latent vectors at those positions.

The loss compares predicted target latents to target encoder latents. It does
not ask the model to reconstruct the exact amino-acid sequence.

This is why the model is JEPA-style: it learns by predicting missing information
in representation space.

## What This Code Does

The code supports three workflows:

1. Self-supervised JEPA pretraining on unlabeled protein sequences.
2. Downstream probing to test whether the learned encoder contains biological
   information.
3. Dataset conversion helpers for creating professional secondary-structure
   probe splits.

The main entry point is:

```bash
python scripts/train_protein_jepa.py
```

Reusable package code lives in `src/protein_jepa/`:

- `alphabet.py`: amino-acid tokenization.
- `data.py`: FASTA, Hugging Face, synthetic datasets, and batch collation.
- `masking.py`: contiguous span masking.
- `model.py`: context encoder, EMA target encoder, and latent predictor.
- `train.py`: JEPA pretraining loop.
- `probe.py`: secondary-structure probe.
- `metrics.py`: JSONL/CSV logging and PNG/SVG plots.
- `visualize.py`: predicted-vs-target latent embedding plots.
- `report.py`: Markdown report generation from saved run directories.
- `download_secondary.py`: download/convert secondary-structure labels to TSV.
- `download_netsurfp.py`: download/convert NetSurfP train/validation/test splits
  to TSV.
- `publish_hf_secondary.py`: stage and upload NetSurfP probe splits to a
  Hugging Face dataset repo.

Tests live in `tests/`.

## Install And Test

The local environment already has PyTorch. To run the unit and smoke tests:

```bash
python -m unittest discover -s tests
```

For Hugging Face datasets and dataset publishing, install the optional
dependencies if needed:

```bash
python -m pip install datasets huggingface_hub
```

## End-To-End Runbook

This is the full command sequence for a real run: train JEPA on UniRef50,
prepare professional secondary-structure splits, train probes, evaluate external
tests, and build one Markdown report containing the metrics and figures.

This workflow writes NetSurfP-derived TSV files under `data/netsurfp/`, trains
JEPA, trains probes, evaluates CB513/TS115/CASP12, and writes one report.

1. Confirm the code works:

```bash
python -m unittest discover -s tests
```

2. Download and convert the NetSurfP-3.0 secondary-structure splits:

```bash
python scripts/download_netsurfp.py \
  --output-dir data/netsurfp
```

This uses the official NetSurfP-3.0 dataset page by default:
<https://services.healthtech.dtu.dk/services/NetSurfP-3.0/5-Dataset.php>.
It creates:

- `data/netsurfp/train.tsv`: train the probe.
- `data/netsurfp/validation.tsv`: tune the probe.
- `data/netsurfp/cb513.tsv`: final external test.
- `data/netsurfp/ts115.tsv`: final external test.
- `data/netsurfp/casp12.tsv`: final external test.

If you have a CASP14_FM file, add it with `--casp14-fm-npz path/to/file.npz` or
`--casp14-fm-tsv path/to/file.tsv`; it will be written as
`data/netsurfp/casp14_fm.tsv`.

3. Optionally publish the NetSurfP splits to Hugging Face:

```bash
huggingface-cli login

python scripts/publish_netsurfp_to_hf.py \
  --repo-id lamm-mit/protein-secondary-structure-netsurfp \
  --output-dir data/netsurfp \
  --staging-dir data/netsurfp_hf
```

Use `--dry-run` first if you want to verify the staged JSONL/TSV files without
uploading. The uploaded repo contains JSONL files for direct Hugging Face use
and TSV copies under `tsv/`.

4. Pretrain Protein-I-JEPA on a bounded UniRef50 sample:

```bash
python scripts/train_protein_jepa.py pretrain \
  --hf-dataset lamm-mit/UniRef50_512_all \
  --max-sequences 10000 \
  --steps 1000 \
  --batch-size 32 \
  --max-length 256 \
  --device auto \
  --output-dir runs/uniref50_jepa
```

5. Train a frozen secondary-structure probe on the JEPA encoder.

Local TSV version:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --train-labels-tsv data/netsurfp/train.tsv \
  --val-labels-tsv data/netsurfp/validation.tsv \
  --test-labels-tsv data/netsurfp/cb513.tsv \
  --test-labels-tsv data/netsurfp/ts115.tsv \
  --test-labels-tsv data/netsurfp/casp12.tsv \
  --steps 500 \
  --batch-size 32 \
  --max-length 256 \
  --device auto \
  --output-dir runs/secondary_probe_jepa
```

Hugging Face version after publishing:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --hf-dataset lamm-mit/protein-secondary-structure-netsurfp \
  --hf-train-split train \
  --hf-val-split validation \
  --hf-test-split cb513 \
  --hf-test-split ts115 \
  --hf-test-split casp12 \
  --steps 500 \
  --batch-size 32 \
  --max-length 256 \
  --device auto \
  --output-dir runs/secondary_probe_jepa
```

6. Train a scratch probe baseline with no JEPA checkpoint:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --train-labels-tsv data/netsurfp/train.tsv \
  --val-labels-tsv data/netsurfp/validation.tsv \
  --test-labels-tsv data/netsurfp/cb513.tsv \
  --test-labels-tsv data/netsurfp/ts115.tsv \
  --test-labels-tsv data/netsurfp/casp12.tsv \
  --steps 500 \
  --batch-size 32 \
  --max-length 256 \
  --device auto \
  --output-dir runs/secondary_probe_scratch
```

7. Optionally fine-tune the JEPA encoder during probing:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --train-labels-tsv data/netsurfp/train.tsv \
  --val-labels-tsv data/netsurfp/validation.tsv \
  --test-labels-tsv data/netsurfp/cb513.tsv \
  --test-labels-tsv data/netsurfp/ts115.tsv \
  --test-labels-tsv data/netsurfp/casp12.tsv \
  --unfreeze-encoder \
  --steps 500 \
  --batch-size 32 \
  --max-length 256 \
  --device auto \
  --output-dir runs/secondary_probe_finetuned
```

8. Plot predicted versus target JEPA latents in 2D:

```bash
python scripts/train_protein_jepa.py plot-embeddings \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --hf-dataset lamm-mit/UniRef50_512_all \
  --max-sequences 512 \
  --num-batches 8 \
  --max-points 2000 \
  --device auto \
  --output-dir runs/uniref50_jepa
```

This writes `embedding_predicted_vs_target.png` and
`embedding_predicted_vs_target.svg`.

9. Build a report that embeds all generated figures:

```bash
python scripts/make_training_report.py \
  --pretrain-dir runs/uniref50_jepa \
  --probe-dir runs/secondary_probe_jepa \
  --probe-dir runs/secondary_probe_scratch \
  --probe-dir runs/secondary_probe_finetuned \
  --output runs/reports/uniref50_jepa_report.md
```

The report is written to `runs/reports/uniref50_jepa_report.md`. It embeds the PNG
figures and links the SVG versions so you can use either bitmap or vector
graphics in slides and documents.

If you do not have a labeled secondary-structure dataset yet, run the complete
synthetic smoke workflow instead:

```bash
python scripts/train_protein_jepa.py pretrain \
  --synthetic \
  --steps 10 \
  --batch-size 8 \
  --max-length 96 \
  --device auto \
  --output-dir runs/smoke

python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/smoke/protein_jepa.pt \
  --synthetic \
  --steps 10 \
  --batch-size 8 \
  --max-length 96 \
  --device auto \
  --output-dir runs/probe-smoke

python scripts/train_protein_jepa.py plot-embeddings \
  --checkpoint runs/smoke/protein_jepa.pt \
  --synthetic \
  --num-batches 2 \
  --max-points 500 \
  --device auto \
  --output-dir runs/smoke

python scripts/make_training_report.py \
  --pretrain-dir runs/smoke \
  --probe-dir runs/probe-smoke \
  --output runs/reports/smoke_report.md
```

## Pretraining

Pretraining uses unlabeled sequences only. Each batch is tokenized, contiguous
residue spans are masked, and the model learns to predict the target encoder's
latent vectors at those masked positions.

Run a tiny synthetic smoke job:

```bash
python scripts/train_protein_jepa.py pretrain \
  --synthetic \
  --steps 10 \
  --batch-size 8 \
  --max-length 96 \
  --output-dir runs/smoke
```

Train from a local FASTA file:

```bash
python scripts/train_protein_jepa.py pretrain \
  --fasta data/uniref_sample.fasta \
  --steps 1000 \
  --batch-size 32 \
  --output-dir runs/protein_jepa
```

Train from the Hugging Face UniRef50 dataset:

```bash
python scripts/train_protein_jepa.py pretrain \
  --hf-dataset lamm-mit/UniRef50_512_all \
  --max-sequences 10000 \
  --steps 1000 \
  --batch-size 32 \
  --output-dir runs/uniref50_jepa
```

The default Hugging Face fields match `lamm-mit/UniRef50_512_all`:

- `Sequence`: amino-acid sequence.
- `Seq_Length`: sequence length.

For another dataset, override the field names:

```bash
python scripts/train_protein_jepa.py pretrain \
  --hf-dataset owner/dataset_name \
  --hf-sequence-field sequence \
  --hf-length-field length \
  --max-sequences 10000
```

For very large datasets, keep `--max-sequences` set. You can also pass a split
slice:

```bash
python scripts/train_protein_jepa.py pretrain \
  --hf-dataset lamm-mit/UniRef50_512_all \
  --hf-split 'train[:50000]'
```

## Pretraining Outputs

Each pretraining run writes these files to `--output-dir`:

- `config.json`: exact run configuration.
- `protein_jepa.pt`: checkpoint containing model weights, config, alphabet, and
  final metrics.
- `metrics.jsonl`: one JSON object per logging step.
- `metrics.csv`: same metrics in a spreadsheet-friendly format.
- `training_curves.png`: raster plot for reports and notebooks.
- `training_curves.svg`: vector plot for slides and papers.
- `embedding_predicted_vs_target.png`: 2D PCA plot of predicted and target JEPA
  latents, if you run `plot-embeddings`.
- `embedding_predicted_vs_target.svg`: vector version of the embedding plot.

The logged metrics include:

- `train_loss`: total JEPA training loss.
- `latent_loss`: representation prediction loss.
- `variance_loss`: small anti-collapse regularizer.
- `val_loss`: held-out JEPA validation loss.
- `latent_cosine`: cosine similarity between predicted and target latents on
  the training batch.
- `val_cosine`: same alignment metric on validation batches.
- `pred_std`: standard deviation of predicted latents.
- `target_std`: standard deviation of target latents.
- `targets_per_batch`: number of masked residue positions used for the JEPA
  objective.

How to read the curves:

- Falling `train_loss` and `val_loss` means the predictor is getting better at
  matching target latents.
- Rising or stable positive `val_cosine` means predicted latents are becoming
  directionally aligned with target latents.
- Very small `pred_std` can indicate representation collapse, where the model
  predicts nearly the same vector everywhere.
- A large train/validation gap suggests overfitting to the sampled sequences or
  too little validation data.

## Embedding Visualization

The `plot-embeddings` command visualizes what the JEPA objective is doing. It
loads a checkpoint, samples masked spans, collects two sets of vectors, and
projects them to two dimensions with PCA:

- predicted latents: the predictor output from the masked context.
- target latents: the EMA target encoder output from the unmasked sequence.

Run it on the same sequence source used for pretraining:

```bash
python scripts/train_protein_jepa.py plot-embeddings \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --hf-dataset lamm-mit/UniRef50_512_all \
  --max-sequences 512 \
  --num-batches 8 \
  --max-points 2000 \
  --device auto \
  --output-dir runs/uniref50_jepa
```

The plot is saved as:

- `embedding_predicted_vs_target.png`
- `embedding_predicted_vs_target.svg`

How to interpret it:

- If predicted and target clouds overlap more over training, the predictor is
  learning to match the target latent distribution.
- Shorter gray lines between matched predicted/target points indicate better
  per-position latent prediction in the 2D projection.
- If predicted points collapse into a tiny cluster while target points remain
  spread out, the model may be collapsing or the predictor may be too weak.
- This is a qualitative visualization. Use `val_loss`, `val_cosine`, and
  downstream probe accuracy for quantitative judgment.

## Probes

A probe is a small supervised model trained on top of the learned encoder. It is
used after self-supervised pretraining to ask:

> Did the unlabeled JEPA model learn representations that contain useful
> biological information?

The implemented probe is a Q3 secondary-structure probe.

Secondary structure is a per-residue label:

- `H`: helix.
- `E`: beta strand.
- `C`: coil/other.

The probe takes a sequence, runs it through the JEPA context encoder, and trains
a small linear classifier to predict one secondary-structure label per residue.
By default the encoder is frozen. That matters: if the frozen encoder performs
well, the structural information was already present in the representation
learned from unlabeled sequences.

## Probe Data Format

The secondary-structure probe expects a TSV file with two columns:

```text
sequence	labels
ACDEFGHIK	CCHHHCEEE
```

The `sequence` column contains amino-acid sequences. The `labels` column
contains one secondary-structure character per residue.

Labels may be Q3:

- `C`
- `E`
- `H`

or DSSP-style Q8:

- helix-like: `H`, `G`, `I`
- strand-like: `E`, `B`
- coil-like: `C`, `S`, `T`, `-`

Q8 labels are automatically mapped to Q3.

The special label `.` means "ignore this residue for the supervised loss and
accuracy." The NetSurfP converter uses this to respect external-test evaluation
masks while preserving the full sequence context.

For a serious probe workflow, create explicit train, validation, and external
test TSV files from NetSurfP-3.0:

```bash
python scripts/download_netsurfp.py \
  --output-dir data/netsurfp
```

The resulting split policy is:

- `data/netsurfp/train.tsv`: train the supervised probe.
- `data/netsurfp/validation.tsv`: tune probe settings and pick checkpoints.
- `data/netsurfp/cb513.tsv`: final external test.
- `data/netsurfp/ts115.tsv`: final external test.
- `data/netsurfp/casp12.tsv`: final external test.
- `data/netsurfp/casp14_fm.tsv`: optional final external test if you provide
  `--casp14-fm-npz` or `--casp14-fm-tsv`.

You can also stage and upload those splits into the `lamm-mit` Hugging Face
account:

```bash
huggingface-cli login

python scripts/publish_netsurfp_to_hf.py \
  --repo-id lamm-mit/protein-secondary-structure-netsurfp \
  --output-dir data/netsurfp \
  --staging-dir data/netsurfp_hf
```

That command downloads/converts NetSurfP, writes upload-ready JSONL files, keeps
TSV copies, creates a dataset card, and uploads the folder through the Hugging
Face Hub API. Use `--dry-run` to do everything except the upload.

Train a frozen probe from a JEPA checkpoint and evaluate the external tests:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --train-labels-tsv data/netsurfp/train.tsv \
  --val-labels-tsv data/netsurfp/validation.tsv \
  --test-labels-tsv data/netsurfp/cb513.tsv \
  --test-labels-tsv data/netsurfp/ts115.tsv \
  --test-labels-tsv data/netsurfp/casp12.tsv \
  --steps 500 \
  --output-dir runs/secondary_probe_jepa
```

Or train the same probe from the uploaded Hugging Face dataset:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --hf-dataset lamm-mit/protein-secondary-structure-netsurfp \
  --hf-train-split train \
  --hf-val-split validation \
  --hf-test-split cb513 \
  --hf-test-split ts115 \
  --hf-test-split casp12 \
  --steps 500 \
  --output-dir runs/secondary_probe_jepa
```

Train a scratch baseline with the same explicit splits:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --train-labels-tsv data/netsurfp/train.tsv \
  --val-labels-tsv data/netsurfp/validation.tsv \
  --test-labels-tsv data/netsurfp/cb513.tsv \
  --test-labels-tsv data/netsurfp/ts115.tsv \
  --test-labels-tsv data/netsurfp/casp12.tsv \
  --steps 500 \
  --output-dir runs/secondary_probe_scratch
```

Fine-tune the encoder instead of freezing it:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --train-labels-tsv data/netsurfp/train.tsv \
  --val-labels-tsv data/netsurfp/validation.tsv \
  --test-labels-tsv data/netsurfp/cb513.tsv \
  --test-labels-tsv data/netsurfp/ts115.tsv \
  --test-labels-tsv data/netsurfp/casp12.tsv \
  --unfreeze-encoder \
  --steps 500 \
  --output-dir runs/secondary_probe_finetuned
```

The original simple dataset remains supported. For quick demos, the single-file
mode randomly splits one TSV into train and validation:

```bash
python scripts/download_secondary_structure.py \
  --output data/secondary_structure.tsv

python scripts/train_protein_jepa.py probe-secondary \
  --labels-tsv data/secondary_structure.tsv \
  --steps 500 \
  --output-dir runs/secondary_probe_quick
```

Run a synthetic probe smoke test:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/smoke/protein_jepa.pt \
  --synthetic \
  --steps 10 \
  --batch-size 8 \
  --max-length 96 \
  --output-dir runs/probe-smoke
```

## Probe Outputs

Each probe run writes:

- `probe_config.json`: exact probe configuration.
- `secondary_probe.pt`: probe checkpoint.
- `metrics.jsonl`: one JSON object per logging step.
- `metrics.csv`: spreadsheet-friendly probe metrics.
- `test_metrics.json`: external test metrics, if `--test-labels-tsv` or
  `--hf-test-split` was used.
- `probe_curves.png`: raster plot.
- `probe_curves.svg`: vector plot.

The logged probe metrics include:

- `train_loss`: supervised cross-entropy on labeled residues.
- `val_loss`: held-out supervised cross-entropy.
- `train_q3`: per-residue Q3 accuracy on the training batch.
- `val_q3`: per-residue Q3 accuracy on validation batches.
- `test_cb513_q3`, `test_ts115_q3`, `test_casp12_q3`: final external-test Q3
  accuracy when those test splits are provided.

What the probe tells us:

- If JEPA + frozen probe beats a scratch frozen encoder, the JEPA pretraining
  learned sequence features that help secondary-structure prediction.
- NetSurfP validation is for tuning. CB513, TS115, CASP12, and optional
  CASP14_FM should be treated as final external tests.
- If JEPA helps most when labeled data is scarce, it is improving label
  efficiency.
- If fine-tuning helps much more than frozen probing, the representation may be
  useful but not linearly accessible.
- If probe validation accuracy is poor while JEPA validation loss looks good,
  the model may be learning predictable sequence statistics that do not transfer
  to the biological property being tested.

The probe does not prove that the model understands full 3D structure. It tests
whether local structural information is present in the learned sequence
representations.

## Validation Already Implemented

The repository currently includes:

- Held-out JEPA validation loss.
- Latent cosine similarity on train and validation batches.
- Predicted and target latent standard deviation diagnostics for collapse.
- Q3 secondary-structure probing with frozen or fine-tuned encoders.
- Explicit NetSurfP train/validation split probing.
- External test evaluation on CB513, TS115, CASP12, and optional CASP14_FM from
  local TSVs or Hugging Face splits.
- Synthetic smoke tests for pretraining and probing.

## Good Next Validation Tasks

These are not implemented yet, but they are natural extensions:

- Solvent accessibility: per-residue buried/exposed prediction.
- Disorder prediction: per-residue ordered/disordered classification.
- Remote homology: protein-level fold or family classification with pooled
  embeddings.
- Fitness or stability prediction: mutation-effect regression/classification.
- Masked-token baseline: train an MLM on the same sequence sample and compare
  downstream probes against JEPA.

For serious scientific evaluation, prefer curated or homology-aware external
tests. Random single-TSV splits can overestimate performance because related
proteins may appear in both training and validation sets.
