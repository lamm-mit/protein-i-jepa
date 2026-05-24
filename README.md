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

The code supports two workflows:

1. Self-supervised JEPA pretraining on unlabeled protein sequences.
2. Downstream probing to test whether the learned encoder contains biological
   information.
3. Dataset conversion helpers for creating the probe TSV.

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

Tests live in `tests/`.

## Install And Test

The local environment already has PyTorch. To run the unit and smoke tests:

```bash
python -m unittest discover -s tests
```

For Hugging Face datasets, install the optional dependency if needed:

```bash
python -m pip install datasets
```

## End-To-End Runbook

This is the full command sequence for a real run: train JEPA on UniRef50,
train secondary-structure probes, then build one Markdown report containing the
metrics and figures.

This workflow downloads a small labeled secondary-structure dataset to
`data/secondary_structure.tsv`, trains JEPA, trains probes, and writes one
report.

1. Confirm the code works:

```bash
python -m unittest discover -s tests
```

2. Download and convert a secondary-structure probe dataset:

```bash
python scripts/download_secondary_structure.py \
  --output data/secondary_structure.tsv
```

The default source is `lamm-mit/protein-secondary-structure-nppe2` from
Hugging Face. It provides protein sequences in `seq` and per-residue secondary
structure labels in `sst3` and `sst8`.

3. Pretrain Protein-I-JEPA on a bounded UniRef50 sample:

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

4. Train a frozen secondary-structure probe on the JEPA encoder:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --labels-tsv data/secondary_structure.tsv \
  --steps 500 \
  --batch-size 32 \
  --max-length 256 \
  --device auto \
  --output-dir runs/secondary_probe_jepa
```

5. Train a scratch probe baseline with no JEPA checkpoint:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --labels-tsv data/secondary_structure.tsv \
  --steps 500 \
  --batch-size 32 \
  --max-length 256 \
  --device auto \
  --output-dir runs/secondary_probe_scratch
```

6. Optionally fine-tune the JEPA encoder during probing:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --labels-tsv data/secondary_structure.tsv \
  --unfreeze-encoder \
  --steps 500 \
  --batch-size 32 \
  --max-length 256 \
  --device auto \
  --output-dir runs/secondary_probe_finetuned
```

7. Plot predicted versus target JEPA latents in 2D:

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

8. Build a report that embeds all generated figures:

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

Create this file from the default Hugging Face dataset:

```bash
python scripts/download_secondary_structure.py \
  --output data/secondary_structure.tsv
```

Or explicitly choose fields:

```bash
python scripts/download_secondary_structure.py \
  --hf-dataset lamm-mit/protein-secondary-structure-nppe2 \
  --split train \
  --sequence-field seq \
  --label-field sst3 \
  --output data/secondary_structure.tsv
```

Train a probe from a JEPA checkpoint:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --labels-tsv data/secondary_structure.tsv \
  --steps 500 \
  --output-dir runs/secondary_probe_jepa
```

Train a scratch baseline with the same probe code but no JEPA checkpoint:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --labels-tsv data/secondary_structure.tsv \
  --steps 500 \
  --output-dir runs/secondary_probe_scratch
```

Fine-tune the encoder instead of freezing it:

```bash
python scripts/train_protein_jepa.py probe-secondary \
  --checkpoint runs/uniref50_jepa/protein_jepa.pt \
  --labels-tsv data/secondary_structure.tsv \
  --unfreeze-encoder \
  --steps 500 \
  --output-dir runs/secondary_probe_finetuned
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
- `probe_curves.png`: raster plot.
- `probe_curves.svg`: vector plot.

The logged probe metrics include:

- `train_loss`: supervised cross-entropy on labeled residues.
- `val_loss`: held-out supervised cross-entropy.
- `train_q3`: per-residue Q3 accuracy on the training batch.
- `val_q3`: per-residue Q3 accuracy on validation batches.

What the probe tells us:

- If JEPA + frozen probe beats a scratch frozen encoder, the JEPA pretraining
  learned sequence features that help secondary-structure prediction.
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

For serious scientific evaluation, use homology-aware splits. Random splits can
overestimate performance because related proteins may appear in both training
and validation sets.
