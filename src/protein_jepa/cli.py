from __future__ import annotations

import argparse
import json

from protein_jepa.probe import ProbeConfig, train_secondary_probe
from protein_jepa.train import TrainConfig, train
from protein_jepa.visualize import EmbeddingPlotConfig, plot_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small Protein-I-JEPA model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain", help="Run self-supervised JEPA pretraining.")
    pretrain.add_argument("--fasta", type=str, default=None, help="Path to a FASTA file.")
    pretrain.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face dataset name, e.g. lamm-mit/UniRef50_512_all.")
    pretrain.add_argument("--hf-split", type=str, default="train", help="Hugging Face split or split slice, e.g. train[:10000].")
    pretrain.add_argument("--hf-sequence-field", type=str, default="Sequence", help="Column containing protein sequences.")
    pretrain.add_argument("--hf-length-field", type=str, default="Seq_Length", help="Optional length column; use '' to disable.")
    pretrain.add_argument("--hf-streaming", action="store_true", help="Stream rows instead of materializing the dataset first.")
    pretrain.add_argument("--synthetic", action="store_true", help="Use generated synthetic protein sequences.")
    pretrain.add_argument("--output-dir", type=str, default="runs/protein_jepa")
    pretrain.add_argument("--max-sequences", type=int, default=None)
    pretrain.add_argument("--synthetic-sequences", type=int, default=1024)
    pretrain.add_argument("--min-length", type=int, default=48)
    pretrain.add_argument("--max-length", type=int, default=256)
    pretrain.add_argument("--batch-size", type=int, default=16)
    pretrain.add_argument("--steps", type=int, default=100)
    pretrain.add_argument("--eval-batches", type=int, default=4)
    pretrain.add_argument("--log-interval", type=int, default=10)
    pretrain.add_argument("--seed", type=int, default=0)
    pretrain.add_argument("--embed-dim", type=int, default=192)
    pretrain.add_argument("--depth", type=int, default=4)
    pretrain.add_argument("--num-heads", type=int, default=6)
    pretrain.add_argument("--dropout", type=float, default=0.1)
    pretrain.add_argument("--mask-fraction", type=float, default=0.25)
    pretrain.add_argument("--min-span", type=int, default=4)
    pretrain.add_argument("--max-span", type=int, default=32)
    pretrain.add_argument("--lr", type=float, default=3e-4)
    pretrain.add_argument("--weight-decay", type=float, default=0.05)
    pretrain.add_argument("--ema-momentum", type=float, default=0.996)
    pretrain.add_argument("--grad-clip-norm", type=float, default=1.0)
    pretrain.add_argument("--variance-weight", type=float, default=0.01)
    pretrain.add_argument("--device", type=str, default="auto")

    probe = subparsers.add_parser("probe-secondary", help="Train a Q3 secondary-structure probe.")
    probe.add_argument("--checkpoint", type=str, default=None, help="Protein-I-JEPA checkpoint. Omit for scratch baseline.")
    probe.add_argument("--labels-tsv", type=str, default=None, help="Single TSV with sequence and labels; split randomly into train/validation.")
    probe.add_argument("--train-labels-tsv", type=str, default=None, help="Training TSV for explicit split mode.")
    probe.add_argument("--val-labels-tsv", type=str, default=None, help="Validation TSV for explicit split mode.")
    probe.add_argument(
        "--test-labels-tsv",
        type=str,
        action="append",
        default=[],
        help="External test TSV for final evaluation. Can be repeated.",
    )
    probe.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face dataset with labeled secondary-structure rows.")
    probe.add_argument("--hf-split", type=str, default="train", help="Single Hugging Face split to randomly divide into train/validation.")
    probe.add_argument("--hf-train-split", type=str, default=None, help="Hugging Face split used to train the probe.")
    probe.add_argument("--hf-val-split", type=str, default=None, help="Hugging Face split used to tune the probe.")
    probe.add_argument(
        "--hf-test-split",
        "--test-hf-split",
        dest="hf_test_splits",
        type=str,
        action="append",
        default=[],
        help="Hugging Face split used as a final external test. Can be repeated.",
    )
    probe.add_argument("--hf-sequence-field", type=str, default="sequence", help="Hugging Face field containing protein sequences.")
    probe.add_argument("--hf-label-field", type=str, default="labels", help="Hugging Face field containing per-residue labels.")
    probe.add_argument("--hf-streaming", action="store_true", help="Stream Hugging Face probe rows.")
    probe.add_argument("--hf-max-samples", type=int, default=None, help="Optional row cap per Hugging Face probe split.")
    probe.add_argument("--synthetic", action="store_true", help="Use generated synthetic Q3 labels for smoke testing.")
    probe.add_argument("--output-dir", type=str, default="runs/secondary_probe")
    probe.add_argument("--synthetic-sequences", type=int, default=256)
    probe.add_argument("--min-length", type=int, default=48)
    probe.add_argument("--max-length", type=int, default=256)
    probe.add_argument("--batch-size", type=int, default=16)
    probe.add_argument("--steps", type=int, default=100)
    probe.add_argument("--eval-batches", type=int, default=4)
    probe.add_argument("--test-eval-batches", type=int, default=None, help="Limit external test evaluation batches; default uses full test sets.")
    probe.add_argument("--log-interval", type=int, default=10)
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--lr", type=float, default=1e-3)
    probe.add_argument("--weight-decay", type=float, default=0.01)
    probe.add_argument("--unfreeze-encoder", action="store_true", help="Fine-tune the encoder instead of training a frozen probe.")
    probe.add_argument("--device", type=str, default="auto")
    probe.add_argument("--embed-dim", type=int, default=192, help="Scratch baseline encoder dimension.")
    probe.add_argument("--depth", type=int, default=4, help="Scratch baseline encoder depth.")
    probe.add_argument("--num-heads", type=int, default=6, help="Scratch baseline attention heads.")
    probe.add_argument("--dropout", type=float, default=0.1)

    embeddings = subparsers.add_parser("plot-embeddings", help="Plot predicted and target JEPA latents in 2D.")
    embeddings.add_argument("--checkpoint", type=str, required=True, help="Protein-I-JEPA checkpoint.")
    embeddings.add_argument("--fasta", type=str, default=None, help="Optional FASTA file for visualization examples.")
    embeddings.add_argument("--hf-dataset", type=str, default=None, help="Optional Hugging Face dataset name.")
    embeddings.add_argument("--hf-split", type=str, default="train")
    embeddings.add_argument("--hf-sequence-field", type=str, default="Sequence")
    embeddings.add_argument("--hf-length-field", type=str, default="Seq_Length")
    embeddings.add_argument("--hf-streaming", action="store_true")
    embeddings.add_argument("--synthetic", action="store_true", help="Use synthetic protein sequences.")
    embeddings.add_argument("--output-dir", type=str, default=None, help="Defaults to the checkpoint directory.")
    embeddings.add_argument("--max-sequences", type=int, default=None)
    embeddings.add_argument("--synthetic-sequences", type=int, default=128)
    embeddings.add_argument("--min-length", type=int, default=48)
    embeddings.add_argument("--max-length", type=int, default=None)
    embeddings.add_argument("--batch-size", type=int, default=16)
    embeddings.add_argument("--num-batches", type=int, default=8)
    embeddings.add_argument("--max-points", type=int, default=2000)
    embeddings.add_argument("--seed", type=int, default=0)
    embeddings.add_argument("--mask-fraction", type=float, default=None)
    embeddings.add_argument("--min-span", type=int, default=None)
    embeddings.add_argument("--max-span", type=int, default=None)
    embeddings.add_argument("--device", type=str, default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pretrain":
        config = TrainConfig(
            fasta=args.fasta,
            hf_dataset=args.hf_dataset,
            hf_split=args.hf_split,
            hf_sequence_field=args.hf_sequence_field,
            hf_length_field=args.hf_length_field or None,
            hf_streaming=args.hf_streaming,
            synthetic=args.synthetic,
            output_dir=args.output_dir,
            max_sequences=args.max_sequences,
            synthetic_sequences=args.synthetic_sequences,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_batches=args.eval_batches,
            log_interval=args.log_interval,
            seed=args.seed,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            dropout=args.dropout,
            mask_fraction=args.mask_fraction,
            min_span=args.min_span,
            max_span=args.max_span,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ema_momentum=args.ema_momentum,
            grad_clip_norm=args.grad_clip_norm,
            variance_weight=args.variance_weight,
            device=args.device,
        )
        train(config)
    elif args.command == "probe-secondary":
        config = ProbeConfig(
            checkpoint=args.checkpoint,
            labels_tsv=args.labels_tsv,
            train_labels_tsv=args.train_labels_tsv,
            val_labels_tsv=args.val_labels_tsv,
            test_labels_tsv=args.test_labels_tsv,
            hf_dataset=args.hf_dataset,
            hf_split=args.hf_split,
            hf_train_split=args.hf_train_split,
            hf_val_split=args.hf_val_split,
            hf_test_splits=args.hf_test_splits,
            hf_sequence_field=args.hf_sequence_field,
            hf_label_field=args.hf_label_field,
            hf_streaming=args.hf_streaming,
            hf_max_samples=args.hf_max_samples,
            synthetic=args.synthetic,
            output_dir=args.output_dir,
            synthetic_sequences=args.synthetic_sequences,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_batches=args.eval_batches,
            test_eval_batches=args.test_eval_batches,
            log_interval=args.log_interval,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            freeze_encoder=not args.unfreeze_encoder,
            device=args.device,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
        train_secondary_probe(config)
    elif args.command == "plot-embeddings":
        config = EmbeddingPlotConfig(
            checkpoint=args.checkpoint,
            fasta=args.fasta,
            hf_dataset=args.hf_dataset,
            hf_split=args.hf_split,
            hf_sequence_field=args.hf_sequence_field,
            hf_length_field=args.hf_length_field or None,
            hf_streaming=args.hf_streaming,
            synthetic=args.synthetic,
            output_dir=args.output_dir,
            max_sequences=args.max_sequences,
            synthetic_sequences=args.synthetic_sequences,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            max_points=args.max_points,
            seed=args.seed,
            mask_fraction=args.mask_fraction,
            min_span=args.min_span,
            max_span=args.max_span,
            device=args.device,
        )
        print(json.dumps(plot_embeddings(config)), flush=True)
    else:
        parser.error(f"Unknown command: {args.command}")
