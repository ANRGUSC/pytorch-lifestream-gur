# pytorch-lifestream-gur

**GUR (Generalized User Representations)** extension for [pytorch-lifestream](https://github.com/pytorch-lifestream/pytorch-lifestream).

> This repository is a fork of [pytorch-lifestream/pytorch-lifestream](https://github.com/pytorch-lifestream/pytorch-lifestream) (Apache 2.0 License). All original code and credit belongs to the pytorch-lifestream authors. We add the GUR module described below.

## What is GUR?

GUR implements the autoencoder-based user representation approach from Spotify's paper:

**"Generalized User Representations for Transfer Learning"**
Fazelnia et al., 2024 ([arXiv:2403.00584](https://arxiv.org/abs/2403.00584))

The key idea: compress multi-timescale aggregated event embeddings into compact user representations using an autoencoder trained with reconstruction loss. Unlike sequential models (Transformers, RNNs), GUR aggregates event features over configurable time windows (e.g., 1 week / 1 month / 6 months) before encoding, capturing both short-term intent and long-term preferences.

## Architecture

```
Event Sequences (PaddedBatch)
       |
  TrxEncoder          (embed categorical + numerical event features)
       |
  MultiTimescaleAggregator   (mean pool over 7d / 30d / 180d windows)
       |
  Encoder MLP (SELU)  -->  Latent Embedding (default: 120-dim)
       |                        |
  Decoder MLP (SELU)       [USER EMBEDDING]
       |                   (used for downstream tasks)
  Reconstruction
       |
  MSE Loss
```

## Interface Compatibility with CoLES

GUR is a **drop-in replacement** for CoLES. The interface is identical:

| Feature | CoLES | GUR |
|---------|-------|-----|
| Base class | `ABSModule` | `ABSModule` |
| Dataset | `ColesDataset` | `GURDataset` |
| Encoder | `SeqEncoderContainer` | `AutoEncoderSeqEncoder` |
| `forward(x)` output | `Tensor(B, E)` | `Tensor(B, E)` |
| Loss | Contrastive | MSE Reconstruction |
| Data module | `PtlsDataModule` | `PtlsDataModule` |
| Embedding extraction | `module.seq_encoder` | `module.seq_encoder` |

## New Files

| File | Description |
|------|-------------|
| `ptls/nn/seq_encoder/autoencoder_encoder.py` | `AutoEncoderSeqEncoder`, `MultiTimescaleAggregator` |
| `ptls/frames/gur/gur_module.py` | `GURModule(ABSModule)` training framework |
| `ptls/frames/gur/gur_dataset.py` | `GURDataset`, `GURIterableDataset` |
| `ptls/frames/gur/__init__.py` | Package exports |
| `demo/demo_gur.py` | End-to-end demo with synthetic data |

## Quick Start

```python
from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.seq_encoder.autoencoder_encoder import AutoEncoderSeqEncoder
from ptls.frames.gur import GURModule, GURDataset
from ptls.frames import PtlsDataModule

# 1. Configure encoder
trx_encoder = TrxEncoder(
    embeddings={'mcc_code': {'in': 100, 'out': 24}},
    numeric_values={'amount': 'identity'},
)

seq_encoder = AutoEncoderSeqEncoder(
    trx_encoder=trx_encoder,
    latent_dim=120,              # paper default
    time_windows=[7, 30, 180],   # 1 week, 1 month, 6 months
    dropout=0.1,
)

# 2. Create training module
module = GURModule(
    seq_encoder=seq_encoder,
    optimizer_partial=partial(torch.optim.Adam, lr=1e-3),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.9),
)

# 3. Train with PyTorch Lightning
dm = PtlsDataModule(
    train_data=GURDataset(train_data),
    valid_data=GURDataset(valid_data),
    train_batch_size=64,
)
trainer = pl.Trainer(max_epochs=50)
trainer.fit(module, datamodule=dm)

# 4. Extract embeddings (same API as CoLES)
embeddings = module(batch)  # Tensor(B, 120)
```

## Running the Demo

```bash
pip install -e .
python demo/demo_gur.py
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 120 | Dimension of user embedding |
| `time_windows` | [7, 30, 180] | Aggregation windows in days |
| `time_unit` | 'auto' | Auto-detects epoch seconds vs day offsets |
| `encoder_hidden_dims` | [agg_dim, agg_dim//2] | Encoder MLP hidden layers |
| `dropout` | 0.1 | AlphaDropout rate (SELU-compatible) |

## Citation

If you use this code, please cite both works:

```bibtex
@article{fazelnia2024generalized,
  title={Generalized User Representations for Transfer Learning},
  author={Fazelnia, Ghazal and Gupta, Sanket and Keum, Claire and Koh, Mark and Anderson, Ian and Lalmas, Mounia},
  journal={arXiv preprint arXiv:2403.00584},
  year={2024}
}

@article{babaev2022coles,
  title={CoLES: Contrastive Learning for Event Sequences with Self-Supervision},
  author={Babaev, Dmitrii and Ovsov, Nikita and Kireev, Ivan and Ivanov, Gleb and Burnaev, Evgeny and Babenko, Artem},
  journal={arXiv preprint arXiv:2002.08232},
  year={2022}
}
```

## License

Apache 2.0 (same as the original pytorch-lifestream).
